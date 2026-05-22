#!/usr/bin/env python3
"""
backtest.py — EV Grid Filter historical backtest on KXBTC15M markets.

Usage:
    cd /home/user/KalshiEdging
    python scripts/backtest.py
    python scripts/backtest.py --days 14 --min-ev 0.010
    python scripts/backtest.py --days 30 --grid-min 0.55 --grid-max 0.75
    python scripts/backtest.py --days 7 --verbose

Data:
  - Settled KXBTC15M markets + 1m candlestick history from Kalshi API
  - Binance BTC/USDT 1m klines for CVD, cross-asset, and 15m candle signals

Limitations:
  - Fills simulated at candle close prices (no intra-candle slippage)
  - time_pressure feature returns 0 for historical markets (close_ts is in the past)
  - 1m resolution only — no sub-minute tick data

Output:
  - Console summary: trades, win rate, PnL, Sharpe, feature correlations
  - ev_backtest_results.csv: per-trade record for further analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Bootstrap path so we can import core/ ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pykalshi import KalshiClient
from pykalshi.enums import MarketStatus, CandlestickPeriod

from core.signal_engine_ev import EVSignalEngine
from core.models import SignalType

logger = logging.getLogger("backtest")

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BTC_SERIES         = "KXBTC15M"


# ── Config stub ───────────────────────────────────────────────────────────

class BacktestConfig:
    def __init__(self, args):
        self.ev_grid_min  = args.grid_min
        self.ev_grid_max  = args.grid_max
        self.ev_min_entry = args.min_ev
        self.ev_min_exit  = args.min_exit_ev
        self.ev_fee_rate  = args.fee_rate


# ── Mock data feeds ───────────────────────────────────────────────────────

@dataclass
class MockCandle:
    open_p:  float
    high_p:  float
    low_p:   float
    close_p: float

    @property
    def is_bullish(self) -> bool:
        return self.close_p >= self.open_p

    @property
    def body_size(self) -> float:
        return abs(self.close_p - self.open_p)

    @property
    def range_size(self) -> float:
        return max(self.high_p - self.low_p, 1e-8)


class MockBinanceFeed:
    """Replays Binance spot data from pre-loaded 1m klines."""

    def __init__(self, klines: Dict[int, dict]) -> None:
        self._klines = klines
        self._current: Optional[dict] = None

    def set_time(self, ts: int) -> None:
        minute = (ts // 60) * 60
        self._current = self._klines.get(minute)

    @property
    def mid_price(self) -> Optional[float]:
        return self._current["close"] if self._current else None

    @property
    def cvd(self) -> float:
        if not self._current:
            return 0.0
        vol = self._current.get("volume", 0)
        tb  = self._current.get("taker_buy", vol / 2)
        if vol <= 0:
            return 0.0
        # Normalize net taker buying to [-1, 1]
        return max(-1.0, min(1.0, (2.0 * tb - vol) / vol))


class MockBtcFeed:
    """Replays Coinbase-style 15m candles derived from Binance klines."""

    def __init__(self, candles_15m: Dict[int, MockCandle]) -> None:
        self._candles    = candles_15m
        self._current_ts = 0

    def set_time(self, ts: int) -> None:
        self._current_ts = ts

    @property
    def latest_candles(self) -> List[MockCandle]:
        # Return the most recent COMPLETED 15m candle prior to current_ts
        bucket    = (self._current_ts // 900) * 900
        completed = bucket - 900
        for b in range(completed, completed - 5 * 900, -900):
            if b in self._candles:
                return [self._candles[b]]
        return []


class MockBinanceFuturesFeed:
    """Stub futures feed — OI and funding rate not available historically."""

    @property
    def mark_price(self) -> Optional[float]:
        return None

    @property
    def funding_rate(self) -> float:
        return 0.0

    @property
    def oi_delta(self) -> float:
        return 0.0


# ── Per-trade result ──────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    ticker:      str
    btc_target:  float
    side:        str
    entry_ts:    int
    entry_price: float
    exit_ts:     Optional[int]
    exit_price:  float
    exit_reason: str    # stop_loss | ev_flip | settlement
    pnl:         float  # per contract
    outcome:     int    # 1 = win, 0 = loss
    features:    dict   = field(default_factory=dict)


# ── Data fetching ─────────────────────────────────────────────────────────

def _parse_ts(val) -> int:
    if val is None:
        return 0
    try:
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val)
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S"):
            try:
                return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                pass
    except Exception:
        pass
    return 0


def fetch_settled_markets(
    client: KalshiClient, start_ts: int, max_markets: int
) -> List[Dict[str, Any]]:
    """Return list of {ticker, close_ts} for settled KXBTC15M markets."""
    markets: List[dict] = []
    cursor = None

    while len(markets) < max_markets:
        try:
            resp = client.get_markets(
                series_ticker=BTC_SERIES,
                status=MarketStatus.SETTLED,
                limit=200,
                cursor=cursor,
            )
        except Exception as exc:
            logger.warning("Markets fetch error: %s", exc)
            break

        if hasattr(resp, "markets"):
            batch  = resp.markets or []
            cursor = getattr(resp, "cursor", None)
        elif isinstance(resp, (list, tuple)):
            batch  = list(resp)
            cursor = None
        else:
            break

        if not batch:
            break

        for m in batch:
            def _g(obj, *keys):
                for k in keys:
                    v = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
                    if v is not None:
                        return v
                return None

            ticker = _g(m, "ticker")
            if not ticker:
                continue

            close_ts = _parse_ts(
                _g(m, "close_ts", "close_time", "expiration_time",
                   "close_time_ts", "closeTime", "settle_time")
            )

            if close_ts and close_ts >= start_ts:
                markets.append({"ticker": ticker, "close_ts": close_ts})

        if not cursor:
            break

    logger.info("Found %d settled KXBTC15M markets (capped at %d)", len(markets), max_markets)
    return markets[:max_markets]


def fetch_binance_klines(start_ms: int, end_ms: int) -> Dict[int, dict]:
    """
    Download all Binance BTC/USDT 1m klines for [start_ms, end_ms].
    Returns {unix_second_ts: {open, high, low, close, volume, taker_buy}}.
    """
    result: Dict[int, dict] = {}
    current = start_ms

    while current < end_ms:
        batch_end = min(current + 1_000 * 60_000, end_ms)
        url = (
            f"{BINANCE_KLINES_URL}?symbol=BTCUSDT&interval=1m"
            f"&startTime={current}&endTime={batch_end}&limit=1000"
        )
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                klines = json.loads(resp.read())
        except Exception as exc:
            logger.warning("Binance klines fetch failed: %s", exc)
            break

        if not klines:
            break

        for k in klines:
            ts = k[0] // 1000  # ms → s
            result[ts] = {
                "open":      float(k[1]),
                "high":      float(k[2]),
                "low":       float(k[3]),
                "close":     float(k[4]),
                "volume":    float(k[5]),
                "taker_buy": float(k[9]),
            }

        current = klines[-1][0] + 60_000
        time.sleep(0.05)

    return result


def build_15m_candles(btc_klines: Dict[int, dict]) -> Dict[int, MockCandle]:
    """Aggregate 1m BTC klines into 15m MockCandle objects."""
    buckets: Dict[int, List[dict]] = defaultdict(list)
    for ts, data in btc_klines.items():
        buckets[(ts // 900) * 900].append(data)

    candles: Dict[int, MockCandle] = {}
    for bucket, bars in buckets.items():
        if len(bars) < 10:
            continue
        candles[bucket] = MockCandle(
            open_p  = bars[0]["open"],
            high_p  = max(b["high"] for b in bars),
            low_p   = min(b["low"]  for b in bars),
            close_p = bars[-1]["close"],
        )
    return candles


def extract_btc_target(ticker: str) -> Optional[float]:
    """Parse BTC target price from ticker e.g. KXBTC15M-23OCT0314-T64000 → 64000."""
    m = re.search(r"-T(\d+(?:\.\d+)?)$", ticker.upper())
    return float(m.group(1)) if m else None


def _safe_dollars(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ── Per-market backtest ───────────────────────────────────────────────────

def run_market_backtest(
    ticker:         str,
    btc_target:     float,
    close_ts:       int,
    kalshi_candles: list,
    btc_klines:     Dict[int, dict],
    candles_15m:    Dict[int, MockCandle],
    config:         BacktestConfig,
) -> List[BacktestTrade]:
    """Replay one KXBTC15M market through the EV engine. Returns any trades."""
    trades: List[BacktestTrade] = []

    binance_feed = MockBinanceFeed(btc_klines)
    btc_feed     = MockBtcFeed(candles_15m)
    futures_feed = MockBinanceFuturesFeed()

    engine    = EVSignalEngine(config, btc_feed, binance_feed, futures_feed)
    market_id = abs(hash(ticker)) % 1_000_000

    # Create state and set BTC 15m context
    engine.get_or_create_state(ticker, market_id)
    engine.update_market_context(ticker, btc_target, float(close_ts))

    current_trade: Optional[BacktestTrade] = None

    for candle in kalshi_candles:
        ts = candle.end_period_ts

        price = _safe_dollars(candle.price.close_dollars)
        bid   = _safe_dollars(candle.yes_bid.close_dollars  if candle.yes_bid  else None)
        ask   = _safe_dollars(candle.yes_ask.close_dollars  if candle.yes_ask  else None)
        vol   = float(candle.volume_fp or 0)

        if price is None:
            continue
        # Fill missing bid/ask with reasonable estimates
        if bid is None and ask is not None:
            bid = round(ask - 0.02, 4)
        elif ask is None and bid is not None:
            ask = round(bid + 0.02, 4)
        elif bid is None and ask is None:
            bid = round(price - 0.01, 4)
            ask = round(price + 0.01, 4)

        binance_feed.set_time(ts)
        btc_feed.set_time(ts)

        sig = engine.process_tick(ticker, market_id, price, bid, ask, vol)
        if sig is None:
            continue

        if sig.signal_type == SignalType.ENTRY and current_trade is None:
            feats = engine.get_last_features(ticker) or {}
            side  = feats.get("side", "YES")
            engine.mark_position_open(ticker, market_id, sig.price, side=side)
            current_trade = BacktestTrade(
                ticker      = ticker,
                btc_target  = btc_target,
                side        = side,
                entry_ts    = ts,
                entry_price = sig.price,
                exit_ts     = None,
                exit_price  = 0.0,
                exit_reason = "",
                pnl         = 0.0,
                outcome     = 0,
                features    = {k: v for k, v in feats.items()
                               if k not in ("side", "entry_price", "ev")},
            )

        elif sig.signal_type in (SignalType.EXIT, SignalType.STOP_LOSS) and current_trade:
            reason = (
                "stop_loss" if sig.signal_type == SignalType.STOP_LOSS
                else (sig.metadata or {}).get("reason", "ev_flip")
            )
            _close_trade(current_trade, sig.price, ts, reason)
            trades.append(current_trade)
            current_trade = None
            engine.mark_position_closed(ticker)

    # ── Handle position still open at market close (settlement) ──────────
    if current_trade:
        settled_yes = _determine_settlement(close_ts, btc_target, btc_klines, kalshi_candles)
        if current_trade.side == "YES":
            exit_price = 1.0 if settled_yes else 0.0
        else:
            exit_price = 1.0 if not settled_yes else 0.0
        _close_trade(current_trade, exit_price, close_ts, "settlement")
        trades.append(current_trade)

    return trades


def _close_trade(
    trade: BacktestTrade, exit_price: float, ts: int, reason: str
) -> None:
    trade.exit_ts     = ts
    trade.exit_price  = exit_price
    trade.exit_reason = reason
    trade.pnl         = round(exit_price - trade.entry_price, 6)
    trade.outcome     = 1 if trade.pnl > 0 else 0


def _determine_settlement(
    close_ts: int,
    btc_target: float,
    btc_klines: Dict[int, dict],
    kalshi_candles: list,
) -> bool:
    """
    Return True if YES resolved (BTC was above target at close).
    First tries Binance klines at close_ts; falls back to last Kalshi price.
    """
    close_minute = (close_ts // 60) * 60
    for offset in (0, -60, 60, -120, 120, -180, 180):
        entry = btc_klines.get(close_minute + offset)
        if entry:
            return entry["close"] > btc_target

    # Fallback: last Kalshi candle price near 1.0 → YES, near 0.0 → NO
    if kalshi_candles:
        last_price = _safe_dollars(kalshi_candles[-1].price.close_dollars)
        if last_price is not None:
            return last_price >= 0.5
    return False


# ── Results ───────────────────────────────────────────────────────────────

def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-10 or dy < 1e-10:
        return 0.0
    return num / (dx * dy)


def write_csv(trades: List[BacktestTrade], path: str) -> None:
    if not trades:
        return
    feat_keys = sorted(
        {k for t in trades for k in t.features}
        - {"side", "entry_price", "ev"}
    )
    base_cols = ["ticker", "btc_target", "side", "entry_ts", "entry_price",
                 "exit_ts", "exit_price", "exit_reason", "pnl", "outcome"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(base_cols + feat_keys)
        for t in trades:
            row = [
                t.ticker, t.btc_target, t.side, t.entry_ts, t.entry_price,
                t.exit_ts, t.exit_price, t.exit_reason, t.pnl, t.outcome,
            ] + [t.features.get(k, "") for k in feat_keys]
            w.writerow(row)


def print_results(trades: List[BacktestTrade], days: int, args) -> None:
    W = 56
    print(f"\n{'='*W}")
    print(f"  BACKTEST  |  {days}d  |  KXBTC15M  |  EV Grid Filter")
    print(f"  grid=[{args.grid_min:.2f},{args.grid_max:.2f}]  min_ev={args.min_ev}  "
          f"min_exit_ev={args.min_exit_ev}")
    print(f"{'='*W}")

    if not trades:
        print("  No trades generated.")
        print(f"{'='*W}\n")
        return

    wins      = sum(1 for t in trades if t.outcome == 1)
    total     = len(trades)
    pnls      = [t.pnl for t in trades]
    total_pnl = sum(pnls)
    avg_pnl   = total_pnl / total

    # Sharpe — annualised, assuming ~96 markets/day as base rate
    if len(pnls) > 1:
        variance = sum((p - avg_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(variance)
        sharpe = (avg_pnl / std * math.sqrt(96 * 365)) if std > 1e-10 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    cum = peak = max_dd = 0.0
    for p in pnls:
        cum += p
        peak  = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    print(f"  Trades:          {total:>7d}")
    print(f"  Win rate:        {wins / total * 100:>7.1f}%")
    print(f"  Total PnL:       {total_pnl:>+8.4f}  (per contract)")
    print(f"  Avg PnL/trade:   {avg_pnl:>+8.5f}")
    print(f"  Sharpe (ann):    {sharpe:>7.2f}")
    print(f"  Max drawdown:   {-max_dd:>+8.4f}")

    # By exit reason
    by_reason: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        by_reason[t.exit_reason].append(t.pnl)
    print(f"\n  Exit breakdown:")
    for reason, ps in sorted(by_reason.items()):
        w = sum(1 for p in ps if p > 0)
        print(f"    {reason:<15}  {len(ps):>4} trades  "
              f"{w / len(ps) * 100:>5.1f}% win  {sum(ps) / len(ps):>+.5f} avg")

    # Feature correlations with outcome
    feat_keys = sorted(
        {k for t in trades for k in t.features}
        - {"p_model", "ev", "side", "entry_price"}
    )
    corrs = []
    for key in feat_keys:
        vals = [t.features.get(key, 0.0) or 0.0 for t in trades]
        outs = [float(t.outcome) for t in trades]
        corr = _pearson(vals, outs)
        corrs.append((abs(corr), key, corr))

    print(f"\n  Feature → outcome correlations (Pearson):")
    for _, key, corr in sorted(corrs, reverse=True):
        stars = "★" * max(0, min(5, int(abs(corr) * 25)))
        print(f"    {key:<24}  {corr:>+.4f}  {stars}")

    print(f"\n  NOTE: time_pressure not simulated (historical markets already closed)")
    print(f"{'='*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EV Grid Filter backtest on KXBTC15M",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days",         type=int,   default=30,
                        help="Days of history to backtest")
    parser.add_argument("--min-ev",       type=float, default=0.005,
                        help="Min EV to open a position")
    parser.add_argument("--min-exit-ev",  type=float, default=-0.003,
                        help="Auto-exit when EV drops below this")
    parser.add_argument("--grid-min",     type=float, default=0.50,
                        help="Grid lower price bound")
    parser.add_argument("--grid-max",     type=float, default=0.80,
                        help="Grid upper price bound")
    parser.add_argument("--fee-rate",     type=float, default=0.007,
                        help="Fee rate for EV formula")
    parser.add_argument("--max-markets",  type=int,   default=500,
                        help="Max settled markets to test")
    parser.add_argument("--output",       default="ev_backtest_results.csv",
                        help="Output CSV path")
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    config   = BacktestConfig(args)
    now_ts   = int(time.time())
    start_ts = now_ts - args.days * 86400
    start_ms = start_ts * 1000
    end_ms   = now_ts * 1000

    # ── Binance klines ────────────────────────────────────────────────
    logger.info("Fetching %d days of Binance BTC/USDT 1m klines...", args.days)
    btc_klines  = fetch_binance_klines(start_ms, end_ms)
    candles_15m = build_15m_candles(btc_klines)
    logger.info("Binance: %d 1m bars, %d 15m candles", len(btc_klines), len(candles_15m))

    # ── Kalshi settled markets ────────────────────────────────────────
    logger.info("Fetching settled KXBTC15M markets (max %d)...", args.max_markets)
    client  = KalshiClient.from_env()
    markets = fetch_settled_markets(client, start_ts, args.max_markets)

    if not markets:
        logger.error("No settled markets found for the period. "
                     "Try --days with a smaller value or check your API key.")
        sys.exit(1)

    # ── Replay ───────────────────────────────────────────────────────
    all_trades: List[BacktestTrade] = []
    skipped = 0

    for i, m in enumerate(markets):
        ticker     = m["ticker"]
        close_ts   = m["close_ts"]
        btc_target = extract_btc_target(ticker)

        if not btc_target:
            skipped += 1
            continue

        if i > 0 and i % 50 == 0:
            logger.info(
                "Progress: %d/%d markets  trades=%d  skipped=%d",
                i, len(markets), len(all_trades), skipped,
            )

        # Fetch 1m Kalshi candles for this market
        mkt_start = close_ts - 960   # 16 min back for buffer
        mkt_end   = close_ts + 60

        try:
            mkt     = client.get_market(ticker)
            resp    = mkt.get_candlesticks(mkt_start, mkt_end, CandlestickPeriod.ONE_MINUTE)
            candles = resp.candlesticks
        except Exception as exc:
            logger.debug("[%s] Candle fetch failed: %s", ticker, exc)
            skipped += 1
            time.sleep(0.3)
            continue

        if not candles:
            skipped += 1
            continue

        trades = run_market_backtest(
            ticker         = ticker,
            btc_target     = btc_target,
            close_ts       = close_ts,
            kalshi_candles = candles,
            btc_klines     = btc_klines,
            candles_15m    = candles_15m,
            config         = config,
        )
        all_trades.extend(trades)
        time.sleep(0.12)   # stay well under Kalshi rate limit

    logger.info(
        "Done: %d markets  %d trades  %d skipped",
        len(markets), len(all_trades), skipped,
    )

    if all_trades:
        write_csv(all_trades, args.output)
        logger.info("Per-trade CSV: %s", args.output)

    print_results(all_trades, args.days, args)


if __name__ == "__main__":
    main()
