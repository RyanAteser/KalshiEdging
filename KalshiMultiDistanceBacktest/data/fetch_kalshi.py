"""
fetch_kalshi.py — Parameterized Kalshi tick fetcher for any 15-minute series.

For each series:
  1. Fetch settled markets (last DAYS days)
  2. For each market, fetch 1-minute Kalshi candlesticks
  3. Save all ticks (strike, ask, bid, t_left, close_ts) to parquet
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Field helpers (same as market_fetcher.py) ─────────────────────────────

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


def _get(obj, attr):
    return obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)


def _unwrap_market(m):
    """Unwrap pykalshi single-market response to the actual data dict/object."""
    md = getattr(m, "data", m)
    if isinstance(md, dict) and "market" in md:
        return md["market"]
    if hasattr(md, "market") and getattr(md, "market") is not None:
        return getattr(md, "market")
    return md


def _extract_field(m, *keys):
    md = _unwrap_market(m)
    for key in keys:
        val = _get(md, key)
        if val is not None:
            return val
        val = _get(m, key)
        if val is not None:
            return val
    return None


def _to_dollars(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return round(f / 100.0 if f > 1.0 else f, 6)


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Strike extraction — parameterized by min_strike ───────────────────────

def _extract_strike(m, ticker: str = "", min_strike: float = 0.0) -> float:
    """
    Extract asset strike price using multi-field approach.
    Subtitle → numeric fields → settlement_timer_values → title → ticker suffix.

    min_strike replaces the hardcoded >1000 guard so low-value assets
    (XRP $2, DOGE $0.20, HYPE $20) parse correctly.
    """
    md = _unwrap_market(m)

    def _parse_text(text) -> Optional[float]:
        if not text or "tbd" in str(text).lower():
            return None
        for pat in (
            r"(?:target price|above|below|over|under)[:\s]+\$?([\d,]+(?:\.\d+)?)",
            r"\$([\d,]+(?:\.\d+)?)",
        ):
            hit = re.search(pat, str(text), re.IGNORECASE)
            if hit:
                try:
                    v = float(hit.group(1).replace(",", ""))
                    if v > min_strike and v > 0:
                        return v
                except Exception:
                    pass
        return None

    # 1. subtitle
    for attr in ("yes_sub_title", "no_sub_title", "subtitle", "sub_title"):
        v = _parse_text(_get(md, attr) or _get(m, attr))
        if v:
            return v

    # 2. numeric strike fields
    for attr in ("floor_strike_dollars", "cap_strike_dollars", "reference_price",
                 "floor_strike", "strike_price", "cap_strike", "strike",
                 "settlement_price", "price_to_beat"):
        raw = _get(md, attr) or _get(m, attr)
        try:
            v = float(raw) if raw is not None else None
            if v and v > min_strike and v > 0:
                return v
        except Exception:
            pass

    # 3. settlement_timer_values
    stv = _get(md, "settlement_timer_values") or _get(m, "settlement_timer_values")
    if isinstance(stv, dict):
        for val in stv.values():
            try:
                v = float(val)
                if v > min_strike and v > 0:
                    return v
            except Exception:
                pass

    # 4. title / rules_primary
    for attr in ("title", "rules_primary"):
        v = _parse_text(_get(md, attr) or _get(m, attr))
        if v:
            return v

    # 5. ticker suffix — last resort
    upper = ticker.upper()
    for pat in [
        r"-T([\d]+(?:\.\d+)?)$",
        r"-([\d]+(?:\.\d+)?)T$",
        r"-[AB]([\d]+(?:\.\d+)?)$",
        r"[-_]([\d]+(?:\.\d+)?)(?:[-T]|$)",
    ]:
        hit = re.search(pat, upper)
        if hit:
            try:
                v = float(hit.group(1))
                if v > min_strike and v > 0:
                    return v
            except Exception:
                pass

    return 0.0


# ── Market fetching ────────────────────────────────────────────────────────

def _fetch_markets_by_series(client, series: str, start_ts: int) -> List[Any]:
    """Fetch settled markets for a series since start_ts."""
    from pykalshi.enums import MarketStatus
    markets = []
    cursor  = None

    for page in range(20):
        try:
            resp = client.get_markets(
                series_ticker = series,
                status        = MarketStatus.SETTLED,
                limit         = 200,
                cursor        = cursor,
            )
        except Exception as exc:
            logger.warning("Markets page %d failed for %s: %s", page, series, exc)
            break

        if hasattr(resp, "markets"):
            batch  = resp.markets or []
            cursor = getattr(resp, "cursor", None)
        elif isinstance(resp, (list, tuple)):
            batch  = list(resp)
            cursor = None
        elif isinstance(resp, dict):
            batch  = resp.get("markets") or []
            cursor = resp.get("cursor")
        else:
            break

        if not batch:
            break

        for m in batch:
            close_ts = _parse_ts(
                _extract_field(m, "close_ts", "close_time", "expiration_time",
                               "close_time_ts", "closeTime", "settle_time")
            )
            if close_ts >= start_ts:
                markets.append(m)

        if not cursor:
            break

    logger.info("Found %d settled %s markets since %d", len(markets), series, start_ts)
    return markets


def get_market_snapshot(client, ticker: str, min_strike: float = 0.0, retries: int = 3) -> Optional[Dict]:
    """Fetch current snapshot for a single market ticker."""
    delay = 0.5
    for attempt in range(retries):
        try:
            m = client.get_market(ticker)

            raw_bid  = _extract_field(m, "yes_bid_dollars", "yes_bid",  "bid_dollars", "bid")
            raw_ask  = _extract_field(m, "yes_ask_dollars", "yes_ask",  "ask_dollars", "ask")
            raw_last = _extract_field(m, "last_price_dollars", "last_price", "price_dollars", "price")
            raw_close = _extract_field(m, "close_ts", "close_time", "close_time_ts",
                                       "close_timestamp", "closeTime", "close")

            strike = _extract_strike(m, ticker, min_strike=min_strike)

            return {
                "ticker":     ticker,
                "best_bid":   _to_dollars(raw_bid),
                "best_ask":   _to_dollars(raw_ask),
                "last_price": _to_dollars(raw_last),
                "strike":     strike,
                "close_ts":   _parse_ts(raw_close),
            }
        except Exception as exc:
            s = str(exc).lower()
            if "429" in s or "too many" in s:
                time.sleep(delay)
                delay = min(delay * 2.0, 5.0)
            elif attempt < retries - 1:
                time.sleep(0.3)
            else:
                return None
    return None


def fetch_market_snapshots(client, markets, ticks_per_market: int = 10, min_strike: float = 0.0):
    """Fetch multiple snapshots per market (for live use). Not used in backtest pipeline."""
    results = []
    for m in markets:
        ticker = _get(m, "ticker") if not isinstance(m, str) else m
        if not ticker:
            continue
        snap = get_market_snapshot(client, ticker, min_strike=min_strike)
        if snap:
            results.append(snap)
    return results


# ── Candlestick-based tick building (used in backtest pipeline) ────────────

def _safe_dollars_candle(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _build_ticks_from_candles(
    ticker: str,
    strike: float,
    close_ts: int,
    candles,
) -> List[dict]:
    """Convert 1-minute Kalshi candlesticks to tick rows."""
    rows = []
    for c in candles:
        ts = getattr(c, "end_period_ts", None) or _get(c, "end_period_ts")
        if ts is None:
            continue

        price_obj = getattr(c, "price", None) or _get(c, "price")
        bid_obj   = getattr(c, "yes_bid",  None) or _get(c, "yes_bid")
        ask_obj   = getattr(c, "yes_ask",  None) or _get(c, "yes_ask")

        def _cd(obj):
            if obj is None:
                return None
            cd = getattr(obj, "close_dollars", None) or _get(obj, "close_dollars")
            return _safe_dollars_candle(cd)

        price = _cd(price_obj)
        bid   = _cd(bid_obj)
        ask   = _cd(ask_obj)

        if price is None:
            continue

        # Fill missing bid/ask
        if bid is None and ask is not None:
            bid = round(ask - 0.02, 4)
        elif ask is None and bid is not None:
            ask = round(bid + 0.02, 4)
        elif bid is None and ask is None:
            bid = round(price - 0.01, 4)
            ask = round(price + 0.01, 4)

        t_left = max(0, close_ts - int(ts))

        rows.append({
            "ticker":   ticker,
            "tick_time": int(ts),
            "strike":   strike,
            "price":    price,
            "ask":      ask,
            "bid":      bid,
            "t_left":   t_left,
            "close_ts": close_ts,
        })

    return rows


# ── Main save function ─────────────────────────────────────────────────────

def save_kalshi_series(asset_cfg: dict, days: int = 30, out_dir: str = "data") -> pd.DataFrame:
    """
    Fetch settled Kalshi markets for a series + their candlestick ticks.
    Saves to parquet keyed by series name.
    """
    from pykalshi import KalshiClient
    from pykalshi.enums import CandlestickPeriod

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    series     = asset_cfg["kalshi_series"]
    min_strike = asset_cfg["min_strike"]
    out_path   = Path(out_dir) / f"kalshi_{series.lower()}.parquet"

    try:
        client = KalshiClient.from_env()
    except Exception as exc:
        logger.error("Kalshi client init failed: %s", exc)
        raise

    now_ts   = int(datetime.now(timezone.utc).timestamp())
    start_ts = now_ts - days * 86_400

    markets  = _fetch_markets_by_series(client, series, start_ts)
    if not markets:
        logger.warning("No settled %s markets found", series)
        return pd.DataFrame()

    all_ticks: List[dict] = []
    skipped = 0

    for i, m in enumerate(markets):
        ticker = _get(m, "ticker") if not isinstance(m, str) else m
        if not ticker:
            continue

        close_ts = _parse_ts(
            _extract_field(m, "close_ts", "close_time", "expiration_time",
                           "close_time_ts", "closeTime", "settle_time")
        )
        strike = _extract_strike(m, ticker, min_strike=min_strike)

        if strike == 0.0:
            logger.debug("[%s] strike=0 — skip", ticker)
            skipped += 1
            continue

        if i > 0 and i % 50 == 0:
            logger.info("  Progress: %d/%d markets (%d ticks so far)", i, len(markets), len(all_ticks))

        # Fetch 1m candlesticks for this market
        mkt_start = close_ts - 960   # 16 min buffer
        mkt_end   = close_ts + 60

        try:
            mkt_obj = client.get_market(ticker)
            resp    = mkt_obj.get_candlesticks(mkt_start, mkt_end, CandlestickPeriod.ONE_MINUTE)
            candles = resp.candlesticks
        except Exception as exc:
            logger.debug("[%s] candlestick fetch failed: %s", ticker, exc)
            skipped += 1
            time.sleep(0.3)
            continue

        if not candles:
            skipped += 1
            continue

        ticks = _build_ticks_from_candles(ticker, strike, close_ts, candles)
        all_ticks.extend(ticks)
        time.sleep(0.12)

    if not all_ticks:
        logger.warning("No ticks collected for %s", series)
        return pd.DataFrame()

    df_new = pd.DataFrame(all_ticks)
    df_new["tick_time"] = pd.to_datetime(df_new["tick_time"], unit="s", utc=True)
    df_new["close_ts"]  = pd.to_datetime(df_new["close_ts"],  unit="s", utc=True)
    df_new = df_new.drop_duplicates(["ticker", "tick_time"]).sort_values("tick_time").reset_index(drop=True)

    # Merge with existing cache
    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        for col in ("tick_time", "close_ts"):
            if col in df_old.columns:
                df_old[col] = pd.to_datetime(df_old[col], utc=True)
        df_merged = (
            pd.concat([df_old, df_new], ignore_index=True)
            .drop_duplicates(["ticker", "tick_time"])
            .sort_values("tick_time")
            .reset_index(drop=True)
        )
    else:
        df_merged = df_new

    df_merged.to_parquet(out_path, index=False)
    logger.info(
        "Saved %d ticks for %s (skipped %d markets) → %s",
        len(df_merged), series, skipped, out_path,
    )
    print(f"  Saved {len(df_merged):,} ticks → {out_path}  (skipped {skipped} markets)")
    return df_merged
