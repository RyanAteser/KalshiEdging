"""
trader.py — Live buy-to-settle trader for Kalshi BTC 15m up/down markets.

Strategy:
  In the last 5-89 seconds of each market:
    - Compute z-score (BTC vs open price, normalized by realized vol)
    - If z > Z_THRESHOLD: buy YES at ask
    - If z < -Z_THRESHOLD: buy NO at ask
    - Hold to settlement — never exit early

Run:
  python -m live.trader            # paper mode (default)
  python -m live.trader --live     # REAL MONEY — be sure
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from live.config import (
    Z_THRESHOLD, T_ENTER_MAX, T_ENTER_MIN,
    CONTRACTS, MAX_OPEN,
    SCAN_INTERVAL_SEC, BTC_HISTORY_LEN,
    PAPER_MODE, MAX_DAILY_LOSS,
    EWMA_DECAY, MIN_SIGMA_FLOOR,
)
from live.price_feed import BinancePriceFeed
from live.market_scanner import scan_active_markets, cache_open_price
from core.z_score import compute_z_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trader")


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _fee(entry_dollars: float) -> float:
    """Kalshi fee per side: 7% of potential profit, capped at $0.07."""
    return min(0.07, 0.07 * (1.0 - entry_dollars))


class LiveTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None
        self.feed = BinancePriceFeed(history_len=BTC_HISTORY_LEN)

        self.open_positions: dict[str, dict] = {}   # ticker → position
        self.daily_pnl: float = 0.0
        self.trade_log: list[dict] = []

        # Track open_price per market (keyed by close_ts)
        self._open_prices: dict[int, float] = {}    # close_ts → BTC at open

    # ── Setup ─────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        from pykalshi import KalshiClient
        self.client = KalshiClient.from_env()
        log.info("Kalshi connected  [%s]", "PAPER" if self.paper else "*** LIVE ***")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("Starting BTC feed...")
        self.feed.start()

        log.info("Connecting to Kalshi...")
        self.connect()

        log.info("=" * 60)
        log.info("TRADER RUNNING  z≥%.1f  window=%d-%ds  contracts=%d  mode=%s",
                 Z_THRESHOLD, T_ENTER_MIN, T_ENTER_MAX,
                 CONTRACTS, "PAPER" if self.paper else "LIVE")
        log.info("=" * 60)

        last_scan = 0.0

        while True:
            now = time.time()

            # ── Record open prices for markets about to start ─────────────
            btc_price, btc_history = self.feed.get()
            if btc_price is None:
                time.sleep(1)
                continue

            # Cache the current BTC price keyed to the next 15m boundary
            # (used as open_price when that market enters its last 90s)
            next_boundary = ((_now_ts() // 900) + 1) * 900
            if next_boundary not in self._open_prices:
                self._open_prices[next_boundary] = btc_price
                cache_open_price(next_boundary, btc_price)

            # ── Scan for markets entering our window ──────────────────────
            if now - last_scan >= SCAN_INTERVAL_SEC:
                last_scan = now
                self._scan_and_enter(btc_price, btc_history)

            # ── Check daily loss limit ────────────────────────────────────
            if self.daily_pnl < -MAX_DAILY_LOSS:
                log.warning("Daily loss limit hit ($%.2f) — stopping.", self.daily_pnl)
                break

            time.sleep(1)

    # ── Scanner + entry ───────────────────────────────────────────────────────

    def _scan_and_enter(self, btc_price: float, btc_history: list[float]) -> None:
        if len(self.open_positions) >= MAX_OPEN:
            return

        markets = scan_active_markets(self.client, btc_price)

        for mkt in markets:
            ticker   = mkt["ticker"]
            t_left   = mkt["t_left"]
            ask      = mkt["ask"]       # YES ask (0-1 fraction)
            open_price = mkt["open_price"]

            if ticker in self.open_positions:
                continue
            if t_left < T_ENTER_MIN or t_left > T_ENTER_MAX:
                continue
            if open_price <= 0 or open_price == btc_price:
                # open_price fell back to current price — not reliable, skip
                continue

            z, sigma = compute_z_score(
                btc_price       = btc_price,
                strike          = open_price,
                t_remaining_sec = t_left,
                price_history   = btc_history,
                ewma_decay      = EWMA_DECAY,
                min_sigma       = MIN_SIGMA_FLOOR,
            )

            if z >= Z_THRESHOLD:
                self._enter(ticker, "YES", ask, z, t_left, open_price, btc_price)
            elif z <= -Z_THRESHOLD:
                no_ask = 1.0 - mkt["bid"]   # cost to buy NO = 1 - YES bid
                self._enter(ticker, "NO", no_ask, z, t_left, open_price, btc_price)

    def _enter(
        self,
        ticker: str,
        side: str,
        ask: float,
        z: float,
        t_left: float,
        open_price: float,
        btc_price: float,
    ) -> None:
        entry_dollars = ask
        fee = _fee(entry_dollars) * 2 * CONTRACTS
        expected_profit = (1.0 - entry_dollars) * CONTRACTS - fee

        log.info(
            "ENTRY  %-40s  side=%-3s  ask=%.3f  z=%+.2f  t=%ds  "
            "contracts=%d  expected=+$%.2f",
            ticker, side, ask, z, int(t_left), CONTRACTS, expected_profit,
        )

        if not self.paper:
            self._place_order(ticker, side, ask)

        self.open_positions[ticker] = {
            "side":        side,
            "entry_ask":   ask,
            "z":           z,
            "t_left":      t_left,
            "open_price":  open_price,
            "btc_at_entry": btc_price,
            "contracts":   CONTRACTS,
            "entry_time":  _now_ts(),
            "expected_profit": expected_profit,
        }

    def _place_order(self, ticker: str, side: str, ask: float) -> None:
        try:
            # Kalshi order: buy YES contracts at limit price
            price_cents = int(round(ask * 100))
            action = "buy"
            yes_price = price_cents if side == "YES" else (100 - price_cents)

            self.client.create_order(
                ticker=ticker,
                action=action,
                side="yes",
                type="limit",
                count=CONTRACTS,
                yes_price=yes_price,
            )
            log.info("  ORDER PLACED: %s %s × %d @ %dc", action, ticker, CONTRACTS, yes_price)
        except Exception as exc:
            log.error("  ORDER FAILED: %s", exc)

    # ── Settlement tracking ────────────────────────────────────────────────────

    def settle_position(self, ticker: str, outcome: int) -> None:
        pos = self.open_positions.pop(ticker, None)
        if pos is None:
            return

        side      = pos["side"]
        entry_ask = pos["entry_ask"]
        contracts = pos["contracts"]

        won = (side == "YES" and outcome == 1) or (side == "NO" and outcome == 0)
        entry_dollars = entry_ask if side == "YES" else (1.0 - entry_ask)
        fee = _fee(entry_dollars) * 2 * contracts

        if won:
            gross = (1.0 - entry_dollars) * contracts
            pnl   = gross - fee
        else:
            pnl = -entry_dollars * contracts - fee

        self.daily_pnl += pnl
        self.trade_log.append({**pos, "outcome": outcome, "won": won, "pnl": pnl})

        log.info(
            "SETTLE %-40s  side=%-3s  %s  pnl=$%+.2f  daily=$%+.2f",
            ticker, side, "WIN" if won else "LOSS", pnl, self.daily_pnl,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Place real orders (default: paper mode)")
    args = parser.parse_args()

    paper = not args.live
    if not paper:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE MODE — real money will be risked")
        print("!" * 60)
        confirm = input("  Type YES to confirm: ")
        if confirm.strip() != "YES":
            print("  Aborted.")
            return

    trader = LiveTrader(paper=paper)
    try:
        trader.run()
    except KeyboardInterrupt:
        log.info("Stopped by user.  Daily PnL: $%+.2f", trader.daily_pnl)
        if trader.trade_log:
            import pandas as pd
            pd.DataFrame(trader.trade_log).to_csv("trade_log.csv", index=False)
            log.info("Trade log saved → trade_log.csv")


if __name__ == "__main__":
    main()
