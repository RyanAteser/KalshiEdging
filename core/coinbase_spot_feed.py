"""
coinbase_spot_feed.py — Coinbase BTC-USD spot feed via REST polling.

Uses the same api.exchange.coinbase.com domain as btc_feed, avoiding the
DNS/geo-block issues with the Advanced Trade WebSocket endpoint.

Provides:
  - mid_price  — (bid + ask) / 2 from the ticker endpoint, polled every 2s
  - cvd        — rolling CVD from the trades endpoint, polled every 10s
  - is_connected

Drop-in replacement for BinanceFeed: same public interface.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL            = "https://api.exchange.coinbase.com"
TICKER_URL          = f"{BASE_URL}/products/BTC-USD/ticker"
TRADES_URL          = f"{BASE_URL}/products/BTC-USD/trades?limit=100"
TICKER_POLL         = 2.0    # seconds between ticker polls
TRADES_POLL         = 10.0   # seconds between trade-history polls
CVD_WINDOW          = 200    # rolling trade count for CVD
MAX_ERRORS          = 20
RECONNECT_DELAY     = 5.0

_instance: Optional["CoinbaseSpotFeed"] = None
_instance_lock = threading.Lock()


def get_instance() -> "CoinbaseSpotFeed":
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = CoinbaseSpotFeed()
            _instance.start()
    return _instance


class CoinbaseSpotFeed(threading.Thread):
    """
    Background thread that REST-polls Coinbase Exchange for BTC-USD spot data.

    ticker endpoint → bid/ask/mid price updated every TICKER_POLL seconds.
    trades endpoint → rolling CVD updated every TRADES_POLL seconds.
      side='buy'  → taker buy  (bullish) → +size
      side='sell' → taker sell (bearish) → -size
    New trades are identified by trade_id to avoid double-counting on repeated polls.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True, name="coinbase-spot-feed")
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._error_count = 0

        self._bid: Optional[float] = None
        self._ask: Optional[float] = None
        self._mid: Optional[float] = None

        self._trade_qtys: deque = deque(maxlen=CVD_WINDOW)
        self._cvd_sum: float = 0.0
        self._last_trade_id: Optional[int] = None

    # ── Public properties ─────────────────────────────────────────────

    @property
    def mid_price(self) -> Optional[float]:
        with self._lock:
            return self._mid

    @property
    def cvd(self) -> float:
        """
        Cumulative Volume Delta over last CVD_WINDOW trades.
        Normalized to [-1, 1].
        """
        with self._lock:
            if not self._trade_qtys:
                return 0.0
            abs_sum = sum(abs(q) for q in self._trade_qtys)
            return self._cvd_sum / abs_sum if abs_sum > 0 else 0.0

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._mid is not None

    # ── Thread lifecycle ──────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("CoinbaseSpotFeed started (REST polling)")
        last_trades_poll = 0.0

        while not self._stop_event.is_set():
            if self._error_count >= MAX_ERRORS:
                logger.error("CoinbaseSpotFeed stopping — too many errors (%d)", MAX_ERRORS)
                break

            now = time.time()

            try:
                self._poll_ticker()
                self._error_count = 0
            except Exception as exc:
                self._error_count += 1
                logger.warning("CoinbaseSpotFeed ticker poll failed (%d/%d): %s",
                               self._error_count, MAX_ERRORS, exc)

            if now - last_trades_poll >= TRADES_POLL:
                try:
                    self._poll_trades()
                    last_trades_poll = now
                except Exception as exc:
                    logger.debug("CoinbaseSpotFeed trades poll failed: %s", exc)

            self._stop_event.wait(TICKER_POLL)

        logger.info("CoinbaseSpotFeed stopped")

    # ── REST polling ──────────────────────────────────────────────────

    def _fetch_json(self, url: str):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; btc-trader/1.0)",
                "Accept":     "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())

    def _poll_ticker(self) -> None:
        data = self._fetch_json(TICKER_URL)
        bid  = float(data["bid"])
        ask  = float(data["ask"])
        mid  = (bid + ask) / 2.0
        with self._lock:
            self._bid = bid
            self._ask = ask
            self._mid = mid

    def _poll_trades(self) -> None:
        """
        Fetch recent trades and append new ones to the CVD window.
        Coinbase returns trades newest-first; we reverse to process chronologically.
        side='buy' = taker buy (bullish), side='sell' = taker sell (bearish).
        """
        trades = self._fetch_json(TRADES_URL)
        if not isinstance(trades, list) or not trades:
            return

        # Trades are newest-first; process oldest-first so CVD builds correctly.
        trades_asc = list(reversed(trades))

        with self._lock:
            for t in trades_asc:
                try:
                    trade_id = int(t["trade_id"])
                    if self._last_trade_id is not None and trade_id <= self._last_trade_id:
                        continue   # already counted
                    size = float(t["size"])
                    side = t["side"]   # "buy" or "sell"
                    signed_qty = size if side == "buy" else -size
                    if len(self._trade_qtys) == CVD_WINDOW:
                        self._cvd_sum -= self._trade_qtys[0]
                    self._trade_qtys.append(signed_qty)
                    self._cvd_sum += signed_qty
                    self._last_trade_id = trade_id
                except (KeyError, ValueError, TypeError):
                    continue
