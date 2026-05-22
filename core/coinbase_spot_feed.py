"""
coinbase_spot_feed.py — Coinbase BTC-USD spot WebSocket feed.

Provides:
  - Rolling CVD (Cumulative Volume Delta) from market_trades channel
  - Current mid price from ticker channel
  - Price change direction for cross_asset_boost in EVSignalEngine

No API key required — uses public Coinbase Advanced Trade WebSocket.

Drop-in replacement for BinanceFeed: exposes identical mid_price, cvd,
and is_connected properties.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Optional

import websocket

logger = logging.getLogger(__name__)

WS_URL          = "wss://advanced-trade-api.coinbase.com/ws/public"
PRODUCT_ID      = "BTC-USD"
CVD_WINDOW      = 200
MAX_ERRORS      = 20
RECONNECT_DELAY = 5.0

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
    Background thread streaming Coinbase BTC-USD spot data.

    ticker channel: maintains current best bid/ask and mid price.
    market_trades channel: computes rolling CVD.
      - side=BUY  → taker buy  (bullish) → +size
      - side=SELL → taker sell (bearish) → -size
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

        self._ws: Optional[websocket.WebSocketApp] = None

    # ── Public properties ─────────────────────────────────────────────

    @property
    def mid_price(self) -> Optional[float]:
        """Latest BTC-USD mid price from Coinbase ticker."""
        with self._lock:
            return self._mid

    @property
    def cvd(self) -> float:
        """
        Cumulative Volume Delta over last CVD_WINDOW trades.
        Positive = net taker buying (bullish).
        Normalized to [-1, 1] by dividing by sum of absolute values.
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
        if self._ws:
            self._ws.close()

    def run(self) -> None:
        logger.info("CoinbaseSpotFeed started")
        while not self._stop_event.is_set():
            if self._error_count >= MAX_ERRORS:
                logger.error("CoinbaseSpotFeed stopping — too many errors (%d)", MAX_ERRORS)
                break
            try:
                self._connect_and_run()
            except Exception as exc:
                self._error_count += 1
                logger.warning(
                    "CoinbaseSpotFeed error #%d: %s — reconnecting in %.0fs",
                    self._error_count, exc, RECONNECT_DELAY,
                )
            if not self._stop_event.is_set():
                self._stop_event.wait(RECONNECT_DELAY)
        logger.info("CoinbaseSpotFeed stopped")

    def _connect_and_run(self) -> None:
        self._ws = websocket.WebSocketApp(
            WS_URL,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._ws.run_forever(ping_interval=20, ping_timeout=10)

    # ── WebSocket callbacks ───────────────────────────────────────────

    def _on_open(self, ws) -> None:
        self._error_count = 0
        logger.info("CoinbaseSpotFeed WebSocket connected")
        for channel in ("ticker", "market_trades"):
            ws.send(json.dumps({
                "type":        "subscribe",
                "product_ids": [PRODUCT_ID],
                "channel":     channel,
            }))

    def _on_error(self, ws, error) -> None:
        logger.warning("CoinbaseSpotFeed WS error: %s", error)

    def _on_close(self, ws, code, msg) -> None:
        logger.info("CoinbaseSpotFeed WS closed (code=%s)", code)

    def _on_message(self, ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return

        channel = msg.get("channel", "")
        events  = msg.get("events", [])

        if channel == "ticker":
            for event in events:
                for ticker in event.get("tickers", []):
                    self._handle_ticker(ticker)
        elif channel == "market_trades":
            for event in events:
                for trade in event.get("trades", []):
                    self._handle_trade(trade)

    # ── Message handlers ──────────────────────────────────────────────

    def _handle_ticker(self, data: dict) -> None:
        try:
            bid = float(data["best_bid"])
            ask = float(data["best_ask"])
            mid = (bid + ask) / 2.0
            with self._lock:
                self._bid = bid
                self._ask = ask
                self._mid = mid
        except (KeyError, ValueError, TypeError):
            pass

    def _handle_trade(self, data: dict) -> None:
        try:
            size = float(data["size"])
            side = data["side"]   # "BUY" or "SELL"
            signed_qty = size if side == "BUY" else -size
            with self._lock:
                if len(self._trade_qtys) == CVD_WINDOW:
                    evicted = self._trade_qtys[0]
                    self._cvd_sum -= evicted
                self._trade_qtys.append(signed_qty)
                self._cvd_sum += signed_qty
        except (KeyError, ValueError, TypeError):
            pass
