"""
price_feed.py — Real-time BTC price via Coinbase Advanced Trade WebSocket.

Maintains a rolling history of BTC prices for z-score computation.
Thread-safe: read `feed.price` and `feed.history` from any thread.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque

import websocket


class CoinbasePriceFeed:
    """
    Subscribes to Coinbase Advanced Trade ticker stream for BTC-USD.
    Keeps the last N prices for EWMA vol computation.
    """

    WS_URL = "wss://advanced-trade-ws.coinbase.com"

    def __init__(self, history_len: int = 60):
        self.history_len = history_len
        self.price: float | None = None
        self.history: deque[float] = deque(maxlen=history_len)
        self._lock = threading.Lock()
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        deadline = time.time() + 15
        while self.price is None and time.time() < deadline:
            time.sleep(0.1)
        if self.price is None:
            raise RuntimeError("BTC price feed failed to connect within 15s")
        print(f"  BTC feed connected — price: ${self.price:,.2f}")

    def stop(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()

    def get(self) -> tuple[float | None, list[float]]:
        with self._lock:
            return self.price, list(self.history)

    def _on_open(self, ws):
        sub = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channel": "ticker",
        }
        ws.send(json.dumps(sub))

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
            if msg.get("channel") == "ticker":
                for event in msg.get("events", []):
                    for ticker in event.get("tickers", []):
                        p = float(ticker["price"])
                        with self._lock:
                            self.price = p
                            self.history.append(p)
        except Exception:
            pass

    def _on_error(self, ws, err):
        print(f"  [price_feed] WS error: {err}")

    def _on_close(self, ws, *args):
        if self._running:
            print("  [price_feed] disconnected — reconnecting in 3s")
            time.sleep(3)
            self._run()

    def _run(self):
        self._ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever(ping_interval=20)


# Alias so trader.py import doesn't change
BinancePriceFeed = CoinbasePriceFeed
