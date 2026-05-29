"""
market_scanner.py — Scan Kalshi for active BTC 15m markets closing soon.

Returns markets with:
  - t_left between T_ENTER_MIN and T_ENTER_MAX + buffer
  - best_ask and best_bid available
  - open_price (strike) = BTC price at market open, looked up from Binance 1s klines
    or approximated from the market's open timestamp
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from live.config import KALSHI_SERIES, T_ENTER_MAX


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _extract_close_ts(market) -> int:
    """Extract close timestamp from Kalshi market object."""
    for attr in ("close_ts", "close_time", "expiration_time", "close_time_ts"):
        val = getattr(market, attr, None) or (market.get(attr) if isinstance(market, dict) else None)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return int(val)
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
            return int(dt.timestamp())
        except Exception:
            pass
    return 0


def _extract_ask(market) -> float | None:
    for attr in ("yes_ask_dollars", "yes_ask", "ask_dollars", "ask"):
        val = getattr(market, attr, None) or (market.get(attr) if isinstance(market, dict) else None)
        if val is not None:
            try:
                v = float(val)
                return v if v <= 1.0 else v / 100.0
            except Exception:
                pass
    return None


def _extract_bid(market) -> float | None:
    for attr in ("yes_bid_dollars", "yes_bid", "bid_dollars", "bid"):
        val = getattr(market, attr, None) or (market.get(attr) if isinstance(market, dict) else None)
        if val is not None:
            try:
                v = float(val)
                return v if v <= 1.0 else v / 100.0
            except Exception:
                pass
    return None


def _extract_ticker(market) -> str:
    ticker = getattr(market, "ticker", None) or (market.get("ticker") if isinstance(market, dict) else None)
    return str(ticker) if ticker else ""


def scan_active_markets(client, btc_price: float) -> list[dict]:
    """
    Return list of active markets closing within T_ENTER_MAX + 30s.
    Each dict has: ticker, close_ts, t_left, ask, bid, open_price, mid
    """
    now = _now_ts()
    lookahead = T_ENTER_MAX + 30   # scan slightly ahead of entry window

    results = []

    try:
        from pykalshi.enums import MarketStatus
        resp = client.get_markets(
            series_ticker=KALSHI_SERIES,
            status=MarketStatus.OPEN,
            limit=100,
        )
        markets = getattr(resp, "markets", None) or (resp if isinstance(resp, list) else [])
    except Exception as exc:
        print(f"  [scanner] get_markets failed: {exc}")
        return []

    for m in markets:
        close_ts = _extract_close_ts(m)
        if close_ts == 0:
            continue

        t_left = close_ts - now
        if t_left < 0 or t_left > lookahead:
            continue

        ask = _extract_ask(m)
        bid = _extract_bid(m)
        if ask is None:
            continue

        ticker = _extract_ticker(m)
        mid = (ask + bid) / 2 if bid is not None else ask

        # Open price = BTC price at market open (close_ts - 900)
        # We approximate using the current BTC price minus drift as open_price.
        # In production, look this up from a 1-second kline cache.
        open_ts = close_ts - 900
        open_price = _lookup_open_price(open_ts, btc_price)

        results.append({
            "ticker":     ticker,
            "close_ts":   close_ts,
            "t_left":     t_left,
            "ask":        ask,
            "bid":        bid,
            "mid":        mid,
            "open_price": open_price,
        })

    return results


# ── Open price cache ──────────────────────────────────────────────────────────
# Maps open_ts → BTC price at that second.
# Populated by the main trader loop from the price feed history.
_open_price_cache: dict[int, float] = {}


def cache_open_price(open_ts: int, price: float) -> None:
    _open_price_cache[open_ts] = price
    # Evict entries older than 2 hours
    cutoff = _now_ts() - 7200
    for k in list(_open_price_cache):
        if k < cutoff:
            del _open_price_cache[k]


def _lookup_open_price(open_ts: int, current_btc: float) -> float:
    if open_ts in _open_price_cache:
        return _open_price_cache[open_ts]
    # Fallback: use current price (will give z≈0, won't trigger entry — safe default)
    return current_btc
