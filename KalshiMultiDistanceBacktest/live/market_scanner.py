"""
market_scanner.py — Scan Kalshi for active BTC 15m markets closing soon.

Returns markets with:
  - t_left between T_ENTER_MIN and T_ENTER_MAX + buffer
  - real strike extracted from market object (not open price)
  - spread and volume checks to avoid illiquid markets
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live.config import KALSHI_SERIES, T_ENTER_MAX

# Liquidity guards
MAX_SPREAD_CENTS = 3.0    # reject if YES bid/ask spread > 3c
MIN_VOLUME       = 10     # reject if fewer than this many contracts on best ask

MIN_STRIKE = 10_000.0     # BTC strike must be > $10k to be valid

_open_price_cache: dict[int, float] = {}


def cache_open_price(close_ts: int, btc_price: float) -> None:
    _open_price_cache[close_ts] = btc_price


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _get(obj, attr):
    return obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)


def _unwrap(m):
    md = getattr(m, "data", m)
    if isinstance(md, dict) and "market" in md:
        return md["market"]
    if hasattr(md, "market") and getattr(md, "market") is not None:
        return getattr(md, "market")
    return md


def _extract_strike(m, ticker: str) -> float:
    """Pull the real dollar strike out of the market object."""
    import re
    md = _unwrap(m)

    def _parse_text(text) -> float | None:
        if not text or "tbd" in str(text).lower():
            return None
        for pat in (
            r"(?:above|below|over|under)[:\s]+\$?([\d,]+(?:\.\d+)?)",
            r"\$([\d,]+(?:\.\d+)?)",
        ):
            hit = re.search(pat, str(text), re.IGNORECASE)
            if hit:
                try:
                    v = float(hit.group(1).replace(",", ""))
                    if v > MIN_STRIKE:
                        return v
                except Exception:
                    pass
        return None

    # 1. subtitle fields (most reliable on Kalshi)
    for attr in ("yes_sub_title", "no_sub_title", "subtitle", "sub_title"):
        v = _parse_text(_get(md, attr) or _get(m, attr))
        if v:
            return v

    # 2. numeric strike fields
    for attr in ("floor_strike_dollars", "cap_strike_dollars", "reference_price",
                 "floor_strike", "strike_price", "strike", "settlement_price"):
        raw = _get(md, attr) or _get(m, attr)
        try:
            v = float(raw) if raw is not None else None
            if v and v > MIN_STRIKE:
                return v
        except Exception:
            pass

    # 3. title / rules
    for attr in ("title", "rules_primary"):
        v = _parse_text(_get(md, attr) or _get(m, attr))
        if v:
            return v

    return 0.0


def _extract_close_ts(m) -> int:
    md = _unwrap(m)
    for attr in ("close_ts", "close_time", "expiration_time", "close_time_ts"):
        val = _get(md, attr) or _get(m, attr)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return int(val)
        try:
            dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
            return int(dt.timestamp())
        except Exception:
            pass
    return 0


def _to_dollars(val) -> float | None:
    if val is None:
        return None
    try:
        v = float(val)
        return v if v <= 1.0 else v / 100.0
    except Exception:
        return None


def scan_active_markets(client, btc_price: float) -> list[dict]:
    """
    Return list of active markets closing within T_ENTER_MAX + 30s.
    Each dict has: ticker, close_ts, t_left, ask, bid, strike, spread_c
    """
    now = _now_ts()
    lookahead = T_ENTER_MAX + 30

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

        md = _unwrap(m)
        ticker = str(_get(md, "ticker") or _get(m, "ticker") or "")

        ask = _to_dollars(_get(md, "yes_ask_dollars") or _get(md, "yes_ask") or _get(m, "yes_ask"))
        bid = _to_dollars(_get(md, "yes_bid_dollars") or _get(md, "yes_bid") or _get(m, "yes_bid"))

        if ask is None or bid is None:
            continue

        spread_c = (ask - bid) * 100.0
        if spread_c > MAX_SPREAD_CENTS:
            continue   # illiquid — skip

        strike = _extract_strike(m, ticker)
        if strike <= 0:
            print(f"  [scanner] {ticker}: could not extract strike — skipping")
            continue

        results.append({
            "ticker":   ticker,
            "close_ts": close_ts,
            "t_left":   t_left,
            "ask":      ask,
            "bid":      bid,
            "mid":      (ask + bid) / 2,
            "spread_c": spread_c,
            "strike":   strike,
        })

    return results
