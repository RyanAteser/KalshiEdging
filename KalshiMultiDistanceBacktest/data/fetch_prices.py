"""
fetch_prices.py — Multi-source 1-minute OHLCV fetcher.

Try order: Coinbase Exchange → Kraken → Binance (US fallback).
Saves results to parquet keyed by asset config.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

_COINBASE_BASE   = "https://api.exchange.coinbase.com/products/{pair}/candles"
_KRAKEN_URL      = "https://api.kraken.com/0/public/OHLC"
_BINANCE_HOSTS   = [
    "https://api.binance.com/api/v3/klines",
    "https://api.binance.us/api/v3/klines",
]

import os
_BINANCE_OVERRIDE = os.getenv("BINANCE_BASE_URL", "")


def _request_json(url: str, timeout: int = 15):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "kalshi-multibacktest/1.0",
            "Accept":     "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _probe_binance(symbol: str) -> Optional[str]:
    """Return the first working Binance base URL for this symbol, or None."""
    candidates = [_BINANCE_OVERRIDE] if _BINANCE_OVERRIDE else _BINANCE_HOSTS
    for base_url in candidates:
        test_url = f"{base_url}?symbol={symbol}&interval=1m&limit=1"
        try:
            _request_json(test_url, timeout=8)
            return base_url
        except Exception:
            continue
    return None


def _fetch_binance(start_ms: int, end_ms: int, base_url: str, symbol: str = "BTCUSDT") -> List[dict]:
    """Fetch 1m OHLCV from Binance.  Returns list of row dicts."""
    rows: List[dict] = []
    LIMIT = 1000
    cursor = start_ms

    while cursor < end_ms:
        url = (
            f"{base_url}?symbol={symbol}&interval=1m"
            f"&startTime={cursor}&endTime={end_ms}&limit={LIMIT}"
        )
        try:
            data = _request_json(url)
        except Exception as exc:
            print(f"    Binance batch failed: {exc}")
            break

        if not data:
            break

        for k in data:
            # [open_time, open, high, low, close, volume, close_time, ...]
            rows.append({
                "open_time": int(k[0]),
                "open":      float(k[1]),
                "high":      float(k[2]),
                "low":       float(k[3]),
                "close":     float(k[4]),
                "volume":    float(k[5]),
            })

        last_ts = int(data[-1][0])
        if last_ts <= cursor or len(data) < LIMIT:
            break
        cursor = last_ts + 60_000
        time.sleep(0.12)

    return rows


def _fetch_kraken(start_ms: int, end_ms: int, pair: str = "XBTUSD") -> List[dict]:
    """Fetch 1m OHLCV from Kraken.  Returns list of row dicts."""
    rows: List[dict] = []
    since = start_ms // 1000

    while True:
        url = f"{_KRAKEN_URL}?pair={pair}&interval=1&since={since}"
        try:
            resp = _request_json(url)
        except Exception as exc:
            print(f"    Kraken batch failed: {exc}")
            break

        if resp.get("error"):
            print(f"    Kraken API error: {resp['error']}")
            break

        result = resp.get("result", {})
        # Kraken puts data under the actual pair name (may differ from requested)
        pair_key = next((k for k in result if k != "last"), None)
        if not pair_key:
            break

        candles = result[pair_key]
        if not candles:
            break

        for c in candles:
            # [time, open, high, low, close, vwap, volume, count]
            ts_ms = int(c[0]) * 1000
            if ts_ms > end_ms:
                break
            rows.append({
                "open_time": ts_ms,
                "open":      float(c[1]),
                "high":      float(c[2]),
                "low":       float(c[3]),
                "close":     float(c[4]),
                "volume":    float(c[6]),
            })

        last_ts = int(result.get("last", 0))
        if not last_ts or last_ts * 1000 >= end_ms:
            break
        since = last_ts
        time.sleep(0.5)

    return rows


def _fetch_coinbase(start_ms: int, end_ms: int, coinbase_pair: str = "BTC-USD") -> List[dict]:
    """Fetch 1m OHLCV from Coinbase Exchange.  Returns list of row dicts."""
    rows: List[dict] = []
    url_base = _COINBASE_BASE.format(pair=coinbase_pair)
    BATCH_SECS = 300 * 60   # 300 one-minute candles per request

    current = start_ms // 1000  # convert to seconds

    while current < end_ms // 1000:
        batch_end  = min(current + BATCH_SECS, end_ms // 1000)
        start_iso  = datetime.fromtimestamp(current,    tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso    = datetime.fromtimestamp(batch_end,  tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = f"{url_base}?granularity=60&start={start_iso}&end={end_iso}"

        try:
            candles = _request_json(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"    Coinbase 404 — pair {coinbase_pair!r} not listed")
                return []
            print(f"    Coinbase HTTP {exc.code}: {exc}")
            break
        except Exception as exc:
            print(f"    Coinbase batch failed: {exc}")
            break

        if not isinstance(candles, list) or not candles:
            break

        for c in candles:
            # Coinbase format: [time, low, high, open, close, volume]
            rows.append({
                "open_time": int(c[0]) * 1000,   # convert to ms
                "open":      float(c[3]),
                "high":      float(c[2]),
                "low":       float(c[1]),
                "close":     float(c[4]),
                "volume":    float(c[5]),
            })

        current = batch_end + 1
        time.sleep(0.25)

    return rows


def fetch_asset_klines(
    start_ms: int,
    end_ms: int,
    coinbase_pair: str,
    binance_symbol: str,
    kraken_pair: str,
) -> List[dict]:
    """Try Coinbase → Kraken → Binance in order."""
    print(f"  Trying Coinbase ({coinbase_pair})...")
    rows = _fetch_coinbase(start_ms, end_ms, coinbase_pair)
    if rows:
        return rows

    print(f"  Coinbase failed — trying Kraken ({kraken_pair})...")
    rows = _fetch_kraken(start_ms, end_ms, kraken_pair)
    if rows:
        return rows

    print(f"  Kraken failed — trying Binance ({binance_symbol})...")
    binance_url = _probe_binance(binance_symbol)
    if binance_url:
        return _fetch_binance(start_ms, end_ms, binance_url, binance_symbol)

    print(f"  ERROR: all sources failed for {coinbase_pair}")
    return []


def save_prices(asset_cfg: dict, days: int = 30, out_dir: str = "data") -> pd.DataFrame:
    """Fetch 1m OHLCV for any asset and cache to parquet."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / asset_cfg["price_file"]
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 86_400_000

    raw = fetch_asset_klines(
        start_ms, now_ms,
        coinbase_pair  = asset_cfg["coinbase_pair"],
        binance_symbol = asset_cfg["binance_symbol"],
        kraken_pair    = asset_cfg["kraken_pair"],
    )

    if not raw:
        print(f"  WARNING: no price rows fetched for {asset_cfg['price_file']}")
        return pd.DataFrame()

    df_new = pd.DataFrame(raw)
    df_new["open_time"] = pd.to_datetime(df_new["open_time"], unit="ms", utc=True)
    df_new = df_new.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    # Merge with existing cached data if present
    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        df_old["open_time"] = pd.to_datetime(df_old["open_time"], utc=True)
        df_merged = (
            pd.concat([df_old, df_new], ignore_index=True)
            .drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True)
        )
    else:
        df_merged = df_new

    df_merged.to_parquet(out_path, index=False)
    print(f"  Saved {len(df_merged):,} rows → {out_path}")
    return df_merged
