"""
build_dataset.py — Merge Kalshi ticks + asset price, compute features + outcome.

For each asset, joins the Kalshi tick parquet with the 1m price parquet using
merge_asof (backward), then computes:
  - binance_price  (the merged asset price at tick time)
  - prob           (GBM-implied probability of staying above strike)
  - edge           (prob - ask)
  - delta          (change in Kalshi price since last tick)
  - outcome        (1 if YES resolved, 0 otherwise)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Probability model ─────────────────────────────────────────────────────

def compute_prob(
    price: float,
    strike: float,
    t_left_sec: float,
    annual_vol: float = 0.60,
) -> float:
    """
    GBM-implied probability that asset stays above strike at resolution.

    Uses a simplified Black-Scholes d2 formula.  The distance entry rule
    does NOT use this — it's only for the edge column in the dataset.
    """
    if t_left_sec <= 0 or price <= 0 or strike <= 0:
        return 1.0 if price > strike else 0.0

    sigma_per_sec = annual_vol / np.sqrt(365 * 24 * 3600)
    sigma_t       = sigma_per_sec * np.sqrt(t_left_sec)

    if sigma_t < 1e-10:
        return 1.0 if price > strike else 0.0

    log_ratio = np.log(price / strike)
    d2        = log_ratio / sigma_t - 0.5 * sigma_t

    from scipy.stats import norm
    return float(norm.cdf(d2))


# ── Dataset builder ───────────────────────────────────────────────────────

def build_asset(asset_name: str, asset_cfg: dict, out_dir: str = "data") -> pd.DataFrame:
    """
    Merge Kalshi ticks with 1m price data; compute features and outcome.
    Saves to data/dataset_{asset_name.lower()}.parquet.
    """
    series      = asset_cfg["kalshi_series"]
    annual_vol  = asset_cfg["annual_vol"]
    min_strike  = asset_cfg["min_strike"]

    ticks_path  = Path(out_dir) / f"kalshi_{series.lower()}.parquet"
    prices_path = Path(out_dir) / asset_cfg["price_file"]
    out_path    = Path(out_dir) / f"dataset_{asset_name.lower()}.parquet"

    if not ticks_path.exists():
        raise FileNotFoundError(f"Kalshi ticks not found: {ticks_path}  (run fetch first)")
    if not prices_path.exists():
        raise FileNotFoundError(f"Price data not found: {prices_path}  (run fetch first)")

    ticks  = pd.read_parquet(ticks_path)
    prices = pd.read_parquet(prices_path)

    # Normalise timestamps to UTC datetime
    ticks["tick_time"] = pd.to_datetime(ticks["tick_time"], utc=True)
    ticks["close_ts"]  = pd.to_datetime(ticks["close_ts"],  utc=True)
    prices["open_time"] = pd.to_datetime(prices["open_time"], utc=True)

    # Drop zero strikes
    ticks = ticks[ticks["strike"] > min_strike].copy()
    if ticks.empty:
        logger.warning("%s: all ticks have strike=0 — check _extract_strike fix", asset_name)
        return pd.DataFrame()

    # Sort for merge_asof
    ticks  = ticks.sort_values("tick_time").reset_index(drop=True)
    prices = prices.sort_values("open_time").reset_index(drop=True)

    # Merge: for each Kalshi tick, find the most recent 1m price candle (backward)
    merged = pd.merge_asof(
        ticks,
        prices[["open_time", "close"]].rename(columns={"open_time": "tick_time", "close": "binance_price"}),
        on="tick_time",
        direction="backward",
    )

    # Drop rows where we have no price
    merged = merged.dropna(subset=["binance_price"]).copy()

    # ── Outcome: YES resolved if price > strike at close_ts ──────────────
    # Look up the asset price closest to each market's close_ts
    close_times = merged[["ticker", "close_ts"]].drop_duplicates()

    price_at_close = pd.merge_asof(
        close_times.sort_values("close_ts"),
        prices[["open_time", "close"]].rename(columns={"open_time": "close_ts", "close": "close_price"}),
        on="close_ts",
        direction="nearest",
    )

    merged = merged.merge(price_at_close, on=["ticker", "close_ts"], how="left")
    merged["outcome"] = (merged["close_price"] > merged["strike"]).astype(int)

    # Drop ticks after resolution
    merged = merged[merged["tick_time"] <= merged["close_ts"]].copy()

    # ── GBM probability ───────────────────────────────────────────────────
    def _row_prob(row):
        return compute_prob(
            price      = row["binance_price"],
            strike     = row["strike"],
            t_left_sec = max(0.0, row["t_left"]),
            annual_vol = annual_vol,
        )

    merged["prob"] = merged.apply(_row_prob, axis=1)
    merged["edge"] = merged["prob"] - merged["ask"]

    # ── Delta (price change since previous tick, per market) ─────────────
    merged = merged.sort_values(["ticker", "tick_time"]).reset_index(drop=True)
    merged["delta"] = merged.groupby("ticker")["price"].diff().fillna(0.0)

    # ── Streak (consecutive positive/negative deltas per market) ─────────
    def _streak(series):
        streaks = []
        s = 0
        for v in series:
            if v > 0:
                s = s + 1 if s > 0 else 1
            elif v < 0:
                s = s - 1 if s < 0 else -1
            else:
                s = 0
            streaks.append(s)
        return streaks

    streaks = []
    for _, grp in merged.groupby("ticker"):
        streaks.extend(_streak(grp["delta"].tolist()))
    merged["streak"] = streaks

    merged = merged.sort_values("tick_time").reset_index(drop=True)
    merged.to_parquet(out_path, index=False)

    n_markets = merged["ticker"].nunique()
    logger.info(
        "%s: %d ticks across %d markets → %s",
        asset_name, len(merged), n_markets, out_path,
    )
    print(f"  {asset_name}: {len(merged):,} ticks, {n_markets} markets → {out_path}")
    return merged
