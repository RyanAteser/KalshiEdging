"""
load_bulk.py — Load Kalshi bulk historical data into backtest dataset format.

Reads the folder structure from Kalshi's data download:
  <data_dir>/prices_btc_15m/dt=YYYY-MM-DD/  — tick-level up/down prices
  <data_dir>/binance_btc_klines_1s/dt=YYYY-MM-DD/  — 1s BTC prices

Produces:
  data/dataset_btc.parquet  — same format as build_dataset.py output

Usage:
  python main.py load-bulk --asset BTC --data-dir E:/april_lastweek_15m
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_bulk_dataset(
    asset: str,
    asset_cfg: dict,
    data_dir: str,
    out_dir: str = "data",
) -> pd.DataFrame:
    asset_lower = asset.lower()
    data_path   = Path(data_dir)
    out_path    = Path(out_dir) / f"dataset_{asset_lower}.parquet"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    prices_dir = data_path / f"prices_{asset_lower}_15m"
    klines_dir = data_path / f"binance_{asset_lower}_klines_1s"

    if not prices_dir.exists():
        raise FileNotFoundError(f"prices dir not found: {prices_dir}")
    if not klines_dir.exists():
        raise FileNotFoundError(f"klines dir not found: {klines_dir}")

    # ── Load prices (tick-level up/down ask/bid per market) ───────────────
    print(f"  Loading {asset} prices from {prices_dir} ...")
    price_parts = []
    for dt_dir in sorted(prices_dir.glob("dt=*")):
        try:
            df = pd.read_parquet(dt_dir)
            price_parts.append(df)
        except Exception as e:
            print(f"    Warning: could not read {dt_dir}: {e}")
    if not price_parts:
        raise RuntimeError(f"No price data found in {prices_dir}")
    prices_df = pd.concat(price_parts, ignore_index=True)
    prices_df["time"] = pd.to_datetime(prices_df["time"], utc=True).dt.as_unit("us")
    prices_df = prices_df.sort_values("time").reset_index(drop=True)
    print(f"    {len(prices_df):,} price ticks across {prices_df['slug'].nunique()} markets")

    # ── Load Binance 1s klines ────────────────────────────────────────────
    print(f"  Loading {asset} Binance klines from {klines_dir} ...")
    kline_parts = []
    for dt_dir in sorted(klines_dir.glob("dt=*")):
        try:
            df = pd.read_parquet(dt_dir)
            kline_parts.append(df)
        except Exception as e:
            print(f"    Warning: could not read {dt_dir}: {e}")
    if not kline_parts:
        raise RuntimeError(f"No kline data found in {klines_dir}")
    klines = pd.concat(kline_parts, ignore_index=True)
    klines["time"] = pd.to_datetime(klines["time"], utc=True).dt.as_unit("us")
    klines = klines.sort_values("time").reset_index(drop=True)
    print(f"    {len(klines):,} kline candles")

    # ── Extract close_ts from slug ────────────────────────────────────────
    # slug format: btc-updown-15m-1776988800
    def _close_ts(slug):
        m = re.search(r"-(\d{9,})$", slug)
        return int(m.group(1)) if m else 0

    prices_df["open_ts_unix"] = prices_df["slug"].map(_close_ts)
    prices_df["open_ts"]  = pd.to_datetime(
        prices_df["open_ts_unix"], unit="s", utc=True
    ).dt.as_unit("us")
    prices_df["close_ts"] = prices_df["open_ts"] + pd.Timedelta(seconds=900)
    prices_df["t_left"]        = (prices_df["close_ts"] - prices_df["time"]).dt.total_seconds()

    print(f"    t_left sample: min={prices_df['t_left'].min():.0f}  max={prices_df['t_left'].max():.0f}  "
          f"median={prices_df['t_left'].median():.0f}")
    # Filter to valid window only
    prices_df = prices_df[(prices_df["t_left"] >= 0) & (prices_df["t_left"] <= 960)].copy()
    print(f"    after t_left filter: {len(prices_df):,} rows")

    def _us(col): return col.dt.as_unit("us")

    # ── Strike = BTC price at market open ────────────────────────────────
    market_meta = prices_df[["slug", "open_ts", "close_ts"]].drop_duplicates("slug").copy()
    # slug = open timestamp marker; close_ts = open + 900s
    market_meta["open_ts"]  = _us(market_meta["open_ts"])
    market_meta["close_ts"] = _us(market_meta["close_ts"])
    market_meta = market_meta.sort_values("open_ts").reset_index(drop=True)

    klines_strike = klines[["time", "close"]].copy()
    klines_strike["time"] = _us(klines_strike["time"])
    klines_strike = klines_strike.rename(columns={"time": "open_ts", "close": "strike"})
    market_meta = pd.merge_asof(
        market_meta, klines_strike, on="open_ts", direction="nearest"
    )

    # ── Outcome = BTC at close > BTC at open (i.e. went Up) ──────────────
    klines_close = klines[["time", "close"]].copy()
    klines_close["time"] = _us(klines_close["time"])
    klines_close = klines_close.rename(columns={"time": "close_ts", "close": "close_price"})
    market_meta["close_ts"] = _us(market_meta["close_ts"])
    market_meta = market_meta.sort_values("close_ts").reset_index(drop=True)
    market_meta = pd.merge_asof(
        market_meta, klines_close, on="close_ts", direction="nearest"
    )
    market_meta["outcome"] = (market_meta["close_price"] > market_meta["strike"]).astype(int)

    # ── Merge BTC price at each tick time ────────────────────────────────
    klines_tick = klines[["time", "close"]].copy()
    klines_tick["time"] = _us(klines_tick["time"])
    klines_tick = klines_tick.rename(columns={"time": "tick_time", "close": "binance_price"})
    prices_df = prices_df.rename(columns={"time": "tick_time", "slug": "ticker"})
    prices_df["tick_time"] = _us(prices_df["tick_time"])
    prices_df = prices_df.sort_values("tick_time").reset_index(drop=True)
    prices_df = pd.merge_asof(
        prices_df, klines_tick, on="tick_time", direction="backward"
    )

    # ── Merge market meta ─────────────────────────────────────────────────
    print(f"    prices_df before meta merge: {len(prices_df):,} rows")
    prices_df = prices_df.merge(
        market_meta[["slug" if "slug" in market_meta.columns else "ticker",
                     "strike", "outcome"]].rename(columns={"slug": "ticker"}),
        on="ticker", how="left"
    )
    print(f"    prices_df after meta merge:  {len(prices_df):,} rows")

    # ── Build final columns ───────────────────────────────────────────────
    # YES side = Up side
    prices_df["ask"]   = prices_df["up_ask"]
    prices_df["bid"]   = prices_df["up_bid"]
    prices_df["price"] = prices_df["up_mid"]

    keep = ["ticker", "tick_time", "strike", "price", "ask", "bid",
            "t_left", "close_ts", "binance_price", "outcome"]
    print(f"    NaN binance_price: {prices_df['binance_price'].isna().sum():,}  "
          f"NaN strike: {prices_df['strike'].isna().sum():,}")
    df = prices_df[keep].dropna(subset=["binance_price", "strike"]).copy()
    df = df[df["strike"] > asset_cfg.get("min_strike", 0)].copy()

    # ── GBM probability + edge ────────────────────────────────────────────
    from data.build_dataset import compute_prob
    annual_vol = asset_cfg.get("annual_vol", 0.60)

    def _row_prob(row):
        return compute_prob(row["binance_price"], row["strike"],
                            max(0.0, row["t_left"]), annual_vol)

    print("  Computing GBM probabilities ...")
    df["prob"]  = df.apply(_row_prob, axis=1)
    df["edge"]  = df["prob"] - df["ask"]

    # ── Delta + streak ────────────────────────────────────────────────────
    df = df.sort_values(["ticker", "tick_time"]).reset_index(drop=True)
    df["delta"] = df.groupby("ticker")["price"].diff().fillna(0.0)

    def _streak(series):
        streaks, s = [], 0
        for v in series:
            if v > 0:   s = s + 1 if s > 0 else 1
            elif v < 0: s = s - 1 if s < 0 else -1
            else:       s = 0
            streaks.append(s)
        return streaks

    streaks = []
    for _, grp in df.groupby("ticker"):
        streaks.extend(_streak(grp["delta"].tolist()))
    df["streak"] = streaks

    df = df.sort_values("tick_time").reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    n_markets = df["ticker"].nunique()
    yes = (df.groupby("ticker")["outcome"].last() == 1).sum()
    no  = (df.groupby("ticker")["outcome"].last() == 0).sum()
    print(f"  {asset}: {len(df):,} ticks, {n_markets} markets "
          f"(YES: {yes}, NO: {no}) → {out_path}")
    return df
