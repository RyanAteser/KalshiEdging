"""
settle_backtest.py — Buy-to-settle strategy using z-score as certainty gate.

Strategy:
  When z-score exceeds a threshold → buy YES (or NO) at current ask → hold to settlement.
  Profit = settlement payout minus entry price minus fees.

The core question:
  At what z-score is the win rate high enough to survive the 14c round-trip fee?

  Break-even win rate = (entry_price + fee) / 100
  e.g. entry at 95c, fee 14c: need (95 + 14) / 100 = impossible (>100%)
  e.g. entry at 80c, fee 14c: need (80 + 14) / 100 = 94% win rate to break even

For NO side (buying when z is very negative):
  NO_price = 100 - ask
  Break-even = (NO_price + fee) / 100

Output:
  - Per z-threshold: win rate, avg entry price, avg pnl after fees
  - Break-even win rate curve vs actual win rate
  - The "sweet spot": z where actual win rate > break-even win rate

Usage:
  %PY% main.py settle --asset BTC
  %PY% main.py settle --asset ETH
  %PY% main.py settle --asset SOL
  %PY% main.py settle --asset BTC --side no
  %PY% main.py settle --asset BTC --z-min 3.0 --z-max 10.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.z_score import compute_z_score

FEE_CENTS      = 7.0    # per side
BTC_HISTORY    = 30     # 1m candles

# Only enter within these time windows (seconds remaining)
T_MIN = 60      # don't enter final 60s (liquidity gone)
T_MAX = 900     # full 15 minutes


def _load_btc_1m(asset_cfg: dict, data_dir: str = "data") -> pd.DataFrame | None:
    path = Path(data_dir) / asset_cfg.get("price_file", "")
    if not path.exists():
        return None
    p = pd.read_parquet(path)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    return p[["open_time", "close"]].sort_values("open_time").reset_index(drop=True)


def run_settle_sweep(
    df: pd.DataFrame,
    btc_1m: pd.DataFrame | None = None,
    side: str = "yes",            # "yes" | "no"
    z_values: list[float] | None = None,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    one_entry_per_market: bool = True,  # take only first qualifying tick per market
) -> pd.DataFrame:
    """
    For each z threshold: scan all markets, enter on the first tick that qualifies,
    hold to settlement, record outcome.
    Returns DataFrame with one row per z threshold.
    """
    if z_values is None:
        z_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    if btc_1m is not None and "tick_time" in df.columns:
        df = df.copy()
        df["tick_time"] = pd.to_datetime(df["tick_time"], utc=True)

    # Pre-compute all observations once (z-score per tick across all markets)
    all_obs = _compute_all_obs(df, btc_1m, side, t_min, t_max)
    if all_obs.empty:
        return pd.DataFrame()

    rows = []
    for z_thresh in z_values:
        if side == "yes":
            qualifies = all_obs["z"] >= z_thresh
        else:
            qualifies = all_obs["z"] <= -z_thresh   # z negative for NO side

        candidates = all_obs[qualifies].copy()

        if one_entry_per_market:
            # Take only the first qualifying tick per market
            candidates = candidates.sort_values("tick_idx")
            candidates = candidates.groupby("ticker").first().reset_index()

        n = len(candidates)
        if n == 0:
            rows.append({
                "z_thresh": z_thresh, "trades": 0, "markets": 0,
                "win_rate%": 0.0, "be_rate%": 0.0,
                "avg_entry_c": 0.0, "avg_pnl_c": 0.0,
                "avg_pnl_nofee_c": 0.0, "total_pnl_c": 0.0,
                "avg_z": 0.0, "avg_t_left": 0.0,
            })
            continue

        wins = candidates["won"].sum()
        pnls = candidates["pnl_c"].values
        pnls_nofee = candidates["pnl_nofee_c"].values
        avg_entry = candidates["entry_c"].mean()

        # Break-even win rate for this avg entry price
        be_rate = (avg_entry + FEE_CENTS * 2) / 100.0

        rows.append({
            "z_thresh":        z_thresh,
            "trades":          n,
            "markets":         candidates["ticker"].nunique() if "ticker" in candidates.columns else n,
            "win_rate%":       round(wins / n * 100, 2),
            "be_rate%":        round(be_rate * 100, 2),
            "avg_entry_c":     round(float(avg_entry), 1),
            "avg_pnl_c":       round(float(np.mean(pnls)), 2),
            "avg_pnl_nofee_c": round(float(np.mean(pnls_nofee)), 2),
            "total_pnl_c":     round(float(np.sum(pnls)), 1),
            "avg_z":           round(float(candidates["z"].mean()), 2),
            "avg_t_left":      round(float(candidates["t_left"].mean()), 0),
        })

    return pd.DataFrame(rows)


def _compute_all_obs(
    df: pd.DataFrame,
    btc_1m: pd.DataFrame | None,
    side: str,
    t_min: float,
    t_max: float,
) -> pd.DataFrame:
    """Compute z-score for every valid tick across all markets."""
    rows = []

    for ticker, mkt in df.groupby("ticker"):
        mkt     = mkt.sort_values("tick_time").reset_index(drop=True)
        outcome = int(mkt["outcome"].iloc[-1])
        n       = len(mkt)

        asks    = mkt["ask"].values * 100.0
        t_lefts = mkt["t_left"].values
        btc_arr = mkt["binance_price"].values
        strikes = mkt["strike"].values
        times   = mkt["tick_time"].values

        for i in range(1, n):
            t_left = float(t_lefts[i])
            if t_left < t_min or t_left > t_max:
                continue

            btc    = float(btc_arr[i])
            strike = float(strikes[i])
            ask    = float(asks[i])

            if btc_1m is not None:
                tick_ts = pd.Timestamp(times[i], tz="UTC")
                mask    = btc_1m["open_time"] < tick_ts
                recent  = btc_1m.loc[mask, "close"].values[-BTC_HISTORY:]
                history = list(recent) if len(recent) >= 5 else list(btc_arr[max(0, i-30):i])
            else:
                history = list(btc_arr[max(0, i-BTC_HISTORY):i])

            z, _ = compute_z_score(btc, strike, t_left, history)

            if side == "yes":
                entry_c = ask
                won     = outcome == 1
                pnl_nofee = (100 - entry_c) if won else -entry_c
                pnl_c     = pnl_nofee - FEE_CENTS * 2
            else:
                entry_c = 100 - ask          # cost of NO contract
                won     = outcome == 0
                pnl_nofee = (100 - entry_c) if won else -entry_c
                pnl_c     = pnl_nofee - FEE_CENTS * 2

            rows.append({
                "ticker":      ticker,
                "tick_idx":    i,
                "z":           z,
                "t_left":      t_left,
                "entry_c":     entry_c,
                "won":         int(won),
                "pnl_c":       pnl_c,
                "pnl_nofee_c": pnl_nofee,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Print ───────────────────────────────────────────────────────────────────

def print_settle_sweep(
    results: pd.DataFrame,
    asset: str,
    side: str,
) -> None:
    if results.empty:
        print("  No data.")
        return

    side_str = "YES (buy when z > threshold)" if side == "yes" \
               else "NO  (buy when z < -threshold)"

    print(f"\n{'='*96}")
    print(f"  BUY-TO-SETTLE SWEEP — {asset}   side={side_str}")
    print(f"  Strategy: enter at first qualifying tick per market, hold to settlement")
    print(f"  be_rate% = win rate needed just to break even after 14c fees at avg entry price")
    print(f"  ◀ = win_rate > be_rate  (profitable after fees)")
    print(f"  ★ = win_rate ≥ 99%")
    print(f"{'='*96}")
    print(
        f"  {'z≥':>5}  {'trades':>7}  {'win%':>7}  {'be%':>7}  {'gap':>6}  "
        f"{'avg_entry':>10}  {'pnl(w/fee)':>11}  {'pnl(no fee)':>12}  "
        f"{'tot_pnl':>9}  {'avg_z':>7}  {'avg_t':>7}"
    )
    print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  "
          f"{'─'*10}  {'─'*11}  {'─'*12}  {'─'*9}  {'─'*7}  {'─'*7}")

    for _, row in results.iterrows():
        if row["trades"] == 0:
            z_str = f"{row['z_thresh']:.1f}"
            print(f"  {z_str:>5}  {'0':>7}  {'—':>7}  {'—':>7}  {'—':>6}  "
                  f"{'—':>10}  {'—':>11}  {'—':>12}  {'—':>9}  {'—':>7}  {'—':>7}")
            continue

        win   = row["win_rate%"]
        be    = row["be_rate%"]
        gap   = win - be
        flag  = " ◀" if gap > 0 else ""
        star  = " ★" if win >= 99.0 else ""
        marks = (flag + star).strip()

        print(
            f"  {row['z_thresh']:>4.1f}  {int(row['trades']):>7,}  "
            f"{win:>6.2f}%  {be:>6.2f}%  {gap:>+5.2f}%  "
            f"{row['avg_entry_c']:>9.1f}c  "
            f"{row['avg_pnl_c']:>+10.2f}c  {row['avg_pnl_nofee_c']:>+11.2f}c  "
            f"{row['total_pnl_c']:>+8.1f}c  "
            f"{row['avg_z']:>7.2f}  {int(row['avg_t_left']):>5}s"
            + (f"  {marks}" if marks else "")
        )

    # Summary: find the sweet spot
    profitable = results[(results["win_rate%"] > results["be_rate%"]) & (results["trades"] > 0)]
    if not profitable.empty:
        best = profitable.iloc[0]
        print(f"\n  Sweet spot starts at z≥{best['z_thresh']:.1f}:  "
              f"{best['win_rate%']:.2f}% win rate  vs  {best['be_rate%']:.2f}% needed  "
              f"→ {best['avg_pnl_c']:+.2f}c/trade after fees")
    else:
        print(f"\n  No z threshold produces positive EV after fees on this dataset.")
        # Show closest
        valid = results[results["trades"] > 0].copy()
        if not valid.empty:
            valid["gap"] = valid["win_rate%"] - valid["be_rate%"]
            closest = valid.loc[valid["gap"].idxmax()]
            print(f"  Closest: z≥{closest['z_thresh']:.1f}  gap={closest['gap']:+.2f}%  "
                  f"win={closest['win_rate%']:.2f}%  need={closest['be_rate%']:.2f}%")
    print()
