"""
ladder_backtest.py — Price-ladder (bounce) analysis for Kalshi YES contracts.

For each 10-cent entry level, measures the conditional probability that
the YES price subsequently reaches the next tier within the same market.

Interpretation:
  "Buy YES at 10c" → first tick where ask ≤ 0.10
  "Pop to 20c"     → any subsequent tick where ask ≥ 0.20
  If market settled YES (outcome=1), the price definitely passed through
  all tiers below 1.0, so those count as hits too.

Usage (via main.py):
  python main.py ladder --asset ETH
  python main.py ladder --asset BTC
  python main.py ladder --asset ETH --step 5    # 5-cent increments
"""

from __future__ import annotations

from typing import List

import pandas as pd


def run_ladder_backtest(
    df: pd.DataFrame,
    step_c: int = 10,
) -> pd.DataFrame:
    """
    For each entry tier (5c to 95c at step_c intervals):
      1. Find first tick per market where ask <= entry_c / 100
      2. Check if ask reaches (entry_c + step_c) / 100 in subsequent ticks
      3. If market settled YES, count as hit (price went to 1.0)

    Returns DataFrame with one row per tier.
    """
    step = step_c / 100.0
    results = []

    for entry_c in range(5, 100, step_c):
        entry   = entry_c / 100.0
        target  = round(entry + step, 4)
        if target > 1.001:
            break

        n_entered    = 0
        n_hit        = 0
        n_via_settle = 0
        pnl_list: List[float] = []

        for _, mkt in df.groupby("ticker"):
            mkt = mkt.sort_values("tick_time").reset_index(drop=True)

            # First tick where you can buy at or below entry level
            eligible = mkt[mkt["ask"] <= entry]
            if eligible.empty:
                continue

            entry_idx = int(eligible.index[0])
            entry_ask = float(mkt.loc[entry_idx, "ask"])
            outcome   = int(mkt["outcome"].iloc[-1])
            n_entered += 1

            after = mkt.iloc[entry_idx + 1:].reset_index(drop=True)

            hit_in_ticks     = (not after.empty) and bool((after["ask"] >= target).any())
            hit_via_settle   = (outcome == 1)
            hit              = hit_in_ticks or hit_via_settle

            if hit:
                n_hit += 1
                if hit_in_ticks:
                    exit_price = target
                else:
                    n_via_settle += 1
                    exit_price   = 1.0
            else:
                exit_price = 0.0

            pnl_list.append(round(exit_price - entry_ask, 4))

        n = n_entered
        results.append({
            "entry_c":    entry_c,
            "target_c":   int(round(target * 100)),
            "markets":    n,
            "hit":        n_hit,
            "hit_rate":   round(n_hit / n * 100, 1) if n else 0.0,
            "via_settle": n_via_settle,
            "avg_pnl":    round(sum(pnl_list) / len(pnl_list), 4) if pnl_list else 0.0,
        })

    return pd.DataFrame(results)


def print_ladder_results(results: pd.DataFrame, asset: str = "", step_c: int = 10) -> None:
    if results.empty:
        print("  No data.")
        return

    print(f"\n{'='*68}")
    print(f"  LADDER BACKTEST — {asset}   (entry → entry+{step_c}c)")
    print(f"  Buy YES at entry_c, target is +{step_c}c within same market")
    print(f"  via_settle = hit only confirmed through YES resolution")
    print(f"{'='*68}")
    print(
        f"  {'entry':>6}  {'target':>6}  {'mkts':>5}  "
        f"{'hit':>5}  {'hit%':>6}  {'via_settle':>10}  {'avg_pnl':>8}"
    )
    print(f"  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*8}")

    for _, row in results.iterrows():
        print(
            f"  {int(row['entry_c']):>5}c  {int(row['target_c']):>5}c  "
            f"{int(row['markets']):>5}  {int(row['hit']):>5}  "
            f"{row['hit_rate']:>5.1f}%  {int(row['via_settle']):>10}  "
            f"{row['avg_pnl']:>+8.4f}"
        )
    print()
