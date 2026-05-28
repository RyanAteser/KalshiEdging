"""
ladder_backtest.py — Price-ladder (bounce) analysis for Kalshi YES contracts.

For each 10-cent entry tier, finds markets where the YES ask was trading
near that level, then measures the probability it subsequently reached the
next tier (and expected PnL if you hold to settlement if it doesn't).

Entry logic: ask within ±tolerance_c of entry level (default ±2c).
  → "Buy at 10c" = first tick where ask is between 8c and 12c.

Three outcomes per trade:
  hit_ticks    — ask reached target in a later observed tick (you sold at target)
  settled_yes  — market resolved YES before ask reached target in ticks
                 (you hold to 1.0 — better than target, just not the quick flip)
  settled_no   — market resolved NO (you lose entry cost)

Expected PnL assumes you buy at exactly entry_c and:
  - Sell at target_c if hit in ticks          → pnl = +step_c cents
  - Hold to settlement if not hit in ticks:
      settled YES                              → pnl = (100 - entry_c) cents
      settled NO                              → pnl = -entry_c cents

Usage (via main.py):
  python main.py ladder --asset ETH
  python main.py ladder --asset BTC
  python main.py ladder --asset ETH --step 5    # 5-cent increments
"""

from __future__ import annotations

import pandas as pd


def run_ladder_backtest(
    df: pd.DataFrame,
    step_c: int = 10,
    tolerance_c: int = 2,
) -> pd.DataFrame:
    """
    For each entry tier (5c to 95c at step_c intervals):
      1. Find first tick per market where ask is within ±tolerance_c of entry
      2. Check if ask reaches target in subsequent ticks (hit_ticks)
      3. Otherwise record settlement outcome (settled_yes or settled_no)
    """
    step      = step_c / 100.0
    tolerance = tolerance_c / 100.0
    results   = []

    for entry_c in range(5, 100, step_c):
        entry  = entry_c / 100.0
        target = round(entry + step, 4)
        if target > 1.001:
            break

        n_entered    = 0
        n_hit_ticks  = 0
        n_settled_yes = 0
        n_settled_no  = 0

        for _, mkt in df.groupby("ticker"):
            mkt = mkt.sort_values("tick_time").reset_index(drop=True)

            # First tick where ask is near entry level (within tolerance)
            eligible = mkt[abs(mkt["ask"] - entry) <= tolerance]
            if eligible.empty:
                continue

            entry_idx = int(eligible.index[0])
            outcome   = int(mkt["outcome"].iloc[-1])
            n_entered += 1

            after = mkt.iloc[entry_idx + 1:].reset_index(drop=True)
            hit_ticks = (not after.empty) and bool((after["ask"] >= target).any())

            if hit_ticks:
                n_hit_ticks += 1
            elif outcome == 1:
                n_settled_yes += 1
            else:
                n_settled_no += 1

        n = n_entered
        if n == 0:
            continue

        # Expected PnL (in cents) assuming buy at entry_c, sell at target_c or hold
        exp_pnl_c = (
            n_hit_ticks   / n * step_c
            + n_settled_yes / n * (100 - entry_c)
            + n_settled_no  / n * (-entry_c)
        )

        results.append({
            "entry_c":      entry_c,
            "target_c":     int(round(target * 100)),
            "markets":      n,
            "hit_ticks":    n_hit_ticks,
            "hit_ticks%":   round(n_hit_ticks / n * 100, 1),
            "settled_yes":  n_settled_yes,
            "settled_no":   n_settled_no,
            "exp_pnl_c":    round(exp_pnl_c, 1),
        })

    return pd.DataFrame(results)


def print_ladder_results(results: pd.DataFrame, asset: str = "", step_c: int = 10) -> None:
    if results.empty:
        print("  No data.")
        return

    print(f"\n{'='*75}")
    print(f"  LADDER BACKTEST — {asset}   (buy at entry_c, sell at target_c)")
    print(f"  hit_ticks = reached target in observed ticks (quick flip)")
    print(f"  settled_yes/no = held to resolution (target not hit in ticks)")
    print(f"  exp_pnl_c = expected cents per trade (entry assumed at tier price)")
    print(f"{'='*75}")
    print(
        f"  {'entry':>6}  {'target':>6}  {'mkts':>5}  "
        f"{'hit_tks':>7}  {'hit%':>5}  "
        f"{'s_yes':>5}  {'s_no':>5}  {'exp_pnl_c':>9}"
    )
    print(f"  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*9}")

    for _, row in results.iterrows():
        flag = ""
        if row["hit_ticks%"] >= 50:
            flag = " ◀"
        print(
            f"  {int(row['entry_c']):>5}c  {int(row['target_c']):>5}c  "
            f"{int(row['markets']):>5}  "
            f"{int(row['hit_ticks']):>7}  {row['hit_ticks%']:>4.1f}%  "
            f"{int(row['settled_yes']):>5}  {int(row['settled_no']):>5}  "
            f"{row['exp_pnl_c']:>+8.1f}c"
            f"{flag}"
        )
    print()
