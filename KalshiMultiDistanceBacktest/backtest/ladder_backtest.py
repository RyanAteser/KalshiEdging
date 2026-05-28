"""
ladder_backtest.py — Price-ladder (bounce) analysis for Kalshi YES contracts.

For each entry tier, finds markets where the YES ask was near that level,
then measures whether it reached the next tier (hit_ticks), stopped out
(fell stop_loss_c below entry), or held to settlement.

Entry logic: ask within ±tolerance_c of entry level (default ±2c).

Four outcomes per trade:
  hit_ticks   — ask reached target tier  → pnl = +step_c
  stopped     — ask fell stop_loss_c below entry → pnl = -stop_loss_c
  settled_yes — resolved YES before either level → pnl = 100 - entry_c
  settled_no  — resolved NO  before either level → pnl = -entry_c

Sweep mode runs step_c from 1 to max_step_c and aggregates across all tiers.

Usage (via main.py):
  python main.py ladder --asset ETH                        # default step=10, no stop
  python main.py ladder --asset ETH --step 5 --stop 20    # single step, stop=20c
  python main.py ladder --asset ETH --sweep --stop 20     # sweep steps 1-30, stop=20c
  python main.py ladder --asset BTC --sweep --stop 20 --step 40  # sweep 1-40
"""

from __future__ import annotations

import pandas as pd


def run_ladder_backtest(
    df: pd.DataFrame,
    step_c: int = 10,
    tolerance_c: int = 2,
    stop_loss_c: int = 0,
) -> pd.DataFrame:
    """
    For each entry tier (step_c to 99c at step_c intervals):
      1. Find first tick per market where ask is within ±tolerance_c of entry
      2. Scan subsequent ticks for first of: hit target, hit stop, or settle
    """
    step      = step_c / 100.0
    tolerance = tolerance_c / 100.0
    stop_loss = stop_loss_c / 100.0
    results   = []

    for entry_c in range(step_c, 100, step_c):
        entry  = entry_c / 100.0
        target = round(entry + step, 4)
        if target > 1.001:
            break

        stop_level = (entry - stop_loss) if stop_loss > 0 else -1.0

        n_entered     = 0
        n_hit_ticks   = 0
        n_stopped     = 0
        n_settled_yes = 0
        n_settled_no  = 0

        for _, mkt in df.groupby("ticker"):
            mkt = mkt.sort_values("tick_time").reset_index(drop=True)

            eligible = mkt[abs(mkt["ask"] - entry) <= tolerance]
            if eligible.empty:
                continue

            entry_idx = int(eligible.index[0])
            outcome   = int(mkt["outcome"].iloc[-1])
            n_entered += 1

            after    = mkt.iloc[entry_idx + 1:]
            ask_vals = after["ask"].values  # numpy; NaN comparisons → False (safe)

            hits_target = ask_vals >= target

            if stop_loss > 0 and stop_level > 0:
                hits_stop = ask_vals <= stop_level
                either    = hits_target | hits_stop
                if either.any():
                    first = int(either.argmax())
                    if hits_target[first]:
                        n_hit_ticks += 1
                    else:
                        n_stopped += 1
                elif outcome == 1:
                    n_settled_yes += 1
                else:
                    n_settled_no += 1
            else:
                if hits_target.any():
                    n_hit_ticks += 1
                elif outcome == 1:
                    n_settled_yes += 1
                else:
                    n_settled_no += 1

        n = n_entered
        if n == 0:
            continue

        exp_pnl_c = (
            n_hit_ticks   / n * step_c
            + n_stopped     / n * (-stop_loss_c)
            + n_settled_yes / n * (100 - entry_c)
            + n_settled_no  / n * (-entry_c)
        )

        results.append({
            "entry_c":      entry_c,
            "target_c":     int(round(target * 100)),
            "markets":      n,
            "hit_ticks":    n_hit_ticks,
            "hit_ticks%":   round(n_hit_ticks / n * 100, 1),
            "stopped":      n_stopped,
            "stopped%":     round(n_stopped / n * 100, 1),
            "settled_yes":  n_settled_yes,
            "settled_no":   n_settled_no,
            "exp_pnl_c":    round(exp_pnl_c, 1),
        })

    return pd.DataFrame(results)


def run_ladder_sweep(
    df: pd.DataFrame,
    max_step_c: int = 30,
    tolerance_c: int = 2,
    stop_loss_c: int = 20,
) -> pd.DataFrame:
    """
    Sweep step_c from 1 to max_step_c.
    Returns one row per step_c with stats aggregated (weighted) across all entry tiers.
    """
    rows = []
    for step_c in range(1, max_step_c + 1):
        result = run_ladder_backtest(
            df, step_c=step_c, tolerance_c=tolerance_c, stop_loss_c=stop_loss_c
        )
        if result.empty:
            continue

        total = int(result["markets"].sum())
        if total == 0:
            continue

        hit     = int(result["hit_ticks"].sum())
        stopped = int(result["stopped"].sum())
        s_yes   = int(result["settled_yes"].sum())
        s_no    = int(result["settled_no"].sum())

        exp_pnl_c = float((result["exp_pnl_c"] * result["markets"]).sum()) / total

        rows.append({
            "step_c":    step_c,
            "markets":   total,
            "hit%":      round(hit     / total * 100, 1),
            "stopped%":  round(stopped / total * 100, 1),
            "s_yes%":    round(s_yes   / total * 100, 1),
            "s_no%":     round(s_no    / total * 100, 1),
            "exp_pnl_c": round(exp_pnl_c, 1),
        })

    return pd.DataFrame(rows)


# ── Printing ───────────────────────────────────────────────────────────────

def print_ladder_results(
    results: pd.DataFrame,
    asset: str = "",
    step_c: int = 10,
    stop_loss_c: int = 0,
) -> None:
    if results.empty:
        print("  No data.")
        return

    stop_str = f", stop −{stop_loss_c}c" if stop_loss_c else ""
    has_stop = stop_loss_c > 0

    print(f"\n{'='*82}")
    print(f"  LADDER BACKTEST — {asset}   (buy at entry_c, sell at target_c{stop_str})")
    if has_stop:
        print(f"  hit_ticks = reached target  |  stopped = fell {stop_loss_c}c below entry")
    else:
        print(f"  hit_ticks = reached target in observed ticks (quick flip)")
    print(f"  settled_yes/no = held to resolution  |  exp_pnl_c = expected cents/trade")
    print(f"{'='*82}")

    if has_stop:
        print(
            f"  {'entry':>6}  {'target':>6}  {'mkts':>5}  "
            f"{'hit_tks':>7}  {'hit%':>5}  "
            f"{'stopd':>5}  {'stp%':>5}  "
            f"{'s_yes':>5}  {'s_no':>5}  {'exp_pnl_c':>9}"
        )
        print(f"  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*5}  "
              f"{'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*9}")
        for _, row in results.iterrows():
            flag = " ◀" if row["exp_pnl_c"] > 0 else ""
            print(
                f"  {int(row['entry_c']):>5}c  {int(row['target_c']):>5}c  "
                f"{int(row['markets']):>5}  "
                f"{int(row['hit_ticks']):>7}  {row['hit_ticks%']:>4.1f}%  "
                f"{int(row['stopped']):>5}  {row['stopped%']:>4.1f}%  "
                f"{int(row['settled_yes']):>5}  {int(row['settled_no']):>5}  "
                f"{row['exp_pnl_c']:>+8.1f}c{flag}"
            )
    else:
        print(
            f"  {'entry':>6}  {'target':>6}  {'mkts':>5}  "
            f"{'hit_tks':>7}  {'hit%':>5}  "
            f"{'s_yes':>5}  {'s_no':>5}  {'exp_pnl_c':>9}"
        )
        print(f"  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*9}")
        for _, row in results.iterrows():
            flag = " ◀" if row["exp_pnl_c"] > 0 else ""
            print(
                f"  {int(row['entry_c']):>5}c  {int(row['target_c']):>5}c  "
                f"{int(row['markets']):>5}  "
                f"{int(row['hit_ticks']):>7}  {row['hit_ticks%']:>4.1f}%  "
                f"{int(row['settled_yes']):>5}  {int(row['settled_no']):>5}  "
                f"{row['exp_pnl_c']:>+8.1f}c{flag}"
            )
    print()


def print_sweep_results(
    results: pd.DataFrame,
    asset: str = "",
    stop_loss_c: int = 20,
) -> None:
    if results.empty:
        print("  No data.")
        return

    max_step = int(results["step_c"].max())
    print(f"\n{'='*78}")
    print(f"  LADDER SWEEP — {asset}   stop=−{stop_loss_c}c  tolerance=±2c  steps 1→{max_step}")
    print(f"  exp_pnl_c = weighted avg across all entry tiers for that step size")
    print(f"  BE_hit% = breakeven hit rate = stop / (step + stop)")
    print(f"{'='*78}")
    print(
        f"  {'step':>4}  {'mkts':>5}  {'hit%':>6}  {'stop%':>6}  "
        f"{'s_yes%':>6}  {'s_no%':>6}  {'exp_pnl_c':>9}  {'BE_hit%':>7}"
    )
    print(f"  {'─'*4}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*7}")

    for _, row in results.iterrows():
        step_c = int(row["step_c"])
        be_hit = stop_loss_c / (step_c + stop_loss_c) * 100
        flag   = " ◀" if row["exp_pnl_c"] > 0 else ""
        print(
            f"  {step_c:>3}c  {int(row['markets']):>5}  "
            f"{row['hit%']:>5.1f}%  {row['stopped%']:>5.1f}%  "
            f"{row['s_yes%']:>5.1f}%  {row['s_no%']:>5.1f}%  "
            f"{row['exp_pnl_c']:>+8.1f}c  {be_hit:>6.1f}%"
            f"{flag}"
        )
    print()
