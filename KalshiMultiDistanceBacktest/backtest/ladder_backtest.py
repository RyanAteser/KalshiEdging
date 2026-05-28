"""
ladder_backtest.py — Price-ladder (momentum) analysis for Kalshi YES/NO contracts.

Entry logic: ask within ±tolerance_c of entry level.

Momentum filters (look back `lookback_ticks` ticks before entry):
  from_below_c > 0  → YES trade: ask rose from ≤ (entry − from_below_c)
  from_above_c > 0  → NO  trade: ask fell from ≥ (entry + from_above_c)
  Both set          → trades both directions at each tier

Four outcomes per trade (YES or NO, same pnl formula in either direction):
  hit_ticks   → pnl = +step_c
  stopped     → pnl = -stop_loss_c
  settled_yes → YES: +(100−entry_c)   NO: −(100−entry_c)
  settled_no  → YES: −entry_c         NO: +entry_c

Usage (via main.py):
  python main.py ladder --asset ETH
  python main.py ladder --asset ETH --step 10 --stop 10 --from-below 10
  python main.py ladder --asset ETH --step 10 --stop 10 --from-below 10 --from-above 10
  python main.py ladder --asset ETH --sweep --stop 10 --from-below 10 --from-above 10
"""

from __future__ import annotations

import pandas as pd


def _resolve_trade(
    after_asks,          # numpy array of ask prices after entry
    target: float,
    stop_level: float,   # -1 means no stop (YES) or 2.0 (NO, meaning never triggered)
    stop_active: bool,
    outcome: int,
    direction: str,      # "YES" or "NO"
) -> str:
    """Return 'hit', 'stopped', 'settled_yes', or 'settled_no'."""
    if direction == "YES":
        hits_target = after_asks >= target
        if stop_active and stop_level > 0:
            hits_stop = after_asks <= stop_level
            either = hits_target | hits_stop
            if either.any():
                return "hit" if hits_target[int(either.argmax())] else "stopped"
        else:
            if hits_target.any():
                return "hit"
        return "settled_yes" if outcome == 1 else "settled_no"
    else:  # NO: looking for price to DROP
        hits_target = after_asks <= target
        if stop_active:
            hits_stop = after_asks >= stop_level  # YES rises = NO stopped
            either = hits_target | hits_stop
            if either.any():
                return "hit" if hits_target[int(either.argmax())] else "stopped"
        else:
            if hits_target.any():
                return "hit"
        return "settled_yes" if outcome == 1 else "settled_no"  # consistent: settled_yes = market resolved YES


def run_ladder_backtest(
    df: pd.DataFrame,
    step_c: int = 10,
    tolerance_c: int = 2,
    stop_loss_c: int = 0,
    from_below_c: int = 0,
    from_above_c: int = 0,
    lookback_ticks: int = 10,
) -> pd.DataFrame:
    """
    For each entry tier (step_c to 99c at step_c intervals):
      YES trades: ask near entry AND rose from ≤ (entry − from_below_c)
      NO  trades: ask near entry AND fell from ≥ (entry + from_above_c)
      (if from_below_c=0 and from_above_c=0: all ticks, YES trades only)
    """
    step       = step_c / 100.0
    tolerance  = tolerance_c / 100.0
    stop_loss  = stop_loss_c / 100.0
    from_below = from_below_c / 100.0
    from_above = from_above_c / 100.0
    do_yes     = True          # YES side: always on (unless only from_above set)
    do_no      = from_above_c > 0
    # If only from_above is set (no from_below), suppress unfiltered YES trades
    if from_above_c > 0 and from_below_c == 0:
        do_yes = False

    results = []

    for entry_c in range(step_c, 100, step_c):
        entry      = entry_c / 100.0
        target_yes = round(entry + step, 4)
        target_no  = round(entry - step, 4)

        if target_yes > 1.001:
            break

        stop_yes   = (entry - stop_loss) if stop_loss > 0 else -1.0
        stop_no    = (entry + stop_loss) if stop_loss > 0 else  2.0

        # Counters: YES and NO tracked separately, combined at end
        n_yes = 0;  yes_hit = 0;  yes_stop = 0;  yes_sy = 0;  yes_sn = 0
        n_no  = 0;  no_hit  = 0;  no_stop  = 0;  no_sy  = 0;  no_sn  = 0

        for _, mkt in df.groupby("ticker"):
            mkt     = mkt.sort_values("tick_time").reset_index(drop=True)
            outcome = int(mkt["outcome"].iloc[-1])
            elig    = mkt.index[abs(mkt["ask"] - entry) <= tolerance].tolist()
            if not elig:
                continue

            yes_done = False
            no_done  = False

            for idx in elig:
                if yes_done and no_done:
                    break

                start = max(0, idx - lookback_ticks)
                prior = mkt["ask"].iloc[start:idx].dropna().values
                after_asks = mkt.iloc[idx + 1:]["ask"].values

                # YES trade: price rose through entry from below
                if do_yes and not yes_done:
                    qualifies = (
                        from_below == 0  # no filter
                        or (len(prior) > 0 and (prior <= entry - from_below).any())
                    )
                    if qualifies:
                        yes_done = True
                        n_yes += 1
                        result = _resolve_trade(
                            after_asks, target_yes, stop_yes,
                            stop_loss > 0, outcome, "YES"
                        )
                        if result == "hit":             yes_hit  += 1
                        elif result == "stopped":       yes_stop += 1
                        elif result == "settled_yes":   yes_sy   += 1
                        else:                           yes_sn   += 1

                # NO trade: price fell through entry from above
                if do_no and not no_done and target_no >= 0:
                    qualifies = (
                        len(prior) > 0 and (prior >= entry + from_above).any()
                    )
                    if qualifies:
                        no_done = True
                        n_no += 1
                        result = _resolve_trade(
                            after_asks, target_no, stop_no,
                            stop_loss > 0, outcome, "NO"
                        )
                        if result == "hit":             no_hit  += 1
                        elif result == "stopped":       no_stop += 1
                        elif result == "settled_yes":   no_sy   += 1
                        else:                           no_sn   += 1

        n = n_yes + n_no
        if n == 0:
            continue

        n_hit     = yes_hit  + no_hit
        n_stopped = yes_stop + no_stop

        # Settlement pnl differs by side
        exp_pnl_c = (
            n_hit     / n * step_c
            + n_stopped / n * (-stop_loss_c)
            + yes_sy  / n * (100 - entry_c)
            + yes_sn  / n * (-entry_c)
            + no_sy   / n * (-(100 - entry_c))
            + no_sn   / n * entry_c
        )

        results.append({
            "entry_c":    entry_c,
            "target_c":   int(round(target_yes * 100)),
            "markets":    n,
            "yes_trades": n_yes,
            "no_trades":  n_no,
            "hit_ticks":  n_hit,
            "hit_ticks%": round(n_hit   / n * 100, 1),
            "stopped":    n_stopped,
            "stopped%":   round(n_stopped / n * 100, 1),
            "settled_yes": yes_sy + no_sy,
            "settled_no":  yes_sn + no_sn,
            "exp_pnl_c":  round(exp_pnl_c, 1),
        })

    return pd.DataFrame(results)


def run_ladder_sweep(
    df: pd.DataFrame,
    max_step_c: int = 30,
    tolerance_c: int = 2,
    stop_loss_c: int = 20,
    from_below_c: int = 0,
    from_above_c: int = 0,
    lookback_ticks: int = 10,
) -> pd.DataFrame:
    """Sweep step_c 1→max_step_c, aggregate across all entry tiers."""
    rows = []
    for step_c in range(1, max_step_c + 1):
        result = run_ladder_backtest(
            df,
            step_c=step_c,
            tolerance_c=tolerance_c,
            stop_loss_c=stop_loss_c,
            from_below_c=from_below_c,
            from_above_c=from_above_c,
            lookback_ticks=lookback_ticks,
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
            "yes_trades": int(result["yes_trades"].sum()),
            "no_trades":  int(result["no_trades"].sum()),
            "hit%":      round(hit     / total * 100, 1),
            "stopped%":  round(stopped / total * 100, 1),
            "s_yes%":    round(s_yes   / total * 100, 1),
            "s_no%":     round(s_no    / total * 100, 1),
            "exp_pnl_c": round(exp_pnl_c, 1),
        })
    return pd.DataFrame(rows)


# ── Printing ───────────────────────────────────────────────────────────────

def _header_str(stop_loss_c, from_below_c, from_above_c):
    parts = []
    if stop_loss_c:
        parts.append(f"stop −{stop_loss_c}c")
    if from_below_c and from_above_c:
        parts.append(f"momentum ±{from_below_c}c (YES↑ & NO↓)")
    elif from_below_c:
        parts.append(f"from-below {from_below_c}c (YES↑ only)")
    elif from_above_c:
        parts.append(f"from-above {from_above_c}c (NO↓ only)")
    return ("  " + ", ".join(parts)) if parts else ""


def print_ladder_results(
    results: pd.DataFrame,
    asset: str = "",
    step_c: int = 10,
    stop_loss_c: int = 0,
    from_below_c: int = 0,
    from_above_c: int = 0,
) -> None:
    if results.empty:
        print("  No data.")
        return

    hdr = _header_str(stop_loss_c, from_below_c, from_above_c)
    both_sides = from_below_c > 0 and from_above_c > 0
    has_stop   = stop_loss_c > 0

    print(f"\n{'='*88}")
    print(f"  LADDER BACKTEST — {asset}{hdr}")
    print(f"  hit = reached target  |  stop = reversed {stop_loss_c}c against entry" if has_stop
          else f"  hit = reached target in observed ticks")
    print(f"  settled_yes/no = held to resolution  |  exp_pnl_c = expected cents/trade")
    print(f"{'='*88}")

    if both_sides:
        print(
            f"  {'entry':>6}  {'±target':>7}  {'yes':>4}  {'no':>4}  "
            f"{'hit_tks':>7}  {'hit%':>5}  "
            f"{'stopd':>5}  {'stp%':>5}  "
            f"{'s_yes':>5}  {'s_no':>5}  {'exp_pnl_c':>9}"
        )
        print(f"  {'─'*6}  {'─'*7}  {'─'*4}  {'─'*4}  {'─'*7}  {'─'*5}  "
              f"{'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*9}")
        for _, row in results.iterrows():
            flag = " ◀" if row["exp_pnl_c"] > 0 else ""
            print(
                f"  {int(row['entry_c']):>5}c  "
                f"±{int(row['target_c'] - row['entry_c']):>5}c  "
                f"{int(row['yes_trades']):>4}  {int(row['no_trades']):>4}  "
                f"{int(row['hit_ticks']):>7}  {row['hit_ticks%']:>4.1f}%  "
                f"{int(row['stopped']):>5}  {row['stopped%']:>4.1f}%  "
                f"{int(row['settled_yes']):>5}  {int(row['settled_no']):>5}  "
                f"{row['exp_pnl_c']:>+8.1f}c{flag}"
            )
    else:
        print(
            f"  {'entry':>6}  {'target':>6}  {'mkts':>5}  "
            f"{'hit_tks':>7}  {'hit%':>5}  "
            + (f"{'stopd':>5}  {'stp%':>5}  " if has_stop else "")
            + f"{'s_yes':>5}  {'s_no':>5}  {'exp_pnl_c':>9}"
        )
        sep = f"  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*5}  "
        sep += (f"{'─'*5}  {'─'*5}  " if has_stop else "")
        sep += f"{'─'*5}  {'─'*5}  {'─'*9}"
        print(sep)
        for _, row in results.iterrows():
            flag = " ◀" if row["exp_pnl_c"] > 0 else ""
            line = (
                f"  {int(row['entry_c']):>5}c  {int(row['target_c']):>5}c  "
                f"{int(row['markets']):>5}  "
                f"{int(row['hit_ticks']):>7}  {row['hit_ticks%']:>4.1f}%  "
            )
            if has_stop:
                line += f"{int(row['stopped']):>5}  {row['stopped%']:>4.1f}%  "
            line += (
                f"{int(row['settled_yes']):>5}  {int(row['settled_no']):>5}  "
                f"{row['exp_pnl_c']:>+8.1f}c{flag}"
            )
            print(line)
    print()


def print_sweep_results(
    results: pd.DataFrame,
    asset: str = "",
    stop_loss_c: int = 20,
    from_below_c: int = 0,
    from_above_c: int = 0,
) -> None:
    if results.empty:
        print("  No data.")
        return

    max_step  = int(results["step_c"].max())
    hdr = _header_str(stop_loss_c, from_below_c, from_above_c)
    both_sides = from_below_c > 0 and from_above_c > 0

    print(f"\n{'='*82}")
    print(f"  LADDER SWEEP — {asset}   steps 1→{max_step}{hdr}")
    print(f"  exp_pnl_c = weighted avg across all entry tiers | BE_hit% = stop/(step+stop)")
    print(f"{'='*82}")

    if both_sides:
        print(
            f"  {'step':>4}  {'yes':>5}  {'no':>5}  {'hit%':>6}  {'stop%':>6}  "
            f"{'s_yes%':>6}  {'s_no%':>6}  {'exp_pnl_c':>9}  {'BE_hit%':>7}"
        )
        print(f"  {'─'*4}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*6}  "
              f"{'─'*6}  {'─'*6}  {'─'*9}  {'─'*7}")
        for _, row in results.iterrows():
            step_c = int(row["step_c"])
            be_hit = stop_loss_c / (step_c + stop_loss_c) * 100 if stop_loss_c else 0
            flag   = " ◀" if row["exp_pnl_c"] > 0 else ""
            print(
                f"  {step_c:>3}c  {int(row['yes_trades']):>5}  {int(row['no_trades']):>5}  "
                f"{row['hit%']:>5.1f}%  {row['stopped%']:>5.1f}%  "
                f"{row['s_yes%']:>5.1f}%  {row['s_no%']:>5.1f}%  "
                f"{row['exp_pnl_c']:>+8.1f}c  {be_hit:>6.1f}%{flag}"
            )
    else:
        print(
            f"  {'step':>4}  {'mkts':>5}  {'hit%':>6}  {'stop%':>6}  "
            f"{'s_yes%':>6}  {'s_no%':>6}  {'exp_pnl_c':>9}  {'BE_hit%':>7}"
        )
        print(f"  {'─'*4}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*7}")
        for _, row in results.iterrows():
            step_c = int(row["step_c"])
            be_hit = stop_loss_c / (step_c + stop_loss_c) * 100 if stop_loss_c else 0
            flag   = " ◀" if row["exp_pnl_c"] > 0 else ""
            print(
                f"  {step_c:>3}c  {int(row['markets']):>5}  "
                f"{row['hit%']:>5.1f}%  {row['stopped%']:>5.1f}%  "
                f"{row['s_yes%']:>5.1f}%  {row['s_no%']:>5.1f}%  "
                f"{row['exp_pnl_c']:>+8.1f}c  {be_hit:>6.1f}%{flag}"
            )
    print()
