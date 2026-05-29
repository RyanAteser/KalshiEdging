"""
certainty_backtest.py — Z-score certainty map for Kalshi settlement outcomes.

The question: at a given z-score and time remaining, how certain is the
binary settlement outcome (YES or NO)?

Strategy:
  - High z (BTC well above strike) → buy YES and hold to settlement
  - Low z (BTC well below strike) → buy NO and hold to settlement
  - The z-score tells you HOW certain the outcome is, not whether it's mispriced

For each tick in the dataset:
  1. Compute z-score from 1m BTC price history
  2. Record: z, t_remaining, current ask, settlement outcome
  3. Bucket by (z_range, time_window) and show:
     - YES rate (% that settled YES) when z is positive
     - NO rate  (% that settled NO)  when z is negative
     - avg ask at time of observation
     - EV if you buy YES at current ask and hold to settlement
     - EV if you buy NO at (1 - ask) and hold to settlement

Output is a certainty map — find the (z, t) region where settlement is 95%+
certain, and the contract is still trading at a price that gives positive EV.

Usage:
  %PY% main.py certainty --asset BTC
  %PY% main.py certainty --asset BTC --side yes
  %PY% main.py certainty --asset BTC --side no
  %PY% main.py certainty --asset ETH
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.z_score import compute_z_score

FEE_CENTS = 7.0   # per side

BTC_HISTORY_WINDOW = 30   # 1m candles for vol history

# Z-score buckets (signed: positive = BTC above strike, negative = below)
Z_BUCKETS = [
    (-999, -8), (-8, -6), (-6, -5), (-5, -4), (-4, -3),
    (-3, -2), (-2, -1), (-1, 0),
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 999),
]

# Time buckets (seconds remaining)
T_BUCKETS = [
    (600, 900),   # 10–15 min left
    (300, 600),   # 5–10 min left
    (180, 300),   # 3–5 min left
    (90,  180),   # 1.5–3 min left
    (0,    90),   # final 90s
]

T_LABELS = ["10-15m", "5-10m", "3-5m", "1.5-3m", "<90s"]


def run_certainty_backtest(
    df: pd.DataFrame,
    btc_1m: pd.DataFrame | None = None,
    sample_every: int = 1,   # sample every N ticks per market (1 = all)
) -> pd.DataFrame:
    """
    For every tick (or sampled tick) across all markets, compute z-score
    and record the eventual settlement outcome.
    Returns a flat DataFrame of observations.
    """
    if btc_1m is not None and "tick_time" in df.columns:
        df = df.copy()
        df["tick_time"] = pd.to_datetime(df["tick_time"], utc=True)

    rows: list[dict] = []

    for _, mkt in df.groupby("ticker"):
        mkt     = mkt.sort_values("tick_time").reset_index(drop=True)
        outcome = int(mkt["outcome"].iloc[-1])
        n       = len(mkt)

        asks    = mkt["ask"].values * 100.0
        t_lefts = mkt["t_left"].values
        btc_arr = mkt["binance_price"].values
        strikes = mkt["strike"].values
        times   = mkt["tick_time"].values

        for i in range(1, n, sample_every):
            t_left = float(t_lefts[i])
            if t_left <= 0:
                continue

            btc    = float(btc_arr[i])
            strike = float(strikes[i])
            ask    = float(asks[i])

            # Build price history from 1m candles
            if btc_1m is not None:
                tick_ts  = pd.Timestamp(times[i], tz="UTC")
                mask     = btc_1m["open_time"] < tick_ts
                recent   = btc_1m.loc[mask, "close"].values[-BTC_HISTORY_WINDOW:]
                history  = list(recent) if len(recent) >= 5 else list(btc_arr[max(0,i-30):i])
            else:
                history = list(btc_arr[max(0, i - BTC_HISTORY_WINDOW):i])

            z, sigma = compute_z_score(
                btc_price       = btc,
                strike          = strike,
                t_remaining_sec = t_left,
                price_history   = history,
            )

            # EV of buying YES at current ask, holding to settlement
            ev_yes = outcome * (100 - ask) - (1 - outcome) * ask - FEE_CENTS * 2
            # EV of buying NO at (100 - ask), holding to settlement
            no_price = 100 - ask
            ev_no  = (1 - outcome) * (100 - no_price) - outcome * no_price - FEE_CENTS * 2

            rows.append({
                "z":       z,
                "t_left":  t_left,
                "ask":     ask,
                "outcome": outcome,
                "ev_yes":  ev_yes,
                "ev_no":   ev_no,
                "sigma":   sigma,
            })

    return pd.DataFrame(rows)


def build_certainty_map(
    obs: pd.DataFrame,
    side: str = "both",   # "yes" | "no" | "both"
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Aggregate observations into (z_bucket, t_bucket) cells.
    Reports YES rate, NO rate, avg ask, avg EV, obs count.
    """
    rows = []

    for t_lo, t_hi in T_BUCKETS:
        t_label = T_LABELS[T_BUCKETS.index((t_lo, t_hi))]
        t_mask  = (obs["t_left"] >= t_lo) & (obs["t_left"] < t_hi)
        t_slice = obs[t_mask]

        for z_lo, z_hi in Z_BUCKETS:
            z_mask  = (t_slice["z"] >= z_lo) & (t_slice["z"] < z_hi)
            cell    = t_slice[z_mask]

            if len(cell) < min_obs:
                continue

            yes_rate = cell["outcome"].mean()
            no_rate  = 1 - yes_rate
            avg_ask  = cell["ask"].mean()
            avg_ev_yes = cell["ev_yes"].mean()
            avg_ev_no  = cell["ev_no"].mean()

            # Best side and its EV
            if z_lo >= 0:
                best_side = "YES"
                cert      = yes_rate
                avg_ev    = avg_ev_yes
            else:
                best_side = "NO"
                cert      = no_rate
                avg_ev    = avg_ev_no

            rows.append({
                "t_window":   t_label,
                "t_lo":       t_lo,
                "z_lo":       z_lo,
                "z_hi":       z_hi,
                "z_label":    f"{z_lo:+.0f}→{z_hi:+.0f}" if z_hi < 999 else f">{z_lo:+.0f}",
                "obs":        len(cell),
                "yes_rate%":  round(yes_rate * 100, 1),
                "no_rate%":   round(no_rate  * 100, 1),
                "side":       best_side,
                "cert%":      round(cert * 100, 1),
                "avg_ask_c":  round(avg_ask, 1),
                "avg_ev_c":   round(avg_ev, 2),
            })

    return pd.DataFrame(rows).sort_values(["t_lo", "z_lo"], ascending=[False, True])


def print_certainty_map(results: pd.DataFrame, asset: str, side: str = "both") -> None:
    if results.empty:
        print("  No data.")
        return

    print(f"\n{'='*88}")
    print(f"  CERTAINTY MAP — {asset}   (z-score vs settlement outcome)")
    print(f"  cert% = how often the predicted side (YES if z>0, NO if z<0) was correct")
    print(f"  avg_ev_c = EV of buying that side at avg_ask, holding to settlement (fees included)")
    print(f"  ◀ = cert ≥ 90% AND ev > 0  ←  the actionable zone")
    print(f"{'='*88}")

    for t_window in results["t_window"].unique():
        t_rows = results[results["t_window"] == t_window]

        print(f"\n  ── Time remaining: {t_window} ──")
        print(
            f"  {'z range':>10}  {'obs':>5}  {'yes%':>6}  {'no%':>6}  "
            f"{'side':>4}  {'cert%':>6}  {'avg_ask':>8}  {'avg_ev_c':>9}"
        )
        print(f"  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*6}  "
              f"{'─'*4}  {'─'*6}  {'─'*8}  {'─'*9}")

        for _, row in t_rows.iterrows():
            if side == "yes" and row["z_lo"] < 0:
                continue
            if side == "no" and row["z_lo"] >= 0:
                continue

            cert = row["cert%"]
            ev   = row["avg_ev_c"]
            flag = " ◀" if cert >= 90 and ev > 0 else (
                   " --" if cert >= 90 else ""
            )
            print(
                f"  {row['z_label']:>10}  {int(row['obs']):>5}  "
                f"{row['yes_rate%']:>5.1f}%  {row['no_rate%']:>5.1f}%  "
                f"{row['side']:>4}  {cert:>5.1f}%  "
                f"{row['avg_ask_c']:>7.1f}c  {row['avg_ev_c']:>+8.2f}c{flag}"
            )

    print()
    print(f"  Key: ◀ = actionable (cert≥90%, ev>0 after fees)  | -- = certain but overpriced")
    print()
