"""
zscore_backtest.py — Z-score gated zone scalp backtest.

Replays the three validated zones (A/B/C) on historical tick data, but gates
each entry through the z-score filter. The key question: does z > threshold
actually improve hit rates over ungated entries?

Zones (hardcoded, validated across 200 markets / 2,999 ticks):
    A: buy 60c → target 65c, stop 55c  (hit 57.0%, ev +0.77c)
    B: buy 75c → target 80c, stop 70c  (hit 60.3%, ev +1.06c)
    C: buy 80c → target 85c, stop 75c  (hit 73.2%, ev +2.00c)

Entry conditions per tick (all must pass):
    1. ask within ±ENTRY_TOLERANCE_CENTS of zone entry
    2. prev_ask < zone entry  (approaching from below)
    3. ask <= zone entry + NO_CHASE_MAX_OVERSHOOT
    4. t_remaining >= MIN_SECONDS_REMAINING
    5. z_score >= z_threshold  (sweep this to find the calibration curve)

Outputs:
    - Ungated baseline: hit/stop rates per zone (sanity check vs scalp_backtest)
    - Z-gated results: hit/stop/filtered per zone at each z threshold
    - Calibration table: actual hit rate vs theoretical p_win at each z bucket
    - Filter efficiency: what % of trades are filtered, do filtered trades win less?

Usage:
    %PY% main.py zscore --asset BTC
    %PY% main.py zscore --asset BTC --z-min 2.0 --z-max 6.0
    %PY% main.py zscore --asset BTC --zone A
    %PY% main.py zscore --asset ETH
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.z_score import compute_z_score, p_win_from_z, Z_MIN_THRESHOLD

# ── Zone definitions ────────────────────────────────────────────────────────

ZONES = {
    "A": {"entry": 60.0, "target": 65.0, "stop": 55.0, "hit_pct": 0.570, "ev": 0.77},
    "B": {"entry": 75.0, "target": 80.0, "stop": 70.0, "hit_pct": 0.603, "ev": 1.06},
    "C": {"entry": 80.0, "target": 85.0, "stop": 75.0, "hit_pct": 0.732, "ev": 2.00},
}

ENTRY_TOLERANCE_CENTS    = 2.0
NO_CHASE_MAX_OVERSHOOT   = 2.0
MIN_SECONDS_REMAINING    = 90.0
COOLDOWN_TICKS_AFTER_EXIT = 3
FEE_PER_CONTRACT_CENTS   = 7.0   # each side

# How many prior BTC price ticks to accumulate before computing z-score.
# Dataset tick interval is ~15-30s, so 20 ticks ≈ 5-10 minutes of history.
BTC_HISTORY_WINDOW = 40


# ── Per-market simulation ───────────────────────────────────────────────────

def _simulate_market(
    ticks: pd.DataFrame,
    z_threshold: float,
) -> list[dict]:
    """
    Replay one market tick-by-tick.
    Returns list of trade records (one per closed position per zone).
    """
    outcome  = int(ticks["outcome"].iloc[-1])
    asks     = ticks["ask"].values * 100.0      # → cents
    btc_hist = ticks["binance_price"].values
    t_lefts  = ticks["t_left"].values           # seconds remaining
    strikes  = ticks["strike"].values

    # Per-zone state
    zone_keys  = list(ZONES.keys())
    in_trade   = {k: False for k in zone_keys}
    entry_ask  = {k: 0.0   for k in zone_keys}
    cooldown   = {k: 0     for k in zone_keys}

    trades: list[dict] = []
    n = len(asks)

    for i in range(1, n):
        ask      = asks[i]
        prev_ask = asks[i - 1]
        t_left   = float(t_lefts[i])
        btc      = float(btc_hist[i])
        strike   = float(strikes[i])

        # Build BTC price history window (prices up to but not including tick i)
        start  = max(0, i - BTC_HISTORY_WINDOW)
        btc_window = list(btc_hist[start:i])

        # Compute z-score once per tick (shared across zones)
        z, sigma = compute_z_score(
            btc_price       = btc,
            strike          = strike,
            t_remaining_sec = t_left,
            price_history   = btc_window,
        )

        for zname, zdef in ZONES.items():
            entry  = zdef["entry"]
            target = zdef["target"]
            stop   = zdef["stop"]

            # Cooldown countdown
            if cooldown[zname] > 0:
                cooldown[zname] -= 1
                continue

            # In trade — check exits
            if in_trade[zname]:
                if ask >= target:
                    trades.append({
                        "zone": zname, "result": "hit",
                        "entry_c": entry_ask[zname], "exit_c": ask,
                        "pnl_c": target - entry_ask[zname],
                        "z_score": None,   # z at entry, stored below
                        "sigma": None,
                        "t_entry": None,
                        "outcome": outcome,
                        "z_at_entry": _trade_z_scratch[zname],
                        "sigma_at_entry": _trade_sigma_scratch[zname],
                        "t_at_entry": _trade_t_scratch[zname],
                    })
                    in_trade[zname]  = False
                    cooldown[zname]  = COOLDOWN_TICKS_AFTER_EXIT
                elif ask <= stop:
                    trades.append({
                        "zone": zname, "result": "stopped",
                        "entry_c": entry_ask[zname], "exit_c": ask,
                        "pnl_c": stop - entry_ask[zname],
                        "z_at_entry": _trade_z_scratch[zname],
                        "sigma_at_entry": _trade_sigma_scratch[zname],
                        "t_at_entry": _trade_t_scratch[zname],
                        "outcome": outcome,
                    })
                    in_trade[zname]  = False
                    cooldown[zname]  = COOLDOWN_TICKS_AFTER_EXIT
                continue

            # Idle — check entry conditions
            if t_left < MIN_SECONDS_REMAINING:
                continue
            if ask > entry + NO_CHASE_MAX_OVERSHOOT:
                continue
            if not (abs(ask - entry) <= ENTRY_TOLERANCE_CENTS):
                continue
            if prev_ask >= entry:   # must approach from below
                continue

            # Entry candidate — evaluate z-score gate
            passes_z = z >= z_threshold

            # Record candidate (whether filtered or not) for calibration
            trades.append({
                "zone": zname, "result": "filtered" if not passes_z else "__pending__",
                "entry_c": ask, "exit_c": None,
                "pnl_c": None,
                "z_at_entry": z,
                "sigma_at_entry": sigma,
                "t_at_entry": t_left,
                "outcome": outcome,
            })

            if passes_z:
                # Enter trade — remove the pending marker, update in-flight state
                trades[-1]["result"] = "__entered__"  # will be resolved on exit
                in_trade[zname]  = True
                entry_ask[zname] = ask
                _trade_z_scratch[zname]     = z
                _trade_sigma_scratch[zname] = sigma
                _trade_t_scratch[zname]     = t_left

    # Positions still open at settlement
    for zname in zone_keys:
        if in_trade[zname]:
            settle_c = 100.0 if outcome == 1 else 0.0
            pnl_c    = settle_c - entry_ask[zname]
            trades.append({
                "zone": zname, "result": "settled",
                "entry_c": entry_ask[zname], "exit_c": settle_c,
                "pnl_c": pnl_c,
                "z_at_entry": _trade_z_scratch[zname],
                "sigma_at_entry": _trade_sigma_scratch[zname],
                "t_at_entry": _trade_t_scratch[zname],
                "outcome": outcome,
            })

    # Remove __entered__ placeholders (resolved trades are logged on exit)
    return [t for t in trades if t["result"] != "__entered__"]


# Module-level scratch dicts (reset per market call, safe for single-threaded)
_trade_z_scratch:     dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}
_trade_sigma_scratch: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}
_trade_t_scratch:     dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}


def _reset_scratch() -> None:
    for k in ("A", "B", "C"):
        _trade_z_scratch[k]     = 0.0
        _trade_sigma_scratch[k] = 0.0
        _trade_t_scratch[k]     = 0.0


# ── Full backtest across all markets ───────────────────────────────────────

def run_zscore_backtest(
    df: pd.DataFrame,
    z_threshold: float = Z_MIN_THRESHOLD,
    zone_filter: str | None = None,    # None = all zones
) -> pd.DataFrame:
    """
    Run z-gated zone backtest across all markets.
    Returns DataFrame of all trade records (including filtered ones).
    """
    all_records: list[dict] = []

    for _, mkt in df.groupby("ticker"):
        mkt = mkt.sort_values("tick_time").reset_index(drop=True)
        if len(mkt) < 5:
            continue
        _reset_scratch()
        records = _simulate_market(mkt, z_threshold=z_threshold)
        if zone_filter:
            records = [r for r in records if r["zone"] == zone_filter]
        all_records.extend(records)

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def run_zscore_sweep(
    df: pd.DataFrame,
    z_values: list[float] | None = None,
    zone_filter: str | None = None,
) -> pd.DataFrame:
    """
    Sweep z thresholds, showing how gate tightness affects hit rate and trade count.
    """
    if z_values is None:
        z_values = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    rows = []
    for z_thresh in z_values:
        records = run_zscore_backtest(df, z_threshold=z_thresh, zone_filter=zone_filter)
        if records.empty:
            continue

        entered  = records[records["result"].isin(["hit", "stopped", "settled"])]
        filtered = records[records["result"] == "filtered"]

        n_entered  = len(entered)
        n_filtered = len(filtered)
        n_total    = n_entered + n_filtered

        if n_entered == 0:
            continue

        hits    = (entered["result"] == "hit").sum()
        stops   = (entered["result"] == "stopped").sum()
        settles = (entered["result"] == "settled").sum()
        pnls    = entered["pnl_c"].dropna()

        mean_pnl = float(pnls.mean()) if len(pnls) else 0.0
        std_pnl  = float(pnls.std())  if len(pnls) > 1 else 1e-8
        sharpe   = mean_pnl / std_pnl * (n_entered ** 0.5)

        fee_adj_pnl = mean_pnl - FEE_PER_CONTRACT_CENTS * 2

        # Avg z and theoretical p_win for entered trades
        z_entered = entered["z_at_entry"].dropna()
        avg_z     = float(z_entered.mean()) if len(z_entered) else 0.0
        theory_p  = p_win_from_z(avg_z) * 100 if avg_z > 0 else 0.0

        rows.append({
            "z_thresh":    z_thresh,
            "entered":     n_entered,
            "filtered":    n_filtered,
            "filter_pct":  round(n_filtered / n_total * 100, 1) if n_total else 0.0,
            "hit%":        round(hits    / n_entered * 100, 1),
            "stop%":       round(stops   / n_entered * 100, 1),
            "settle%":     round(settles / n_entered * 100, 1),
            "avg_z":       round(avg_z, 2),
            "theory_p%":   round(theory_p, 1),
            "exp_pnl_c":   round(mean_pnl, 2),
            "fee_adj_pnl": round(fee_adj_pnl, 2),
            "sharpe":      round(sharpe, 3),
            "total_pnl_c": round(float(pnls.sum()), 1),
        })

    return pd.DataFrame(rows)


# ── Printing ────────────────────────────────────────────────────────────────

def print_zscore_sweep(
    results: pd.DataFrame,
    asset: str,
    zone_filter: str | None = None,
) -> None:
    if results.empty:
        print("  No results.")
        return

    zone_str = f"zone {zone_filter}" if zone_filter else "all zones"
    print(f"\n{'='*100}")
    print(f"  Z-SCORE GATE SWEEP — {asset}   {zone_str}")
    print(f"  Does a higher z threshold actually improve hit rates?")
    print(f"  theory_p% = N(avg_z) theoretical win probability | fee_adj = pnl − 14c fees")
    print(f"{'='*100}")
    print(
        f"  {'z≥':>5}  {'entered':>8}  {'filterd':>8}  {'filt%':>6}  "
        f"{'hit%':>6}  {'stop%':>6}  {'setl%':>6}  "
        f"{'avg_z':>6}  {'theory%':>8}  {'exp_pnl':>8}  {'fee_adj':>8}  "
        f"{'sharpe':>7}  {'tot_pnl':>9}"
    )
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  "
          f"{'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*9}")

    for _, row in results.iterrows():
        flag = " ◀" if row["fee_adj_pnl"] > 0 else ""
        z_flag = " ★" if row["z_thresh"] == Z_MIN_THRESHOLD else ""
        print(
            f"  {row['z_thresh']:>4.1f}  {int(row['entered']):>8,}  {int(row['filtered']):>8,}  "
            f"{row['filter_pct']:>5.1f}%  "
            f"{row['hit%']:>5.1f}%  {row['stop%']:>5.1f}%  {row['settle%']:>5.1f}%  "
            f"{row['avg_z']:>6.2f}  {row['theory_p%']:>7.1f}%  "
            f"{row['exp_pnl_c']:>+7.2f}c  {row['fee_adj_pnl']:>+7.2f}c  "
            f"{row['sharpe']:>+6.3f}  {row['total_pnl_c']:>+8.1f}c{flag}{z_flag}"
        )
    print(f"\n  ★ = spec default (z≥{Z_MIN_THRESHOLD})")
    print()


def print_zscore_per_zone(
    df: pd.DataFrame,
    asset: str,
    z_threshold: float,
) -> None:
    """Show hit/stop breakdown per zone at a single z threshold."""
    entered = df[df["result"].isin(["hit", "stopped", "settled"])].copy()
    if entered.empty:
        print("  No trades entered at this threshold.")
        return

    print(f"\n  Per-zone breakdown at z≥{z_threshold} — {asset}")
    print(f"  {'zone':>5}  {'trades':>7}  {'hit%':>6}  {'stop%':>6}  "
          f"{'avg_z':>6}  {'exp_pnl':>8}  {'fee_adj':>8}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")

    for zname in ("A", "B", "C"):
        z_trades = entered[entered["zone"] == zname]
        if z_trades.empty:
            continue
        n    = len(z_trades)
        hits = (z_trades["result"] == "hit").sum()
        stops = (z_trades["result"] == "stopped").sum()
        pnls = z_trades["pnl_c"].dropna()
        avg_z    = float(z_trades["z_at_entry"].dropna().mean()) if len(z_trades) else 0.0
        mean_pnl = float(pnls.mean()) if len(pnls) else 0.0
        fee_adj  = mean_pnl - FEE_PER_CONTRACT_CENTS * 2
        flag = " ◀" if fee_adj > 0 else ""
        print(
            f"  {zname:>5}  {n:>7,}  {hits/n*100:>5.1f}%  {stops/n*100:>5.1f}%  "
            f"{avg_z:>6.2f}  {mean_pnl:>+7.2f}c  {fee_adj:>+7.2f}c{flag}"
        )

    # Filtered trades — what would their outcome have been?
    filtered = df[df["result"] == "filtered"]
    if not filtered.empty:
        print(f"\n  Filtered trades ({len(filtered)}) — hypothetical outcome if allowed:")
        print(f"  (these were blocked by z-gate; outcome inferred from settlement)")
        settle_win = (filtered["outcome"] == 1).sum()
        settle_lose = (filtered["outcome"] == 0).sum()
        print(f"  market resolved YES: {settle_win}  |  NO: {settle_lose}  "
              f"({settle_win/(len(filtered))*100:.1f}% would have settled YES)")
        avg_z_filt = float(filtered["z_at_entry"].dropna().mean())
        print(f"  avg z of filtered entries: {avg_z_filt:.2f}  "
              f"(blocked because < {z_threshold})")
    print()
