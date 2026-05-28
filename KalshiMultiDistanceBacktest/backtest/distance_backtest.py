"""
distance_backtest.py — Exact distance method backtest for Kalshi 15-minute markets.

Strategy:
  - Enter (buy YES) when  binance_price - strike  >=  entry_dist   (price is above strike)
  - Enter (buy NO)  when  strike - binance_price  >=  entry_dist   (price is below strike)
  - Exit (stop)     when  abs(binance_price - strike) < stop_dist
  - Exit (settled)  at the market's binary outcome (1.0 or 0.0)

PnL is in Kalshi dollar terms:
  - Entry at ask price
  - Stop exit at bid price
  - Settlement exit at outcome (1.0 win or 0.0 loss)

Columns required in the dataset DataFrame:
  binance_price, strike, ask, bid, t_left, outcome, ticker, tick_time
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


# ── Single-market simulation ───────────────────────────────────────────────

def _simulate_market(
    market_df: pd.DataFrame,
    entry_dist: float,
    stop_dist: float,
) -> List[dict]:
    """
    Replay one market's ticks through the distance strategy.
    Returns list of trade result dicts.
    """
    trades = []

    in_trade     = False
    entry_price  = 0.0
    entry_side   = ""   # "YES" or "NO"
    entry_tick   = None

    rows = market_df.sort_values("tick_time").reset_index(drop=True)

    for _, row in rows.iterrows():
        price  = row["binance_price"]
        strike = row["strike"]
        ask    = row["ask"]
        bid    = row["bid"]
        t_left = row["t_left"]
        outcome = int(row["outcome"])

        dist = abs(price - strike)

        if not in_trade:
            # Entry signal
            if dist >= entry_dist and ask is not None and not pd.isna(ask):
                in_trade    = True
                entry_price = float(ask)
                entry_side  = "YES" if price > strike else "NO"
                entry_tick  = row
        else:
            # Stop-loss: distance has shrunk below stop_dist
            if dist < stop_dist and bid is not None and not pd.isna(bid):
                exit_price = float(bid)
                pnl        = round(exit_price - entry_price, 6)
                trades.append({
                    "ticker":       row["ticker"],
                    "entry_dist":   entry_dist,
                    "stop_dist":    stop_dist,
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "exit_reason":  "stop",
                    "side":         entry_side,
                    "pnl":          pnl,
                    "win":          int(pnl > 0),
                    "t_left_entry": float(entry_tick["t_left"]) if entry_tick is not None else 0.0,
                })
                in_trade   = False
                entry_tick = None

    # If still in trade at end of market data → settled
    if in_trade and entry_tick is not None:
        settle_price = 1.0 if (
            (entry_side == "YES" and outcome == 1) or
            (entry_side == "NO"  and outcome == 0)
        ) else 0.0
        pnl = round(settle_price - entry_price, 6)
        trades.append({
            "ticker":       rows.iloc[-1]["ticker"],
            "entry_dist":   entry_dist,
            "stop_dist":    stop_dist,
            "entry_price":  entry_price,
            "exit_price":   settle_price,
            "exit_reason":  "settled",
            "side":         entry_side,
            "pnl":          pnl,
            "win":          int(pnl > 0),
            "t_left_entry": float(entry_tick["t_left"]),
        })

    return trades


# ── Full threshold sweep ───────────────────────────────────────────────────

def run_distance_backtest(
    df: pd.DataFrame,
    thresholds: List[float],
    stop_dist: float,
) -> pd.DataFrame:
    """
    Run the distance strategy across all entry_dist thresholds.

    Args:
        df:          Dataset DataFrame (output of build_asset).
        thresholds:  List of entry_dist values to sweep.
        stop_dist:   Stop-loss distance.

    Returns:
        Summary DataFrame with one row per threshold.
    """
    required = {"binance_price", "strike", "ask", "bid", "t_left", "outcome", "ticker", "tick_time"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    summary_rows = []

    for entry_dist in thresholds:
        all_trades: List[dict] = []

        for ticker, mkt_df in df.groupby("ticker"):
            trades = _simulate_market(mkt_df, entry_dist, stop_dist)
            all_trades.extend(trades)

        if not all_trades:
            summary_rows.append({
                "entry_dist": entry_dist,
                "stop_dist":  stop_dist,
                "trades":     0,
                "win_rate":   0.0,
                "stop_rate":  0.0,
                "avg_pnl":    0.0,
                "total_pnl":  0.0,
            })
            continue

        trades_df  = pd.DataFrame(all_trades)
        n          = len(trades_df)
        wins       = trades_df["win"].sum()
        stops      = (trades_df["exit_reason"] == "stop").sum()
        avg_pnl    = trades_df["pnl"].mean()
        total_pnl  = trades_df["pnl"].sum()

        summary_rows.append({
            "entry_dist": entry_dist,
            "stop_dist":  stop_dist,
            "trades":     n,
            "win_rate":   round(wins / n * 100, 2) if n else 0.0,
            "stop_rate":  round(stops / n * 100, 2) if n else 0.0,
            "avg_pnl":    round(avg_pnl, 6),
            "total_pnl":  round(total_pnl, 4),
        })

    return pd.DataFrame(summary_rows)


# ── Printing ───────────────────────────────────────────────────────────────

def _best_row(results: pd.DataFrame) -> int:
    """Return index of best threshold: highest avg_pnl among rows with >=10 trades.
    Falls back to >=5, then any trade, to avoid picking 1-trade outliers."""
    for min_t in (10, 5, 1):
        q = results[results["trades"] >= min_t]
        if not q.empty:
            return q["avg_pnl"].idxmax()
    return results.index[0]


def print_results(results: pd.DataFrame) -> None:
    if results.empty or results["trades"].sum() == 0:
        print("  No trades generated.")
        return

    best_idx = _best_row(results)

    print(
        f"  {'entry_dist':>10}  {'trades':>6}  {'win_rate':>8}  "
        f"{'stop_rate':>9}  {'avg_pnl':>8}  {'total_pnl':>10}"
    )
    print(f"  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*10}")

    for idx, row in results.iterrows():
        marker = " ←best" if idx == best_idx else ""
        print(
            f"  {row['entry_dist']:>10.4g}  {int(row['trades']):>6}  "
            f"{row['win_rate']:>7.1f}%  {row['stop_rate']:>8.1f}%  "
            f"{row['avg_pnl']:>+8.5f}  {row['total_pnl']:>+10.4f}"
            f"{marker}"
        )
