"""
ml/backtest.py — Strategy backtester for BTC 15m binary contracts.

Two modes:
  1. Single entry point  (--entry-second 750)
     Detailed win-rate tables for one specific entry time.

  2. Sweep mode  (--sweep)
     Loads data ONCE, then shows a full grid of win rate + EV across
     every combination of entry time × minimum entry price. Use this
     to find the optimal (when to enter, what price to require).

Usage (Windows):
    # Single time analysis
    python ml/backtest.py ^
        --prices "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip" ^
                 "E:\\prices_btc_15m_2026-04-28_2026-05-05.zip" ^
        --entry-second 750 --output backtest.csv

    # Full sweep (recommended — one run covers all combos)
    python ml/backtest.py ^
        --prices "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip" ^
                 "E:\\prices_btc_15m_2026-04-28_2026-05-05.zip" ^
                 "E:\\prices_btc_15m_2026-05-06_2026-05-12.zip" ^
                 "E:\\prices_btc_15m_2026-05-13_2026-05-18.zip" ^
        --sweep
"""

from __future__ import annotations

import argparse
import gc
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

SETTLED_THRESH = 0.90
MIN_TRADES     = 10   # hide cells with fewer trades

# Seconds to sample for sweep mode
SWEEP_SECONDS  = [300, 540, 600, 660, 720, 750, 780, 810, 840, 870]

# Min-price thresholds to test in sweep mode
SWEEP_PRICES   = [0.80, 0.85, 0.875, 0.90, 0.915, 0.93, 0.945, 0.96]


# ── Strike parsing ────────────────────────────────────────────────────────────

_STRIKE_PATTERNS = [
    r'above[_-]?(\d{4,6})',
    r'below[_-]?(\d{4,6})',
    r'[tTbB](\d{4,6})(?:[^0-9]|$)',
    r'[-_](\d{5,6})[-_uU]',
    r'(\d{5,6})(?:usd|USD)?(?:[-_k]|$)',
    r'btc.*?(\d{4,6})',
]

def _parse_strike(slug: str) -> float | None:
    for pat in _STRIKE_PATTERNS:
        m = re.search(pat, slug, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 10_000 <= v <= 200_000:
                return v
    return None


# ── Per-market extraction ─────────────────────────────────────────────────────

def _process_market(prices_df: pd.DataFrame, sample_seconds: list[int]) -> dict | None:
    """
    Extract one row per market. Records entry_price at every second in
    sample_seconds so sweep mode can test all times without reloading.
    """
    if "slug" not in prices_df.columns or len(prices_df) < 5:
        return None

    prices_df = prices_df.sort_values("time").reset_index(drop=True)
    slug      = prices_df["slug"].iloc[0]

    # Outcome
    final_bid = prices_df["up_bid"].dropna()
    if len(final_bid) == 0:
        return None
    last = float(final_bid.iloc[-1])
    if last >= SETTLED_THRESH:
        outcome = 1
    elif last <= (1.0 - SETTLED_THRESH):
        outcome = 0
    else:
        return None

    # Opening microprice for BTC spot estimate
    open_micro = float(prices_df["up_microprice"].iloc[0]) if "up_microprice" in prices_df.columns else None

    date_str = None
    if "time" in prices_df.columns:
        ts       = pd.to_datetime(prices_df["time"].iloc[0], utc=True)
        date_str = ts.strftime("%Y-%m-%d")

    row: dict = {
        "slug":       slug,
        "date":       date_str,
        "strike":     _parse_strike(slug),
        "open_micro": open_micro,
        "outcome":    outcome,
    }

    # Sample entry price at each requested second
    for sec in sample_seconds:
        idx = min(sec, len(prices_df) - 1)
        er  = prices_df.iloc[idx]
        ask   = float(er.get("up_ask")        or 0.0)
        micro = float(er.get("up_microprice") or 0.0)
        row[f"px_{sec}"] = ask if ask > 0 else micro

    return row


# ── Zip loading ───────────────────────────────────────────────────────────────

def _load_zip(zip_path: str, sample_seconds: list[int]) -> list[dict]:
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".parquet"))
        print(f"  {zip_path}: {len(names)} markets")
        for name in names:
            try:
                with zf.open(name) as fh:
                    df = pd.read_parquet(io.BytesIO(fh.read()))
            except Exception:
                continue
            row = _process_market(df, sample_seconds)
            if row is not None:
                rows.append(row)
            del df
    gc.collect()
    print(f"    → {len(rows)} settled markets extracted")
    return rows


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ev(win_rate: float, entry_price: float) -> float:
    if entry_price <= 0:
        return float("nan")
    return win_rate * (1.0 / entry_price - 1.0) - (1.0 - win_rate)


def _add_distance(df: pd.DataFrame) -> pd.DataFrame:
    has = df["strike"].notna() & df["open_micro"].notna()
    df_s = df[has].copy()
    if len(df_s) == 0:
        return df

    def _atm(grp):
        idx = (grp["open_micro"] - 0.5).abs().idxmin()
        return float(grp.loc[idx, "strike"])

    spot_by_date = df_s.groupby("date").apply(_atm, include_groups=False)
    df_s["btc_spot"] = df_s["date"].map(spot_by_date)
    df_s["distance"] = (df_s["strike"] - df_s["btc_spot"]).abs()
    df = df.copy()
    df.loc[df_s.index, "btc_spot"] = df_s["btc_spot"]
    df.loc[df_s.index, "distance"] = df_s["distance"]
    return df


# ── Sweep mode ────────────────────────────────────────────────────────────────

def _print_sweep(df: pd.DataFrame, metric: str = "wr") -> None:
    """
    Print two grids side-by-side (or sequentially):
      Grid 1 — Win Rate%   (row=entry_second, col=min_price_threshold)
      Grid 2 — EV per $1   (same layout)
      Grid 3 — N trades    (same layout)

    metric: 'wr' | 'ev' | 'n'  — or 'all' to print all three
    """
    seconds = [s for s in SWEEP_SECONDS if f"px_{s}" in df.columns]
    prices  = SWEEP_PRICES

    # Build result table: rows=seconds, cols=min_price
    wr_rows:  list[list] = []
    ev_rows:  list[list] = []
    n_rows:   list[list] = []

    for sec in seconds:
        px_col = f"px_{sec}"
        wr_row, ev_row, n_row = [], [], []
        for min_px in prices:
            mask = (df[px_col] >= min_px) & (df[px_col] < 1.0)
            sub  = df[mask]
            n    = len(sub)
            if n < MIN_TRADES:
                wr_row.append(None)
                ev_row.append(None)
                n_row.append(n)
            else:
                wr  = float(sub["outcome"].mean())
                avg = float(sub[px_col].mean())
                wr_row.append(wr)
                ev_row.append(_ev(wr, avg))
                n_row.append(n)
        wr_rows.append(wr_row)
        ev_rows.append(ev_row)
        n_rows.append(n_row)

    col_w = 14

    def _header(title: str) -> None:
        print(f"\n{'═' * (10 + col_w * len(prices))}")
        print(f"  {title}")
        print(f"{'─' * (10 + col_w * len(prices))}")
        print(f"  {'t(s)':>6}", end="")
        for p in prices:
            print(f"  {f'≥{p:.3f}':>{col_w-2}}", end="")
        print()
        print("  " + "─" * (6 + col_w * len(prices)))

    def _row_label(sec: int) -> str:
        return f"t={sec:>4}"

    # ── Grid 1: Win Rate ──────────────────────────────────────────────────────
    _header("Win Rate  (% of trades that WIN)")
    for i, sec in enumerate(seconds):
        print(f"  {_row_label(sec):>6}", end="")
        for j, val in enumerate(wr_rows[i]):
            n = n_rows[i][j]
            if val is None:
                print(f"  {'—':>{col_w-2}}", end="")
            else:
                cell = f"{val:.1%} n={n}"
                print(f"  {cell:>{col_w-2}}", end="")
        print()
    print(f"{'═' * (10 + col_w * len(prices))}")

    # ── Grid 2: EV per $1 wagered ─────────────────────────────────────────────
    _header("EV per $1 wagered  (+ = edge in your favour)")
    for i, sec in enumerate(seconds):
        print(f"  {_row_label(sec):>6}", end="")
        for j, val in enumerate(ev_rows[i]):
            n = n_rows[i][j]
            if val is None:
                print(f"  {'—':>{col_w-2}}", end="")
            else:
                star = " ★" if val > 0 else "  "
                cell = f"{val:+.4f}{star}"
                print(f"  {cell:>{col_w-2}}", end="")
        print()
    print(f"{'═' * (10 + col_w * len(prices))}")

    # ── Grid 3: Trade count ───────────────────────────────────────────────────
    _header("N Trades  (how many historical trades match the filter)")
    for i, sec in enumerate(seconds):
        print(f"  {_row_label(sec):>6}", end="")
        for j, n in enumerate(n_rows[i]):
            print(f"  {str(n) if n is not None else '—':>{col_w-2}}", end="")
        print()
    print(f"{'═' * (10 + col_w * len(prices))}")

    # ── Best cell summary ─────────────────────────────────────────────────────
    best_ev, best_sec, best_px, best_wr, best_n = -999, None, None, None, None
    for i, sec in enumerate(seconds):
        for j, ev in enumerate(ev_rows[i]):
            if ev is not None and ev > best_ev:
                best_ev  = ev
                best_sec = sec
                best_px  = prices[j]
                best_wr  = wr_rows[i][j]
                best_n   = n_rows[i][j]

    if best_sec is not None:
        print(f"\n  ★  Best EV cell:  t={best_sec}s  price≥{best_px:.3f}  "
              f"→  win={best_wr:.1%}  EV={best_ev:+.4f}  n={best_n}")


# ── Single-time detailed analysis ────────────────────────────────────────────

def _print_price_table(df: pd.DataFrame, entry_second: int) -> None:
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.875,
            0.90, 0.915, 0.93, 0.945, 0.96, 0.975, 1.01]
    px_col = f"px_{entry_second}"
    df = df[df[px_col] > 0].copy()
    df["_b"] = pd.cut(df[px_col], bins=bins)
    g   = df.groupby("_b", observed=True)
    wr  = g["outcome"].mean()
    cnt = g["outcome"].count()
    ep  = g[px_col].mean()

    print(f"\n{'═' * 72}")
    print(f"  Win Rate by Entry Price  (entry at t={entry_second}s)")
    print(f"{'─' * 72}")
    print(f"  {'Bucket':>22}  {'N':>6}  {'Win%':>7}  {'Edge':>7}  {'EV/$1':>7}")
    print(f"{'─' * 72}")
    for b in wr.index:
        n = int(cnt[b])
        if n < MIN_TRADES:
            continue
        w   = wr[b]
        ev  = _ev(w, float(ep[b]))
        print(f"  {str(b):>22}  {n:>6}  {w:>6.1%}  {w-b.mid:>+6.3f}  {ev:>+6.4f}")
    print(f"{'═' * 72}")


def _print_distance_table(df: pd.DataFrame, entry_second: int) -> None:
    px_col = f"px_{entry_second}"
    bins   = [0, 50, 100, 150, 200, 250, 300, 400, 500, 1000, 5001]
    valid  = df["distance"].notna() & (df["distance"] < 5000) & (df[px_col] > 0)
    dg     = df[valid].copy()
    if len(dg) < MIN_TRADES:
        return
    dg["_b"] = pd.cut(dg["distance"], bins=bins)
    g   = dg.groupby("_b", observed=True)
    wr  = g["outcome"].mean()
    cnt = g["outcome"].count()
    ep  = g[px_col].mean()

    print(f"\n{'═' * 76}")
    print("  Win Rate by |BTC_spot − Strike|  (distance at market open)")
    print(f"{'─' * 76}")
    print(f"  {'Bucket ($)':>22}  {'N':>6}  {'Win%':>7}  {'AvgPx':>7}  {'Edge':>7}  {'EV/$1':>7}")
    print(f"{'─' * 76}")
    for b in wr.index:
        n = int(cnt[b])
        if n < MIN_TRADES:
            continue
        w   = wr[b]
        avg = float(ep[b])
        print(f"  {str(b):>22}  {n:>6}  {w:>6.1%}  {avg:>6.3f}  {w-avg:>+6.3f}  {_ev(w,avg):>+6.4f}")
    print(f"{'═' * 76}")


def _print_grid_2d(df: pd.DataFrame, entry_second: int) -> None:
    px_col      = f"px_{entry_second}"
    price_bins  = [0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 1.01]
    dist_bins   = [0, 50, 100, 150, 200, 250, 300, 400, 500, 5001]
    valid = df["distance"].notna() & (df["distance"] < 5000) & (df[px_col] > 0.70)
    dg    = df[valid].copy()
    if len(dg) < 10:
        return

    dg["pb"] = pd.cut(dg[px_col],       bins=price_bins)
    dg["db"] = pd.cut(dg["distance"],   bins=dist_bins)
    piv_wr  = dg.pivot_table(index="db", columns="pb", values="outcome", aggfunc="mean",  observed=True)
    piv_cnt = dg.pivot_table(index="db", columns="pb", values="outcome", aggfunc="count", observed=True)

    ref_mask = (dg["distance"] >= 150) & (dg[px_col] >= 0.90)
    ref_wr   = float(dg[ref_mask]["outcome"].mean()) if len(dg[ref_mask]) >= MIN_TRADES else None

    print(f"\n{'═' * 95}")
    print("  2D Grid: Win Rate  (row=|BTC-strike|  col=entry price)")
    if ref_wr:
        print(f"  ★ = within ±3% of reference win rate ({ref_wr:.1%})")
    print(f"{'─' * 95}")
    cols = piv_wr.columns
    print(f"  {'Distance':>16}", end="")
    for c in cols:
        print(f"  {str(c):>18}", end="")
    print()
    for rb in piv_wr.index:
        print(f"  {str(rb):>16}", end="")
        for c in cols:
            wr  = piv_wr.loc[rb, c]
            cnt = piv_cnt.loc[rb, c] if not pd.isna(piv_cnt.loc[rb, c]) else 0
            if pd.isna(wr) or cnt < MIN_TRADES:
                print(f"  {'— (n<10)':>18}", end="")
            else:
                star = " ★" if ref_wr and abs(wr - ref_wr) <= 0.03 else "  "
                print(f"  {f'{wr:.1%} n={int(cnt)}{star}':>18}", end="")
        print()
    print(f"{'═' * 95}")


# ── Main ──────────────────────────────────────────────────────────────────────

def backtest(
    prices_zips:  list[str],
    entry_second: int,
    sweep:        bool,
    output_path:  str | None,
) -> None:
    sample_seconds = SWEEP_SECONDS if sweep else [entry_second]

    all_rows: list[dict] = []
    for zp in prices_zips:
        all_rows.extend(_load_zip(zp, sample_seconds))

    df = pd.DataFrame(all_rows)
    df["distance"] = np.nan
    df["btc_spot"] = np.nan

    n_up   = int((df["outcome"] == 1).sum())
    n_down = int((df["outcome"] == 0).sum())
    print(f"\nTotal settled markets: {len(df)}  "
          f"(Up={n_up}  Down={n_down}  base_rate={n_up/len(df):.1%})")

    if sweep:
        print(f"\nRunning sweep across {len(SWEEP_SECONDS)} entry times × "
              f"{len(SWEEP_PRICES)} price thresholds...")
        _print_sweep(df)
    else:
        print(f"\nEntry point: t={entry_second}s  "
              f"({entry_second/60:.1f} min in, {(900-entry_second)/60:.1f} min left)")
        _print_price_table(df, entry_second)

        n_strike = int(df["strike"].notna().sum())
        print(f"\n  Parseable strikes: {n_strike}/{len(df)}")
        unparsed = df[df["strike"].isna()]["slug"].dropna().head(5).tolist()
        if unparsed:
            print("  Sample unparsed slugs:")
            for s in unparsed:
                print(f"    {s}")

        if n_strike >= 50:
            df = _add_distance(df)
            if df["distance"].notna().sum() >= 50:
                _print_distance_table(df, entry_second)
                ref_mask = (df["distance"] >= 150) & (df[f"px_{entry_second}"] >= 0.90)
                ref_df   = df[ref_mask]
                ref_wr   = float(ref_df["outcome"].mean()) if len(ref_df) >= MIN_TRADES else None
                print(f"\n  ── Reference (dist≥$150 AND price≥$0.90) ──")
                if ref_wr:
                    print(f"     N={len(ref_df)}  win={ref_wr:.1%}  "
                          f"avg_px={ref_df[f'px_{entry_second}'].mean():.3f}  "
                          f"EV={_ev(ref_wr, ref_df[f'px_{entry_second}'].mean()):+.4f}")
                else:
                    print(f"     N={len(ref_df)}  (need ≥{MIN_TRADES})")
                _print_grid_2d(df, entry_second)

    if output_path:
        save_cols = (["slug", "date", "strike", "btc_spot", "distance", "open_micro", "outcome"]
                     + [f"px_{s}" for s in sample_seconds])
        df[[c for c in save_cols if c in df.columns]].to_csv(output_path, index=False)
        print(f"\nRaw rows saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest BTC 15m binary strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full time × price sweep (recommended first run)
  python ml/backtest.py --prices "E:\\...zip" "E:\\...zip" --sweep

  # Detailed single-time analysis
  python ml/backtest.py --prices "E:\\...zip" --entry-second 750 --output bt.csv
""",
    )
    parser.add_argument("--prices", required=True, nargs="+")
    parser.add_argument("--entry-second", type=int, default=750,
                        help="Seconds into market for single-time mode (default 750)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all entry times × price thresholds in one run")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    backtest(args.prices, args.entry_second, args.sweep, args.output)
