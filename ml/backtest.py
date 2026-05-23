"""
ml/backtest.py — Strategy backtester for BTC 15m binary contracts.

Loads prices parquet zips and computes actual win rates across all
(entry_price, distance_from_strike) combinations, letting you find
entry points that have equivalent historical probability to your baseline
strategy ("buy when dist > 150 AND contract >= $0.90").

Key parameter: --entry-second controls when in the 15-min market we sample
the entry price. Use a late value (e.g. 750 = 12.5 min in) to see the
high-confidence regime where contracts trade at $0.90+.

Usage (Windows):
    # Sample entry at 12.5 minutes in (recommended for 0.90+ strategy)
    python ml/backtest.py ^
        --prices "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip" ^
                 "E:\\prices_btc_15m_2026-04-28_2026-05-05.zip" ^
                 "E:\\prices_btc_15m_2026-05-06_2026-05-12.zip" ^
                 "E:\\prices_btc_15m_2026-05-13_2026-05-18.zip" ^
        --entry-second 750 ^
        --output backtest.csv

    # Sweep multiple entry times to see how win rates evolve
    for %t in (60 300 600 720 750 800) do ^
        python ml/backtest.py --prices ... --entry-second %t

Output:
    - Win-rate table by entry price
    - Win-rate table by distance from strike (if slugs contain parseable prices)
    - 2D grid: price x distance with ★ for iso-probability cells
    - EV per $1 for each cell
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

SETTLED_THRESH = 0.90   # up_bid >= this at end → WIN (contract settles $1)
MIN_TRADES     = 5      # hide cells with fewer than this many trades


# ── Strike parsing ────────────────────────────────────────────────────────────

_STRIKE_PATTERNS = [
    r'above[_-]?(\d{4,6})',             # above-95000, above95000
    r'below[_-]?(\d{4,6})',             # below-95000
    r'[tTbB](\d{4,6})(?:[^0-9]|$)',     # T95000, B94850 (Kalshi-style)
    r'[-_](\d{5,6})[-_uU]',             # -95000-
    r'(\d{5,6})(?:usd|USD)?(?:[-_k]|$)',# 95000usd, 95000k
    r'btc.*?(\d{4,6})',                 # btc...95000
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

def _process_market(prices_df: pd.DataFrame, entry_second: int) -> dict | None:
    if "slug" not in prices_df.columns or len(prices_df) < 5:
        return None

    prices_df = prices_df.sort_values("time").reset_index(drop=True)
    slug      = prices_df["slug"].iloc[0]

    # Entry: tick at entry_second seconds in (or last available tick)
    entry_idx   = min(entry_second, len(prices_df) - 1)
    entry_row   = prices_df.iloc[entry_idx]

    entry_ask   = float(entry_row.get("up_ask")        or 0.0)
    entry_micro = float(entry_row.get("up_microprice") or 0.0)
    entry_price = entry_ask if entry_ask > 0 else entry_micro
    if entry_price <= 0 or entry_price >= 1.0:
        return None

    # Opening microprice (t=0) — used for BTC spot estimation via ATM method
    open_micro = float(prices_df["up_microprice"].iloc[0]) if "up_microprice" in prices_df.columns else None

    # Outcome: final up_bid determines settlement
    final_bid = prices_df["up_bid"].dropna()
    if len(final_bid) == 0:
        return None
    last_bid = float(final_bid.iloc[-1])
    if last_bid >= SETTLED_THRESH:
        outcome = 1
    elif last_bid <= (1.0 - SETTLED_THRESH):
        outcome = 0
    else:
        return None   # ambiguous — skip

    date_str = None
    if "time" in prices_df.columns:
        ts       = pd.to_datetime(prices_df["time"].iloc[0], utc=True)
        date_str = ts.strftime("%Y-%m-%d")

    return {
        "slug":        slug,
        "date":        date_str,
        "strike":      _parse_strike(slug),
        "entry_price": entry_price,
        "open_micro":  open_micro,
        "outcome":     outcome,
    }


# ── Zip loading ───────────────────────────────────────────────────────────────

def _load_zip(zip_path: str, entry_second: int) -> list[dict]:
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
            row = _process_market(df, entry_second)
            if row is not None:
                rows.append(row)
            del df
    gc.collect()
    print(f"    → {len(rows)} settled markets extracted")
    return rows


# ── Display helpers ───────────────────────────────────────────────────────────

def _ev(win_rate: float, entry_price: float) -> float:
    if entry_price <= 0:
        return float("nan")
    return win_rate * (1.0 / entry_price - 1.0) - (1.0 - win_rate)


def _print_price_table(df: pd.DataFrame) -> None:
    """Win rate bucketed by entry price. Edge = actual win% vs price-implied %."""
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.875,
            0.90, 0.915, 0.93, 0.945, 0.96, 0.975, 1.01]
    df = df.copy()
    df["_b"] = pd.cut(df["entry_price"], bins=bins)
    g   = df.groupby("_b", observed=True)
    wr  = g["outcome"].mean()
    cnt = g["outcome"].count()
    ep  = g["entry_price"].mean()

    print(f"\n{'═' * 72}")
    print(f"  Win Rate by Entry Price  (entry at t={df.attrs.get('entry_second','?')}s)")
    print(f"{'─' * 72}")
    print(f"  {'Bucket':>22}  {'N':>6}  {'Win%':>7}  {'Edge':>7}  {'EV/$1':>7}")
    print(f"{'─' * 72}")
    for b in wr.index:
        n = int(cnt[b])
        if n < MIN_TRADES:
            continue
        w    = wr[b]
        imp  = b.mid           # price bucket mid = market-implied probability
        edge = w - imp
        ev   = _ev(w, float(ep[b]))
        print(f"  {str(b):>22}  {n:>6}  {w:>6.1%}  {edge:>+6.3f}  {ev:>+6.4f}")
    print(f"{'═' * 72}")


def _print_distance_table(df: pd.DataFrame) -> None:
    """Win rate bucketed by |BTC_spot - strike|. Edge = actual win% vs avg entry price."""
    bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, 1000, 5001]
    df = df.copy()
    df["_b"] = pd.cut(df["distance"], bins=bins)
    g   = df.groupby("_b", observed=True)
    wr  = g["outcome"].mean()
    cnt = g["outcome"].count()
    ep  = g["entry_price"].mean()   # average price paid in this distance bucket

    print(f"\n{'═' * 72}")
    print("  Win Rate by |BTC_spot − Strike| at Entry  ($)")
    print("  Edge = actual win% minus avg entry price (market-implied %)")
    print(f"{'─' * 72}")
    print(f"  {'Bucket ($)':>22}  {'N':>6}  {'Win%':>7}  {'AvgPx':>7}  {'Edge':>7}  {'EV/$1':>7}")
    print(f"{'─' * 72}")
    for b in wr.index:
        n = int(cnt[b])
        if n < MIN_TRADES:
            continue
        w    = wr[b]
        avg  = float(ep[b])
        edge = w - avg         # actual win rate vs market-implied probability
        ev   = _ev(w, avg)
        print(f"  {str(b):>22}  {n:>6}  {w:>6.1%}  {avg:>6.3f}  {edge:>+6.3f}  {ev:>+6.4f}")
    print(f"{'═' * 72}")


# ── Distance estimation ───────────────────────────────────────────────────────

def _add_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate BTC spot per date from the ATM market (open_micro closest to 0.50),
    then compute |strike - spot| for every market that day.
    """
    has = df["strike"].notna() & df["open_micro"].notna()
    df_s = df[has].copy()
    if len(df_s) == 0:
        return df

    def _atm(grp: pd.DataFrame) -> float | None:
        idx = (grp["open_micro"] - 0.5).abs().idxmin()
        return float(grp.loc[idx, "strike"])

    spot_by_date = df_s.groupby("date").apply(_atm, include_groups=False)
    df_s["btc_spot"] = df_s["date"].map(spot_by_date)
    df_s["distance"] = (df_s["strike"] - df_s["btc_spot"]).abs()

    df = df.copy()
    df.loc[df_s.index, "btc_spot"] = df_s["btc_spot"]
    df.loc[df_s.index, "distance"] = df_s["distance"]
    return df


# ── 2D grid ───────────────────────────────────────────────────────────────────

def _print_grid(df: pd.DataFrame, ref_wr: float | None) -> None:
    price_bins = [0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 1.01]
    dist_bins  = [0, 50, 100, 150, 200, 250, 300, 400, 500, 5001]

    valid = df["distance"].notna() & (df["distance"] < 5000) & (df["entry_price"] > 0.70)
    dg    = df[valid].copy()
    if len(dg) < 10:
        print("  (Not enough data for 2D grid)")
        return

    dg["pb"] = pd.cut(dg["entry_price"], bins=price_bins)
    dg["db"] = pd.cut(dg["distance"],    bins=dist_bins)

    piv_wr  = dg.pivot_table(index="db", columns="pb", values="outcome",
                              aggfunc="mean",  observed=True)
    piv_cnt = dg.pivot_table(index="db", columns="pb", values="outcome",
                              aggfunc="count", observed=True)
    tol = 0.03

    print(f"\n{'═' * 95}")
    print("  2D Win-Rate Grid  (row=|BTC-strike|  col=entry price)")
    if ref_wr is not None:
        print(f"  ★ = within ±{tol:.0%} of reference win rate ({ref_wr:.1%})")
    print(f"{'─' * 95}")
    cols = piv_wr.columns
    print(f"  {'Distance':>18}", end="")
    for c in cols:
        print(f"  {str(c):>18}", end="")
    print()
    print("  " + "─" * (18 + 20 * len(cols)))
    for rb in piv_wr.index:
        print(f"  {str(rb):>18}", end="")
        for c in cols:
            wr  = piv_wr.loc[rb, c]
            cnt = piv_cnt.loc[rb, c] if not pd.isna(piv_cnt.loc[rb, c]) else 0
            if pd.isna(wr) or cnt < MIN_TRADES:
                print(f"  {'— (n<5)':>18}", end="")
            else:
                star = " ★" if ref_wr is not None and abs(wr - ref_wr) <= tol else "  "
                print(f"  {f'{wr:.1%} n={int(cnt)}{star}':>18}", end="")
        print()
    print(f"{'═' * 95}")


# ── Main ──────────────────────────────────────────────────────────────────────

def backtest(prices_zips: list[str], entry_second: int, output_path: str | None) -> None:
    print(f"\nEntry point: t={entry_second}s into 15-min market "
          f"({entry_second/60:.1f} min in, {(900-entry_second)/60:.1f} min left)")

    all_rows: list[dict] = []
    for zp in prices_zips:
        all_rows.extend(_load_zip(zp, entry_second))

    df = pd.DataFrame(all_rows)
    df["distance"] = np.nan
    df["btc_spot"] = np.nan
    df.attrs["entry_second"] = entry_second

    n_up   = int((df["outcome"] == 1).sum())
    n_down = int((df["outcome"] == 0).sum())
    print(f"\nTotal settled markets: {len(df)}  (Up={n_up}  Down={n_down}  "
          f"base_rate={n_up/len(df):.1%})")

    # ── 1. Win rate by entry price ────────────────────────────────────────────
    _print_price_table(df)

    # ── 2. Distance analysis ──────────────────────────────────────────────────
    n_strike = int(df["strike"].notna().sum())
    print(f"\n  Slugs with parseable strike: {n_strike} / {len(df)}")

    # Show 10 unparsed slugs so we can fix the regex if needed
    unparsed = df[df["strike"].isna()]["slug"].dropna().head(10).tolist()
    if unparsed:
        print("  Sample unparsed slugs (to debug strike regex):")
        for s in unparsed:
            print(f"    {s}")

    if n_strike >= 50:
        df = _add_distance(df)
        n_dist = int(df["distance"].notna().sum())
        print(f"  Markets with computed distance: {n_dist}")

        if n_dist >= 50:
            _print_distance_table(df[df["distance"].notna() & (df["distance"] < 5000)])

            # Reference strategy
            ref_mask = (df["distance"] >= 150) & (df["entry_price"] >= 0.90)
            ref_df   = df[ref_mask]
            ref_wr   = float(ref_df["outcome"].mean()) if len(ref_df) >= MIN_TRADES else None

            print(f"\n  ── Reference (dist ≥ $150  AND  price ≥ $0.90) ──")
            if ref_wr is not None:
                ref_ev = _ev(ref_wr, float(ref_df["entry_price"].mean()))
                print(f"     N={len(ref_df)}  win={ref_wr:.1%}  "
                      f"avg_entry={ref_df['entry_price'].mean():.3f}  EV={ref_ev:+.4f}")
            else:
                print(f"     N={len(ref_df)}  (need ≥{MIN_TRADES} trades for reference)")

            _print_grid(df, ref_wr)

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path:
        save_cols = ["slug", "date", "strike", "btc_spot", "distance",
                     "entry_price", "open_micro", "outcome"]
        df[[c for c in save_cols if c in df.columns]].to_csv(output_path, index=False)
        print(f"\nRaw rows saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest BTC 15m binary strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Entry-second guide:
  60   = 1 min in  (market open, most contracts near 50%)
  300  = 5 min in
  600  = 10 min in
  720  = 12 min in
  750  = 12.5 min in  ← recommended for 0.90+ strategy
  800  = 13.3 min in
  840  = 14 min in    (very late, fewer data points)

Examples:
  python ml/backtest.py --prices "E:\\...zip" --entry-second 750
  python ml/backtest.py --prices "E:\\...zip" "E:\\...zip" --entry-second 750 --output bt.csv
""",
    )
    parser.add_argument("--prices", required=True, nargs="+",
                        help="One or more prices zip paths")
    parser.add_argument("--entry-second", type=int, default=750,
                        help="Seconds into market to sample entry price (default 750 = 12.5 min)")
    parser.add_argument("--output", default=None,
                        help="Optional CSV path for raw per-market data")
    args = parser.parse_args()
    backtest(args.prices, args.entry_second, args.output)
