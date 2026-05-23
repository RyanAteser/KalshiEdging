"""
ml/backtest.py — Backtester for Kalshi KXBTC15M 15-minute BTC binary contracts.

Sweeps every combination of entry time × contract price × implied BTC distance
from the strike to find the optimal entry conditions.

Implied distance converts contract price → dollars BTC is from the strike using:
    dist = norm.ppf(price) × BTC_vol × √(time_remaining) × strike

Usage (Windows):
    python ml/backtest.py ^
        --prices "E:\\prices_btc_15m_2026-04-20_2026-04-27.zip" ^
                 "E:\\prices_btc_15m_2026-04-28_2026-05-05.zip" ^
                 "E:\\prices_btc_15m_2026-05-06_2026-05-12.zip" ^
                 "E:\\prices_btc_15m_2026-05-13_2026-05-18.zip" ^
        --sweep

    # Override assumed BTC annual volatility (default 65%)
    python ml/backtest.py --prices ... --sweep --btc-vol 0.70
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

try:
    from scipy.stats import norm as _norm
    def _ppf(p: float | np.ndarray) -> float | np.ndarray:
        return _norm.ppf(p)
except ImportError:
    def _ppf(p: float | np.ndarray) -> float | np.ndarray:
        """Rational approximation of the normal quantile (Acklam 2003)."""
        p = np.asarray(p, dtype=float)
        a = np.array([-3.969683028665376e+01, 2.209460984245205e+02,
                      -2.759285104469687e+02, 1.383577518672690e+02,
                      -3.066479806614716e+01, 2.506628277459239e+00])
        b = np.array([-5.447609879822406e+01, 1.615858368580409e+02,
                      -1.556989798598866e+02, 6.680131188771972e+01,
                      -1.328068155288572e+01])
        c = np.array([-7.784894002430293e-03, -3.223964580411365e-01,
                      -2.400758277161838e+00, -2.549732539343734e+00,
                       4.374664141464968e+00,  2.938163982698783e+00])
        d = np.array([7.784695709041462e-03, 3.224671290700398e-01,
                      2.445134137142996e+00, 3.754408661907416e+00])
        lo, hi = 0.02425, 1 - 0.02425
        out = np.zeros_like(p)
        # Central region
        m = (p >= lo) & (p <= hi)
        q = p[m] - 0.5; r = q * q
        out[m] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                 (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        # Lower tail
        lo_m = p < lo
        q = np.sqrt(-2*np.log(p[lo_m]))
        out[lo_m] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        # Upper tail
        hi_m = p > hi
        q = np.sqrt(-2*np.log(1-p[hi_m]))
        out[hi_m] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                     ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return float(out) if out.ndim == 0 else out

# ── Constants ─────────────────────────────────────────────────────────────────

SETTLED_THRESH   = 0.90
MARKET_DURATION  = 900          # 15-min market = 900 seconds
MIN_TRADES       = 10
SECS_PER_YEAR    = 365.25 * 24 * 3600

SWEEP_SECONDS    = [0, 60, 180, 300, 420, 540, 600, 660, 720, 750, 810, 870]
SWEEP_PRICES     = [0.80, 0.85, 0.875, 0.90, 0.915, 0.93, 0.945, 0.96]
SWEEP_DISTANCES  = [50, 100, 150, 200, 250, 300, 400, 500]   # implied $ from strike


# ── Implied distance ──────────────────────────────────────────────────────────

def implied_dist(px: float | np.ndarray,
                 entry_second: int,
                 strike: float | np.ndarray,
                 btc_vol: float) -> float | np.ndarray:
    """
    Convert contract price → implied $ distance of BTC from strike.

    Uses the Bachelier (normal) binary option formula:
        dist = norm.ppf(price) × btc_vol × √(remaining_years) × strike

    Positive = BTC above strike (YES in the money).
    Negative = BTC below strike (YES out of the money).
    """
    remaining_sec   = max(MARKET_DURATION - entry_second, 1)
    remaining_years = remaining_sec / SECS_PER_YEAR
    sigma_t = btc_vol * np.sqrt(remaining_years)
    px_clipped = np.clip(px, 0.001, 0.999)
    return _ppf(px_clipped) * sigma_t * strike


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
    if "slug" not in prices_df.columns or len(prices_df) < 5:
        return None

    prices_df = prices_df.sort_values("time").reset_index(drop=True)
    slug      = prices_df["slug"].iloc[0]

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

    date_str = None
    if "time" in prices_df.columns:
        ts       = pd.to_datetime(prices_df["time"].iloc[0], utc=True)
        date_str = ts.strftime("%Y-%m-%d")

    row: dict = {
        "slug":       slug,
        "date":       date_str,
        "strike":     _parse_strike(slug),
        "outcome":    outcome,
    }

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


# ── EV helper ─────────────────────────────────────────────────────────────────

def _ev(win_rate: float, avg_price: float) -> float:
    if avg_price <= 0:
        return float("nan")
    return win_rate * (1.0 / avg_price - 1.0) - (1.0 - win_rate)


# ── Reference table: price → implied distance ─────────────────────────────────

def _print_price_dist_ref(btc_vol: float, strike: float = 95_000) -> None:
    """Show how contract price maps to implied BTC distance at each entry time."""
    print(f"\n{'═'*90}")
    print(f"  Price → Implied BTC Distance from Strike  "
          f"(BTC=${strike:,.0f}  vol={btc_vol:.0%} annual)")
    print(f"  Tells you: 'at this price, BTC is approximately $X from the strike'")
    print(f"{'─'*90}")

    col_w = 9
    print(f"  {'Entry':>8}", end="")
    for px in SWEEP_PRICES:
        print(f"  {f'px={px:.3f}':>{col_w}}", end="")
    print()
    print("  " + "─" * (8 + (col_w + 2) * len(SWEEP_PRICES)))

    for sec in SWEEP_SECONDS:
        mins_left = (MARKET_DURATION - sec) / 60
        print(f"  t={sec:>4}s", end="")
        for px in SWEEP_PRICES:
            d = implied_dist(px, sec, strike, btc_vol)
            print(f"  {f'${d:,.0f}':>{col_w}}", end="")
        print(f"  ({mins_left:.1f}m left)")
    print(f"{'═'*90}")


# ── Price-based sweep ─────────────────────────────────────────────────────────

def _print_price_sweep(df: pd.DataFrame) -> None:
    seconds = [s for s in SWEEP_SECONDS if f"px_{s}" in df.columns]
    prices  = SWEEP_PRICES
    col_w   = 14

    wr_rows, ev_rows, n_rows = [], [], []
    for sec in seconds:
        px_col = f"px_{sec}"
        wr_row, ev_row, n_row = [], [], []
        for min_px in prices:
            sub = df[(df[px_col] >= min_px) & (df[px_col] < 1.0)]
            n   = len(sub)
            if n < MIN_TRADES:
                wr_row.append(None); ev_row.append(None); n_row.append(n)
            else:
                wr  = float(sub["outcome"].mean())
                avg = float(sub[px_col].mean())
                wr_row.append(wr); ev_row.append(_ev(wr, avg)); n_row.append(n)
        wr_rows.append(wr_row); ev_rows.append(ev_row); n_rows.append(n_row)

    def _hdr(title):
        print(f"\n{'═'*(10 + col_w*len(prices))}")
        print(f"  {title}")
        print(f"{'─'*(10 + col_w*len(prices))}")
        print(f"  {'t(s)':>6}", end="")
        for p in prices:
            print(f"  {f'≥{p:.3f}':>{col_w-2}}", end="")
        print()
        print("  " + "─" * (6 + col_w * len(prices)))

    _hdr("Win Rate by Entry Time × Min Contract Price")
    for i, sec in enumerate(seconds):
        print(f"  t={sec:>4}", end="")
        for j, v in enumerate(wr_rows[i]):
            n = n_rows[i][j]
            print(f"  {f'{v:.1%} n={n}' if v is not None else '—':>{col_w-2}}", end="")
        print()
    print(f"{'═'*(10 + col_w*len(prices))}")

    _hdr("EV per $1 wagered  (★ = positive edge)")
    for i, sec in enumerate(seconds):
        print(f"  t={sec:>4}", end="")
        for j, v in enumerate(ev_rows[i]):
            if v is None:
                print(f"  {'—':>{col_w-2}}", end="")
            else:
                star = " ★" if v > 0 else "  "
                print(f"  {f'{v:+.4f}{star}':>{col_w-2}}", end="")
        print()
    print(f"{'═'*(10 + col_w*len(prices))}")

    # Best cell
    best = max(
        ((ev_rows[i][j], seconds[i], prices[j], wr_rows[i][j], n_rows[i][j])
         for i in range(len(seconds)) for j in range(len(prices))
         if ev_rows[i][j] is not None),
        key=lambda x: x[0], default=None
    )
    if best:
        ev, sec, px, wr, n = best
        print(f"\n  ★  Best price-based cell: t={sec}s  price≥{px}  "
              f"→  win={wr:.1%}  EV={ev:+.4f}  n={n}")


# ── Distance-based sweep ──────────────────────────────────────────────────────

def _print_distance_sweep(df: pd.DataFrame, btc_vol: float) -> None:
    """
    Sweep by minimum implied distance ($ from strike) instead of minimum price.
    More directly comparable to Kalshi live strategy filters.
    """
    seconds   = [s for s in SWEEP_SECONDS if f"px_{s}" in df.columns]
    distances = SWEEP_DISTANCES
    col_w     = 16

    has_strike = df["strike"].notna()
    if has_strike.sum() < 50:
        print("\n  (Not enough parseable strikes for distance sweep)")
        return

    # Precompute implied distance for each (market, second)
    for sec in seconds:
        px_col   = f"px_{sec}"
        dist_col = f"idist_{sec}"
        if dist_col not in df.columns:
            px      = df[px_col].where(has_strike).clip(0.001, 0.999)
            strike  = df["strike"].fillna(95_000)
            df[dist_col] = implied_dist(px.values, sec, strike.values, btc_vol)

    # Build grids
    wr_rows, ev_rows, n_rows = [], [], []
    for sec in seconds:
        px_col   = f"px_{sec}"
        dist_col = f"idist_{sec}"
        wr_row, ev_row, n_row = [], [], []
        for min_d in distances:
            sub = df[(df[dist_col] >= min_d) & (df[px_col] < 1.0)]
            n   = len(sub)
            if n < MIN_TRADES:
                wr_row.append(None); ev_row.append(None); n_row.append(n)
            else:
                wr  = float(sub["outcome"].mean())
                avg = float(sub[px_col].mean())
                wr_row.append(wr); ev_row.append(_ev(wr, avg)); n_row.append(n)
        wr_rows.append(wr_row); ev_rows.append(ev_row); n_rows.append(n_row)

    def _hdr(title):
        print(f"\n{'═'*(10 + col_w*len(distances))}")
        print(f"  {title}")
        print(f"{'─'*(10 + col_w*len(distances))}")
        print(f"  {'t(s)':>6}", end="")
        for d in distances:
            print(f"  {f'≥${d}':>{col_w-2}}", end="")
        print()
        print("  " + "─" * (6 + col_w * len(distances)))

    _hdr(f"Win Rate by Entry Time × Min Implied Distance  (BTC vol={btc_vol:.0%})")
    for i, sec in enumerate(seconds):
        mins_left = (MARKET_DURATION - sec) / 60
        print(f"  t={sec:>4}", end="")
        for j, v in enumerate(wr_rows[i]):
            n = n_rows[i][j]
            print(f"  {f'{v:.1%} n={n}' if v is not None else '—':>{col_w-2}}", end="")
        print(f"  ({mins_left:.1f}m left)")
    print(f"{'═'*(10 + col_w*len(distances))}")

    _hdr("EV per $1 wagered  (★ = positive edge)")
    for i, sec in enumerate(seconds):
        print(f"  t={sec:>4}", end="")
        for j, v in enumerate(ev_rows[i]):
            if v is None:
                print(f"  {'—':>{col_w-2}}", end="")
            else:
                star = " ★" if v > 0 else "  "
                print(f"  {f'{v:+.4f}{star}':>{col_w-2}}", end="")
        print()
    print(f"{'═'*(10 + col_w*len(distances))}")

    _hdr("Avg contract price at that distance threshold")
    for i, sec in enumerate(seconds):
        px_col   = f"px_{sec}"
        dist_col = f"idist_{sec}"
        print(f"  t={sec:>4}", end="")
        for min_d in distances:
            sub = df[(df[dist_col] >= min_d) & (df[px_col] < 1.0)]
            if len(sub) < MIN_TRADES:
                print(f"  {'—':>{col_w-2}}", end="")
            else:
                avg = float(sub[px_col].mean())
                print(f"  {f'${avg:.3f}':>{col_w-2}}", end="")
        print()
    print(f"{'═'*(10 + col_w*len(distances))}")

    # Best cell
    best = max(
        ((ev_rows[i][j], seconds[i], distances[j], wr_rows[i][j], n_rows[i][j])
         for i in range(len(seconds)) for j in range(len(distances))
         if ev_rows[i][j] is not None),
        key=lambda x: x[0], default=None
    )
    if best:
        ev, sec, dist, wr, n = best
        px_col   = f"px_{sec}"
        dist_col = f"idist_{sec}"
        sub_avg  = df[(df[dist_col] >= dist) & (df[px_col] < 1.0)][px_col].mean()
        print(f"\n  ★  Best distance-based cell: t={sec}s  dist≥${dist}  "
              f"→  win={wr:.1%}  EV={ev:+.4f}  avg_price=${sub_avg:.3f}  n={n}")


# ── 2D grid: price × distance at one entry time ───────────────────────────────

def _print_price_x_dist_grid(df: pd.DataFrame, entry_second: int,
                              btc_vol: float) -> None:
    """
    For a single entry time, show win rate across all (min_price, min_distance) combos.
    """
    px_col   = f"px_{entry_second}"
    dist_col = f"idist_{entry_second}"

    if px_col not in df.columns:
        return

    if dist_col not in df.columns:
        has_strike = df["strike"].notna()
        px     = df[px_col].where(has_strike).clip(0.001, 0.999)
        strike = df["strike"].fillna(95_000)
        df[dist_col] = implied_dist(px.values, entry_second, strike.values, btc_vol)

    mins_left = (MARKET_DURATION - entry_second) / 60
    print(f"\n{'═'*85}")
    print(f"  Win% / EV at t={entry_second}s ({mins_left:.1f} min left)")
    print(f"  row = min implied distance  |  col = min contract price")
    print(f"{'─'*85}")

    col_w = 16
    hdr_label = "dist\\price"
    print(f"  {hdr_label:>12}", end="")
    for px in SWEEP_PRICES:
        print(f"  {f'≥{px:.3f}':>{col_w-2}}", end="")
    print()
    print("  " + "─" * (12 + col_w * len(SWEEP_PRICES)))

    ref_wr = None
    ref_cell = (150, 0.90)

    for min_d in [0] + SWEEP_DISTANCES:
        label = f"≥${min_d}" if min_d > 0 else "any dist"
        print(f"  {label:>12}", end="")
        for min_px in SWEEP_PRICES:
            mask = (
                (df[px_col] >= min_px) &
                (df[px_col] < 1.0) &
                (df[dist_col] >= min_d)
            )
            sub = df[mask]
            n   = len(sub)
            if n < MIN_TRADES:
                print(f"  {'—':>{col_w-2}}", end="")
            else:
                wr  = float(sub["outcome"].mean())
                avg = float(sub[px_col].mean())
                ev  = _ev(wr, avg)
                if min_d == ref_cell[0] and min_px == ref_cell[1]:
                    ref_wr = wr
                star = " ★" if ev > 0 else "  "
                print(f"  {f'{wr:.1%}({ev:+.3f}){star}':>{col_w-2}}", end="")
        print()

    print(f"{'═'*85}")
    if ref_wr:
        print(f"  Reference (dist≥$150, price≥0.90): win={ref_wr:.1%}  "
              f"← cells with ★ have positive EV regardless of win rate")


# ── Main ──────────────────────────────────────────────────────────────────────

def backtest(prices_zips: list[str], entry_second: int, sweep: bool,
             btc_vol: float, output_path: str | None) -> None:

    all_rows: list[dict] = []
    for zp in prices_zips:
        all_rows.extend(_load_zip(zp, SWEEP_SECONDS if sweep else [entry_second]))

    df = pd.DataFrame(all_rows)
    n_up = int((df["outcome"] == 1).sum())
    print(f"\nTotal: {len(df)} settled markets  "
          f"(Up={n_up}  Down={len(df)-n_up}  base_rate={n_up/len(df):.1%})")

    if sweep:
        print(f"\nBTC vol assumption: {btc_vol:.0%} annual  "
              f"(override with --btc-vol)")

        # 1. Price → distance reference table
        avg_strike = float(df["strike"].dropna().mean()) if df["strike"].notna().any() else 95_000
        _print_price_dist_ref(btc_vol, strike=round(avg_strike, -3))

        # 2. Price-based sweep
        _print_price_sweep(df)

        # 3. Distance-based sweep
        _print_distance_sweep(df, btc_vol)

        # 4. 2D grid at the best entry time (t=300s)
        _print_price_x_dist_grid(df, 300, btc_vol)

    else:
        px_col = f"px_{entry_second}"
        if px_col not in df.columns:
            print(f"No data for t={entry_second}s")
            return
        bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.875,
                0.90, 0.915, 0.93, 0.945, 0.96, 0.975, 1.01]
        df["_b"] = pd.cut(df[px_col], bins=bins)
        g = df.groupby("_b", observed=True)
        print(f"\n{'═'*72}")
        print(f"  Win Rate by Entry Price at t={entry_second}s")
        print(f"{'─'*72}")
        print(f"  {'Bucket':>22}  {'N':>6}  {'Win%':>7}  {'Edge':>7}  {'EV/$1':>7}")
        print(f"{'─'*72}")
        for b, grp in g:
            n = len(grp)
            if n < MIN_TRADES:
                continue
            wr  = float(grp["outcome"].mean())
            avg = float(grp[px_col].mean())
            print(f"  {str(b):>22}  {n:>6}  {wr:>6.1%}  {wr-b.mid:>+6.3f}  {_ev(wr,avg):>+6.4f}")
        print(f"{'═'*72}")

    if output_path:
        cols = (["slug", "date", "strike", "outcome"] +
                [f"px_{s}" for s in (SWEEP_SECONDS if sweep else [entry_second])])
        df[[c for c in cols if c in df.columns]].to_csv(output_path, index=False)
        print(f"\nSaved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest KXBTC15M strategy across entry time, price, and distance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml/backtest.py --prices "E:\\...zip" "E:\\...zip" --sweep
  python ml/backtest.py --prices "E:\\...zip" --sweep --btc-vol 0.70
  python ml/backtest.py --prices "E:\\...zip" --entry-second 300 --output bt.csv
""")
    parser.add_argument("--prices", required=True, nargs="+")
    parser.add_argument("--sweep", action="store_true",
                        help="Full sweep: entry time × price × distance")
    parser.add_argument("--entry-second", type=int, default=300,
                        help="Entry time for single-point analysis (default 300)")
    parser.add_argument("--btc-vol", type=float, default=0.65,
                        help="BTC annual volatility assumption (default 0.65 = 65%%)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    backtest(args.prices, args.entry_second, args.sweep, args.btc_vol, args.output)
