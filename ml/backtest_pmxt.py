"""
ml/backtest_pmxt.py — Backtest BTC 15m strategy using the pmxt API.

Fetches KXBTC15M settled markets directly from Kalshi via pmxt, samples
the orderbook at multiple entry times, and runs the same sweep analysis
as backtest.py — but across much larger date ranges without parquet files.

Usage:
    # First run: fetch data and cache to CSV
    python ml\\backtest_pmxt.py --sweep --cache results.csv

    # Re-run analysis on cached data (no API calls)
    python ml\\backtest_pmxt.py --sweep --from-cache results.csv

    # With API key (faster, higher rate limits)
    python ml\\backtest_pmxt.py --sweep --api-key pmxt_xxx --cache results.csv

    # Focus on a specific entry time
    python ml\\backtest_pmxt.py --entry-second 120 --from-cache results.csv

    # Limit date range
    python ml\\backtest_pmxt.py --sweep --days 90 --cache results.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants (shared with backtest.py) ───────────────────────────────────────

MARKET_DURATION  = 900
SECS_PER_YEAR    = 365.25 * 24 * 3600
SETTLED_THRESH   = 0.90
MIN_TRADES       = 10
REF_WIN_RATE     = 0.87
REF_TOLERANCE    = 0.03

SWEEP_SECONDS   = [30, 60, 90, 120, 150, 180, 300, 420, 540, 600, 660, 720, 750, 810, 870]
SWEEP_PRICES    = [0.75, 0.80, 0.85, 0.875, 0.90, 0.915, 0.93, 0.945, 0.96]
SWEEP_DISTANCES = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]

# ── Implied distance (Bachelier) ───────────────────────────────────────────────

try:
    from scipy.stats import norm as _norm
    def _ppf(p):
        return _norm.ppf(p)
except ImportError:
    def _ppf(p):
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
        m = (p >= lo) & (p <= hi); q = p[m] - 0.5; r = q * q
        out[m] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                 (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        lm = p < lo; q = np.sqrt(-2*np.log(p[lm]))
        out[lm] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        hm = p > hi; q = np.sqrt(-2*np.log(1-p[hm]))
        out[hm] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return float(out) if out.ndim == 0 else out


def implied_dist(px, entry_second, strike, btc_vol):
    remaining_sec   = max(MARKET_DURATION - entry_second, 1)
    remaining_years = remaining_sec / SECS_PER_YEAR
    sigma_t         = btc_vol * np.sqrt(remaining_years)
    return _ppf(np.clip(px, 0.001, 0.999)) * sigma_t * strike


# ── Strike parsing ─────────────────────────────────────────────────────────────

_STRIKE_RE = [
    r'[tTbB](\d{4,6})(?:[^0-9]|$)',
    r'above[_-]?(\d{4,6})',
    r'below[_-]?(\d{4,6})',
    r'[-_](\d{5,6})[-_uU]',
    r'(\d{5,6})(?:usd|USD)?(?:[-_k]|$)',
]

def _parse_strike(s: str) -> float | None:
    for pat in _STRIKE_RE:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 10_000 <= v <= 500_000:
                return v
    return None


# ── pmxt data fetch ────────────────────────────────────────────────────────────

def _make_kalshi(api_key: str | None):
    import pmxt
    if api_key:
        return pmxt.Kalshi(pmxt_api_key=api_key)
    return pmxt.Kalshi()


def _with_retry(fn, label: str = "", max_attempts: int = 6):
    """Call fn(), retrying with exponential backoff on 429 / rate-limit errors."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "too many" in msg:
                wait = min(2 ** attempt, 60)
                print(f"    rate-limited{' (' + label + ')' if label else ''} — waiting {wait}s …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gave up after {max_attempts} retries ({label})")


def _is_kxbtc15m(outcome_id: str) -> bool:
    """Strict check: ticker must match KXBTC15M-DDMMMYY-T{strike} pattern."""
    return bool(re.match(r'^KXBTC15M-', outcome_id, re.IGNORECASE))


def _fetch_all_markets(kalshi, days: int | None) -> list:
    """Page through settled KXBTC15M markets only."""
    cutoff = None
    now    = datetime.now(timezone.utc)
    if days:
        cutoff = now - timedelta(days=days)

    markets = []
    cursor  = None
    page    = 0

    while True:
        params: dict = {
            "slug":  "KXBTC15M",   # slug prefix match — tighter than free-text query
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        result = _with_retry(
            lambda: kalshi.fetch_markets_paginated(params),
            label=f"page {page+1}",
        )
        batch = result.data
        page += 1
        kept  = 0

        for mkt in batch:
            if mkt.resolution_date is None:
                continue
            if mkt.resolution_date > now:
                continue
            if cutoff and mkt.resolution_date < cutoff:
                continue
            outcome = mkt.up or mkt.yes
            if outcome is None:
                continue
            # Hard filter: reject anything that isn't exactly KXBTC15M-*
            if not _is_kxbtc15m(outcome.outcome_id):
                continue
            markets.append(mkt)
            kept += 1

        print(f"  page {page}: {len(batch)} fetched, {kept} KXBTC15M kept ({len(markets)} total)")

        if not result.next_cursor or len(batch) == 0:
            break
        cursor = result.next_cursor
        time.sleep(1.5)

    return markets


def _best_bid(ob) -> float | None:
    if ob and ob.bids:
        return ob.bids[0].price
    return None

def _best_ask(ob) -> float | None:
    if ob and ob.asks:
        return ob.asks[0].price
    return None


def _sample_market(kalshi, mkt, entry_seconds: list[int], btc_vol: float, rate_delay: float) -> list[dict]:
    """
    Fetch orderbook snapshots at each entry time and determine settlement.
    Returns one row per (market × entry_second) pair.
    """
    outcome = mkt.up or mkt.yes
    if outcome is None:
        return []

    outcome_id  = outcome.outcome_id
    close_dt    = mkt.resolution_date
    if close_dt is None:
        return []

    close_ms  = int(close_dt.timestamp() * 1000)
    open_ms   = close_ms - MARKET_DURATION * 1000
    strike    = _parse_strike(outcome_id) or _parse_strike(mkt.title or "")
    slug      = outcome_id
    date_str  = close_dt.strftime("%Y-%m-%d")

    # ── Single range call: covers entry window + settlement in one request ──────
    try:
        all_books = _with_retry(
            lambda: kalshi.fetch_order_book(
                outcome_id,
                params={"since": open_ms, "until": close_ms, "limit": 1000},
            ),
            label=outcome_id,
        )
        if not isinstance(all_books, list):
            all_books = [all_books] if all_books else []
    except Exception as e:
        print(f"    fetch error {outcome_id}: {e}", file=sys.stderr)
        return []

    time.sleep(rate_delay)

    if not all_books:
        return []

    # Settlement: use last snapshot in range
    settle_ob  = all_books[-1]
    settle_bid = _best_bid(settle_ob)
    if settle_bid is None:
        return []
    if settle_bid >= SETTLED_THRESH:
        outcome_val = 1
    elif settle_bid <= (1.0 - SETTLED_THRESH):
        outcome_val = 0
    else:
        return []

    books   = all_books
    ts_list = [b.timestamp for b in books if b.timestamp is not None]
    if not ts_list:
        return []

    rows = []
    for t_sec in entry_seconds:
        target_ms = open_ms + t_sec * 1000
        # Find closest snapshot at or before target_ms
        candidates = [(abs(b.timestamp - target_ms), b)
                      for b in books if b.timestamp is not None and b.timestamp <= target_ms + 5000]
        if not candidates:
            continue
        _, ob = min(candidates, key=lambda x: x[0])

        ask = _best_ask(ob)
        bid = _best_bid(ob)
        if ask is None or ask <= 0 or ask >= 1:
            continue
        mid = (bid + ask) / 2 if bid else ask

        ask_size   = ob.asks[0].size if ob.asks else 0.0
        bid_size   = ob.bids[0].size if ob.bids else 0.0
        book_depth = sum(l.size for l in ob.asks) + sum(l.size for l in ob.bids)

        rows.append({
            "slug":         slug,
            "date":         date_str,
            "strike":       strike,
            "entry_second": t_sec,
            "entry_price":  ask,
            "mid":          mid,
            "ask_size":     ask_size,
            "bid_size":     bid_size,
            "book_depth":   book_depth,
            "outcome":      outcome_val,
        })

    return rows


# ── Analysis tables (same as backtest.py) ─────────────────────────────────────

def _ev(win_rate: float, avg_price: float) -> float:
    if avg_price <= 0:
        return float("nan")
    return win_rate * (1.0 / avg_price - 1.0) - (1.0 - win_rate)


def _fmt(win: float, n: int) -> str:
    star = "★" if abs(win - REF_WIN_RATE) <= REF_TOLERANCE else " "
    return f"{win:.0%}/{n:>4}{star}"


def _print_price_sweep(df: pd.DataFrame) -> None:
    print(f"\n{'═'*86}")
    print("  Win Rate by Entry Time × Min Contract Price  (ask ≥ threshold)")
    print(f"{'─'*86}")
    print(f"  {'t(s)':>8}", end="")
    for px in SWEEP_PRICES:
        print(f"  {px:>10.3f}", end="")
    print()
    print("  " + "─" * (8 + 12 * len(SWEEP_PRICES)))
    for t in SWEEP_SECONDS:
        sub = df[df["entry_second"] == t]
        if len(sub) == 0:
            continue
        print(f"  {t:>7}s", end="")
        for px in SWEEP_PRICES:
            grp = sub[sub["entry_price"] >= px]
            n   = len(grp)
            if n < MIN_TRADES:
                print(f"  {'—':>10}", end="")
                continue
            wr = float(grp["outcome"].mean())
            print(f"  {_fmt(wr, n):>10}", end="")
        print()
    print(f"{'═'*86}")
    print(f"  ★ = within ±{REF_TOLERANCE:.0%} of reference ({REF_WIN_RATE:.0%})")

    # EV companion
    print(f"\n{'═'*86}")
    print("  EV per $1 wagered by Entry Time × Min Contract Price")
    print(f"{'─'*86}")
    print(f"  {'t(s)':>8}", end="")
    for px in SWEEP_PRICES:
        print(f"  {px:>10.3f}", end="")
    print()
    print("  " + "─" * (8 + 12 * len(SWEEP_PRICES)))
    for t in SWEEP_SECONDS:
        sub = df[df["entry_second"] == t]
        if len(sub) == 0:
            continue
        print(f"  {t:>7}s", end="")
        for px in SWEEP_PRICES:
            grp = sub[sub["entry_price"] >= px]
            n   = len(grp)
            if n < MIN_TRADES:
                print(f"  {'—':>10}", end="")
                continue
            wr = float(grp["outcome"].mean())
            ep = float(grp["entry_price"].mean())
            e  = _ev(wr, ep)
            star = "★" if e > 0 else " "
            print(f"  {e:>+9.3f}{star}", end="")
        print()
    print(f"{'═'*86}")


def _print_distance_sweep(df: pd.DataFrame, btc_vol: float) -> None:
    rows = df.copy()
    has  = rows["strike"].notna()
    rows.loc[has, "implied_dist"] = rows[has].apply(
        lambda r: implied_dist(r["entry_price"], int(r["entry_second"]), r["strike"], btc_vol),
        axis=1,
    )

    print(f"\n{'═'*90}")
    print(f"  Win Rate by Entry Time × Min Implied Distance  (vol={btc_vol:.0%})")
    print(f"{'─'*90}")
    print(f"  {'t(s)':>8}", end="")
    for d in SWEEP_DISTANCES:
        print(f"  {d:>9}$", end="")
    print()
    print("  " + "─" * (8 + 11 * len(SWEEP_DISTANCES)))
    for t in SWEEP_SECONDS:
        sub = rows[rows["entry_second"] == t]
        hd  = sub["implied_dist"].notna()
        if len(sub) == 0:
            continue
        print(f"  {t:>7}s", end="")
        for d in SWEEP_DISTANCES:
            mask = hd & (sub["implied_dist"] >= d)
            grp  = sub[mask]
            n    = len(grp)
            if n < MIN_TRADES:
                print(f"  {'—':>9}", end="")
                continue
            wr = float(grp["outcome"].mean())
            print(f"  {_fmt(wr, n):>9}", end="")
        print()
    print(f"{'═'*90}")

    # EV companion
    print(f"\n{'═'*90}")
    print("  EV per $1 wagered by Entry Time × Min Implied Distance")
    print(f"{'─'*90}")
    print(f"  {'t(s)':>8}", end="")
    for d in SWEEP_DISTANCES:
        print(f"  {d:>9}$", end="")
    print()
    print("  " + "─" * (8 + 11 * len(SWEEP_DISTANCES)))
    for t in SWEEP_SECONDS:
        sub = rows[rows["entry_second"] == t]
        hd  = sub["implied_dist"].notna()
        if len(sub) == 0:
            continue
        print(f"  {t:>7}s", end="")
        for d in SWEEP_DISTANCES:
            mask = hd & (sub["implied_dist"] >= d)
            grp  = sub[mask]
            n    = len(grp)
            if n < MIN_TRADES:
                print(f"  {'—':>9}", end="")
                continue
            wr  = float(grp["outcome"].mean())
            ep  = float(grp["entry_price"].mean())
            e   = _ev(wr, ep)
            star = "★" if e > 0 else " "
            print(f"  {e:>+8.3f}{star}", end="")
        print()
    print(f"{'═'*90}")


def _print_2d_grid(df: pd.DataFrame, entry_second: int, btc_vol: float) -> None:
    sub = df[df["entry_second"] == entry_second].copy()
    has = sub["strike"].notna()
    sub.loc[has, "implied_dist"] = sub[has].apply(
        lambda r: implied_dist(r["entry_price"], entry_second, r["strike"], btc_vol),
        axis=1,
    )
    sub = sub[sub["implied_dist"].notna()]
    if len(sub) < 10:
        print(f"  (Not enough data at t={entry_second}s for 2D grid)")
        return

    price_bins = [0.65, 0.70, 0.75, 0.80, 0.85, 0.875, 0.90, 0.915, 0.93, 0.945, 0.96, 1.01]
    dist_bins  = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 9999]

    sub["pb"] = pd.cut(sub["entry_price"], bins=price_bins)
    sub["db"] = pd.cut(sub["implied_dist"], bins=dist_bins)

    pwr  = sub.pivot_table(index="db", columns="pb", values="outcome", aggfunc="mean",  observed=True)
    pcnt = sub.pivot_table(index="db", columns="pb", values="outcome", aggfunc="count", observed=True)

    print(f"\n{'═'*100}")
    remaining = MARKET_DURATION - entry_second
    print(f"  2D Grid at t={entry_second}s ({remaining//60}m{remaining%60}s left)"
          f"  row=implied dist  col=entry price")
    print(f"  ★ within ±{REF_TOLERANCE:.0%} of {REF_WIN_RATE:.0%} reference")
    print(f"{'─'*100}")
    hdr_label = "dist \\ price"
    print(f"  {hdr_label:>16}", end="")
    for c in pwr.columns:
        print(f"  {str(c):>15}", end="")
    print()
    print("  " + "─" * (16 + 17 * len(pwr.columns)))
    for row_b in pwr.index:
        print(f"  {str(row_b):>16}", end="")
        for c in pwr.columns:
            wr  = pwr.loc[row_b, c]
            cnt = pcnt.loc[row_b, c] if not pd.isna(pcnt.loc[row_b, c]) else 0
            if pd.isna(wr) or cnt < MIN_TRADES:
                print(f"  {'—':>15}", end="")
            else:
                star = "★" if abs(wr - REF_WIN_RATE) <= REF_TOLERANCE else " "
                cell = f"{wr:.0%} n={int(cnt)}{star}"
                print(f"  {cell:>15}", end="")
        print()
    print(f"{'═'*100}")


# ── Volume analysis (early entry: first 3 minutes) ───────────────────────────

EARLY_SECONDS  = [30, 60, 90, 120, 150, 180]
VOL_THRESHOLDS = [10, 25, 50, 100, 200, 500]   # ask_size contract thresholds


def _print_volume_analysis(df: pd.DataFrame) -> None:
    """
    Tests the volume-gated entry hypothesis:
      Rule A: price ≥ 0.80  AND  ask_size ≥ threshold  (high volume = confident)
      Rule B: price ≥ 0.85  AND  ask_size <  threshold  (low  volume = need more edge)
      Combined: A OR B

    Prints:
      1. Volume distribution at each early entry time
      2. Price × volume tier win rate grid at t=120s
      3. Combined-rule sweep across volume thresholds
    """
    if "ask_size" not in df.columns:
        print("\n  (No volume data — re-fetch with updated script)")
        return

    early = df[df["entry_second"].isin(EARLY_SECONDS)].copy()
    if len(early) < MIN_TRADES:
        print("\n  (Not enough early-entry rows for volume analysis)")
        return

    n_days = early["date"].nunique() if "date" in early.columns else 1

    # ── 1. Volume distribution ────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  Ask-Size Distribution at Entry  (contracts available at best ask)")
    print(f"{'─'*72}")
    print(f"  {'t(s)':>7}  {'p10':>6}  {'p25':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}  {'mean':>7}  {'n':>6}")
    print("  " + "─" * 64)
    for t in EARLY_SECONDS:
        sub = early[(early["entry_second"] == t) & early["ask_size"].notna()]
        if len(sub) < 5:
            continue
        s = sub["ask_size"]
        print(f"  {t:>6}s  {s.quantile(.10):>6.0f}  {s.quantile(.25):>6.0f}  "
              f"{s.quantile(.50):>6.0f}  {s.quantile(.75):>6.0f}  "
              f"{s.quantile(.90):>6.0f}  {s.mean():>7.1f}  {len(sub):>6}")
    print(f"{'═'*72}")

    # ── 2. Price × volume tier grid at t=120s ────────────────────────────────
    focus = early[early["entry_second"] == 120].copy()
    if len(focus) >= MIN_TRADES:
        p33 = focus["ask_size"].quantile(0.33)
        p67 = focus["ask_size"].quantile(0.67)

        def _tier(s):
            if s < p33:   return f"LOW (<{p33:.0f})"
            if s < p67:   return f"MED ({p33:.0f}–{p67:.0f})"
            return         f"HIGH (>{p67:.0f})"

        focus["vtier"] = focus["ask_size"].apply(_tier)
        tiers = [f"LOW (<{p33:.0f})", f"MED ({p33:.0f}–{p67:.0f})", f"HIGH (>{p67:.0f})"]
        prices = [0.75, 0.80, 0.85, 0.875, 0.90]

        print(f"\n{'═'*80}")
        print(f"  Price × Volume Tier at t=120s  "
              f"(LOW=bot 33%  MED=mid  HIGH=top 33%  of ask_size)")
        print(f"  format: win% / n / EV")
        print(f"{'─'*80}")
        col_w = 20
        print(f"  {'price':>10}", end="")
        for tier in tiers:
            print(f"  {tier:>{col_w}}", end="")
        print()
        print("  " + "─" * (10 + (col_w + 2) * len(tiers)))
        for px in prices:
            print(f"  {px:>10.3f}", end="")
            for tier in tiers:
                grp = focus[(focus["entry_price"] >= px) & (focus["vtier"] == tier)]
                n   = len(grp)
                if n < MIN_TRADES:
                    print(f"  {'—':>{col_w}}", end="")
                    continue
                wr = float(grp["outcome"].mean())
                ep = float(grp["entry_price"].mean())
                e  = _ev(wr, ep)
                cell = f"{wr:.0%} / {n} / {e:+.3f}"
                print(f"  {cell:>{col_w}}", end="")
            print()
        print(f"{'═'*80}")

    # ── 3. Combined rule sweep ────────────────────────────────────────────────
    print(f"\n{'═'*92}")
    print("  Combined Rule Sweep (first 3 minutes: t=30–180s)")
    print("  Rule A: price ≥ 0.80  AND  ask_size ≥ threshold  (high volume, lower price ok)")
    print("  Rule B: price ≥ 0.85  AND  ask_size <  threshold  (low  volume, need more edge)")
    print("  Combined = A OR B (mutually exclusive by threshold)")
    print(f"{'─'*92}")
    print(f"  {'threshold':>12}  "
          f"{'Rule A win%':>12}  {'A EV':>8}  {'A n/day':>8}  "
          f"{'Rule B win%':>12}  {'B EV':>8}  {'B n/day':>8}  "
          f"{'Combined win%':>14}  {'Comb EV':>8}  {'Comb n/day':>10}")
    print("  " + "─" * 90)

    for thresh in VOL_THRESHOLDS:
        a = early[(early["entry_price"] >= 0.80) & (early["ask_size"] >= thresh)]
        b = early[(early["entry_price"] >= 0.85) & (early["ask_size"] <  thresh)]
        combined = pd.concat([a, b]).drop_duplicates()

        def _row(grp):
            if len(grp) < MIN_TRADES:
                return "—", float("nan"), 0.0
            wr = float(grp["outcome"].mean())
            ep = float(grp["entry_price"].mean())
            return f"{wr:.1%}", _ev(wr, ep), len(grp) / n_days

        a_wr, a_ev, a_nd = _row(a)
        b_wr, b_ev, b_nd = _row(b)
        c_wr, c_ev, c_nd = _row(combined)

        def _ev_str(e): return f"{e:+.3f}" if not pd.isna(e) else "—"

        print(f"  ask≥{thresh:>7}  "
              f"  {a_wr:>12}  {_ev_str(a_ev):>8}  {a_nd:>8.1f}  "
              f"  {b_wr:>12}  {_ev_str(b_ev):>8}  {b_nd:>8.1f}  "
              f"  {c_wr:>14}  {_ev_str(c_ev):>8}  {c_nd:>10.1f}")

    print(f"{'═'*92}")
    print("  ★ Best combined rule = highest EV × n/day product")

    # Find best combined row
    best = None
    best_score = -999
    for thresh in VOL_THRESHOLDS:
        a = early[(early["entry_price"] >= 0.80) & (early["ask_size"] >= thresh)]
        b = early[(early["entry_price"] >= 0.85) & (early["ask_size"] <  thresh)]
        combined = pd.concat([a, b]).drop_duplicates()
        if len(combined) < MIN_TRADES:
            continue
        wr = float(combined["outcome"].mean())
        ep = float(combined["entry_price"].mean())
        e  = _ev(wr, ep)
        nd = len(combined) / n_days
        score = e * nd
        if score > best_score:
            best_score = score
            best = (thresh, wr, e, nd)

    if best:
        thresh, wr, e, nd = best
        print(f"\n  ★  Best: ask_size threshold = {thresh}")
        print(f"       Rule A (≥0.80 + ask≥{thresh}): buy when volume confirms")
        print(f"       Rule B (≥0.85 + ask<{thresh}):  buy when volume is thin but price is strong")
        print(f"       Combined: win={wr:.1%}  EV={e:+.3f}  ~{nd:.1f} trades/day")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    api_key:      str | None,
    days:         int | None,
    sweep:        bool,
    entry_second: int,
    btc_vol:      float,
    cache_path:   str | None,
    from_cache:   str | None,
    rate_delay:   float,
) -> None:

    if from_cache:
        print(f"Loading cached data from {from_cache} …")
        df = pd.read_csv(from_cache)
        df["entry_second"] = df["entry_second"].astype(int)
    else:
        import pmxt
        kalshi = _make_kalshi(api_key)

        print("Fetching KXBTC15M markets from Kalshi …")
        markets = _fetch_all_markets(kalshi, days)
        print(f"Found {len(markets)} settled markets\n")

        if not markets:
            print("No markets found. Try --days 180 or check your API key.")
            return

        # Always include early seconds so volume analysis has data
        entry_times = SWEEP_SECONDS if sweep else sorted(set(EARLY_SECONDS) | {entry_second, 300})

        all_rows: list[dict] = []
        for i, mkt in enumerate(markets):
            outcome = mkt.up or mkt.yes
            ticker  = outcome.outcome_id if outcome else "?"
            print(f"  [{i+1}/{len(markets)}] {ticker} …", end=" ", flush=True)
            rows = _sample_market(kalshi, mkt, entry_times, btc_vol, rate_delay)
            all_rows.extend(rows)
            print(f"{len(rows)//max(len(entry_times),1)} samples")

        df = pd.DataFrame(all_rows)

        if cache_path and len(df):
            df.to_csv(cache_path, index=False)
            print(f"\nCached {len(df)} rows → {cache_path}")

    if len(df) == 0:
        print("No data to analyse.")
        return

    markets_n = df["slug"].nunique()
    days_n    = df["date"].nunique() if "date" in df.columns else "?"
    print(f"\nDataset: {markets_n} markets  {days_n} days")
    print(f"BTC vol: {btc_vol:.0%}  |  base rate: {df[df['entry_second']==300]['outcome'].mean():.1%} (at t=300s)")

    valid_strikes = df["strike"].dropna()
    strike_ref = float(valid_strikes.median()) if len(valid_strikes) > 0 else 100_000.0
    print(f"Median strike: ${strike_ref:,.0f}")

    if sweep:
        _print_price_sweep(df)
        _print_distance_sweep(df, btc_vol)

    _print_2d_grid(df, entry_second, btc_vol)
    _print_volume_analysis(df)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest BTC 15m strategy via pmxt API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key",      default=None, help="pmxt API key (pmxt_...)")
    parser.add_argument("--days",         type=int, default=None, help="Limit to last N days")
    parser.add_argument("--sweep",        action="store_true", help="Full sweep tables")
    parser.add_argument("--entry-second", type=int, default=120, help="Entry time for 2D grid (default 120)")
    parser.add_argument("--btc-vol",      type=float, default=0.65, help="Annual BTC vol (default 0.65)")
    parser.add_argument("--cache",        default=None, help="Save fetched rows to CSV")
    parser.add_argument("--from-cache",   default=None, help="Skip API fetch, load from CSV")
    parser.add_argument("--rate-delay",   type=float, default=0.3, help="Seconds between API calls (default 0.3)")
    args = parser.parse_args()
    run(
        api_key      = args.api_key,
        days         = args.days,
        sweep        = args.sweep,
        entry_second = args.entry_second,
        btc_vol      = args.btc_vol,
        cache_path   = args.cache,
        from_cache   = args.from_cache,
        rate_delay   = args.rate_delay,
    )
