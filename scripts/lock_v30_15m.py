"""
scripts/lock_v30_15m.py — Lock Strategy V30 backtest on KXBTC15M (15-minute BTC).

Family: MID_70_90_CONSENSUS_EDGE
Candidate 37081 (A+) — fitted for KXBTC15M (15-minute markets).

Active Settings:
  T-left:   [10s, 60s]
  Ask:      [70c, 90c]
  Spread:   max 2c
  Prob Polymarket (proxy via Bachelier):  min 80%
  Edge Polymarket (proxy):               min 2c
  Prob Binance (Bachelier strict):        min 92%
  Distance norm Binance:                  min 1.25

Trade execution:
  Order size: $10  |  Slippage: 3c  |  Exit: hold to settlement

Usage:
  # Fetch data and run
  python scripts/lock_v30_15m.py --days 30 --cache lock_v30_15m.csv

  # Re-run analysis from cache (no API calls)
  python scripts/lock_v30_15m.py --from-cache lock_v30_15m.csv

  # With pmxt API key (higher rate limits)
  python scripts/lock_v30_15m.py --days 60 --api-key pmxt_xxx --cache lock_v30_15m.csv

  # Sweep parameter sensitivity
  python scripts/lock_v30_15m.py --from-cache lock_v30_15m.csv --sweep

  # Override BTC vol assumption
  python scripts/lock_v30_15m.py --from-cache lock_v30_15m.csv --btc-vol 0.70

Notes:
  - "Prob Polymarket" is approximated by the same Bachelier model at a relaxed threshold.
    In live trading this would use the actual Polymarket 15m BTC order book price.
  - "Distance norm Binance" uses actual Coinbase BTC spot vs the Kalshi strike.
  - Slippage (3c) is added on top of the quoted ask at entry.
  - Settlement determined by the last orderbook snapshot before market close.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

MARKET_DURATION  = 900           # 15 minutes in seconds (KXBTC15M)
SECS_PER_YEAR    = 365.25 * 24 * 3600
SETTLED_THRESH   = 0.90
BTC_SERIES       = "KXBTC15M"

# Lock Strategy V30 — Candidate 37081 (A+) parameters for 15m
ORDER_SIZE       = 10.0          # dollars per trade
SLIPPAGE         = 0.03          # 3 cents fill slippage added to ask
ENTRY_T_MIN      = 10            # minimum T-left seconds
ENTRY_T_MAX      = 60            # maximum T-left seconds
ASK_MIN          = 0.70          # 70c
ASK_MAX          = 0.90          # 90c
SPREAD_MAX       = 0.02          # 2c
DIST_NORM_MIN    = 1.25          # distance / σ(t_remaining)
PROB_BINANCE_MIN = 0.92          # Φ(distance_norm) strict gate
PROB_POLY_MIN    = 0.80          # relaxed consensus gate
EDGE_POLY_MIN    = 0.02          # model_prob − ask ≥ 2c

# Entry sampling: every 5s through T-left [10, 60] window
# For 15min market: seconds elapsed = 840..890 → T-left = 10..60
ENTRY_SECONDS = list(range(
    MARKET_DURATION - ENTRY_T_MAX,
    MARKET_DURATION - ENTRY_T_MIN + 1,
    5,
))  # [840, 845, 850, 855, 860, 865, 870, 875, 880, 885, 890]

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# ── Normal CDF / PPF ───────────────────────────────────────────────────────────

try:
    from scipy.stats import norm as _norm
    def _cdf(z: float | np.ndarray) -> float | np.ndarray:
        return _norm.cdf(z)
    def _ppf(p: float | np.ndarray) -> float | np.ndarray:
        return _norm.ppf(p)
except ImportError:
    def _cdf(z: float) -> float:
        """Abramowitz & Stegun approximation of Φ(z)."""
        z = float(z)
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530
               + t * (-0.356563782
               + t * (1.781477937
               + t * (-1.821255978
               + t * 1.330274429))))
        p = 1.0 - 0.3989422804 * np.exp(-0.5 * z * z) * poly
        return p if z >= 0 else 1.0 - p

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
        m = (p >= lo) & (p <= hi)
        q = p[m] - 0.5; r = q * q
        out[m] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                 (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        lm = p < lo
        q = np.sqrt(-2 * np.log(np.maximum(p[lm], 1e-300)))
        out[lm] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        hm = p > hi
        q = np.sqrt(-2 * np.log(np.maximum(1 - p[hm], 1e-300)))
        out[hm] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return float(out) if out.ndim == 0 else out


# ── BTC distance model ─────────────────────────────────────────────────────────

def distance_norm(btc_spot: float, strike: float, t_remaining_sec: float, btc_vol: float) -> float:
    """
    Normalised Bachelier distance: (btc_spot − strike) / (σ × btc_spot × √t).
    Positive = BTC is ABOVE strike → favours YES.
    """
    if t_remaining_sec <= 0 or btc_spot <= 0 or strike <= 0:
        return 0.0
    t_years = t_remaining_sec / SECS_PER_YEAR
    sigma_dollar = btc_vol * btc_spot * np.sqrt(t_years)
    if sigma_dollar < 1e-8:
        return 0.0
    return (btc_spot - strike) / sigma_dollar


def binance_prob(d_norm: float) -> float:
    """Probability YES resolves (BTC above strike) given distance_norm."""
    return float(_cdf(d_norm))


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


# ── Coinbase BTC 1m candle fetch ───────────────────────────────────────────────

def fetch_btc_klines(start_ts: int, end_ts: int) -> dict[int, float]:
    """
    Fetch Coinbase BTC/USD 1m candles and return {unix_second_ts: close_price}.
    Covers the range [start_ts, end_ts] with 300-candle batches.
    """
    result: dict[int, float] = {}
    BATCH_SECS = 300 * 60
    current = start_ts

    while current < end_ts:
        batch_end  = min(current + BATCH_SECS, end_ts)
        start_iso  = datetime.fromtimestamp(current,   tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso    = datetime.fromtimestamp(batch_end, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = f"{COINBASE_CANDLES_URL}?granularity=60&start={start_iso}&end={end_iso}"
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json", "User-Agent": "lock-v30-backtest/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                candles = json.loads(resp.read())
            for c in candles:
                result[int(c[0])] = float(c[4])   # close price
        except Exception as exc:
            print(f"  [warn] Coinbase fetch failed: {exc}", file=sys.stderr)
            time.sleep(1.0)
        current = batch_end + 1
        time.sleep(0.25)

    return result


def btc_price_at(klines: dict[int, float], ts: int) -> float | None:
    """Return the BTC close price at the 1m bar containing timestamp ts."""
    minute = (ts // 60) * 60
    for offset in (0, -60, 60, -120, 120, -180):
        v = klines.get(minute + offset)
        if v is not None:
            return v
    return None


# ── pmxt helpers ──────────────────────────────────────────────────────────────

def _make_kalshi(api_key: str | None):
    import pmxt
    return pmxt.Kalshi(pmxt_api_key=api_key) if api_key else pmxt.Kalshi()


def _with_retry(fn, label: str = "", max_attempts: int = 6):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "too many" in msg:
                wait = min(2 ** attempt, 60)
                print(f"    rate-limited ({label}) — waiting {wait}s …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gave up after {max_attempts} retries ({label})")


def _is_kxbtc15m(s: str) -> bool:
    return bool(re.match(r'^KXBTC15M-', s, re.IGNORECASE))


def _fetch_all_markets(kalshi, days: int | None) -> list:
    cutoff = None
    now    = datetime.now(timezone.utc)
    if days:
        cutoff = now - timedelta(days=days)

    markets, cursor, page = [], None, 0
    while True:
        params: dict = {"series_ticker": BTC_SERIES, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        result = _with_retry(
            lambda: kalshi.fetch_markets_paginated(params), label=f"page {page + 1}"
        )
        batch  = result.data
        page  += 1
        kept   = 0

        for mkt in batch:
            if mkt.resolution_date is None or mkt.resolution_date > now:
                continue
            if cutoff and mkt.resolution_date < cutoff:
                continue
            outcome = mkt.up or mkt.yes
            if outcome is None:
                continue
            ticker = getattr(mkt, "ticker", "") or ""
            out_id = outcome.outcome_id or ""
            if not (_is_kxbtc15m(out_id) or _is_kxbtc15m(ticker)):
                continue
            markets.append(mkt)
            kept += 1

        print(f"  page {page}: {len(batch)} fetched, {kept} KXBTC15M kept ({len(markets)} total)")
        if not result.next_cursor or len(batch) == 0:
            break
        cursor = result.next_cursor
        time.sleep(1.5)

    return markets


def _best_ask(ob) -> float | None:
    return ob.asks[0].price if ob and ob.asks else None


def _best_bid(ob) -> float | None:
    return ob.bids[0].price if ob and ob.bids else None


# ── Market sampling ────────────────────────────────────────────────────────────

def _sample_market(
    kalshi,
    mkt,
    btc_klines: dict[int, float],
    btc_vol: float,
    rate_delay: float,
) -> list[dict]:
    """
    Fetch the full orderbook history for one KXBTC15M market and return one
    candidate row for each tick in the T-left [10, 60] window that passes the
    Lock Strategy V30 entry filters.  Settlement is determined from the last
    snapshot.
    """
    outcome = mkt.up or mkt.yes
    if outcome is None:
        return []
    outcome_id = outcome.outcome_id
    close_dt   = mkt.resolution_date
    if close_dt is None:
        return []

    close_ms = int(close_dt.timestamp() * 1000)
    open_ms  = close_ms - MARKET_DURATION * 1000
    strike   = _parse_strike(outcome_id) or _parse_strike(getattr(mkt, "title", "") or "")
    date_str = close_dt.strftime("%Y-%m-%d")
    slug     = outcome_id

    # Fetch full range orderbook once
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
    finally:
        time.sleep(rate_delay)

    if not all_books:
        return []

    # Settlement from last snapshot
    settle_ob  = all_books[-1]
    settle_bid = _best_bid(settle_ob)
    if settle_bid is None:
        return []
    if settle_bid >= SETTLED_THRESH:
        outcome_val = 1
    elif settle_bid <= 1.0 - SETTLED_THRESH:
        outcome_val = 0
    else:
        return []  # ambiguous settlement

    rows = []
    for elapsed_sec in ENTRY_SECONDS:
        target_ms  = open_ms + elapsed_sec * 1000
        t_left     = MARKET_DURATION - elapsed_sec   # e.g., 60..10 seconds

        # Find closest orderbook snapshot within ±5s of target time
        candidates = [
            (abs(b.timestamp - target_ms), b)
            for b in all_books
            if b.timestamp is not None and abs(b.timestamp - target_ms) <= 5000
        ]
        if not candidates:
            continue
        _, ob = min(candidates, key=lambda x: x[0])

        ask = _best_ask(ob)
        bid = _best_bid(ob)
        if ask is None or bid is None:
            continue
        if not (ASK_MIN <= ask <= ASK_MAX):
            continue
        spread = round(ask - bid, 6)
        if spread > SPREAD_MAX:
            continue

        # BTC spot price at this timestamp
        ts_sec    = ob.timestamp // 1000 if ob.timestamp else (open_ms + elapsed_sec * 1000) // 1000
        btc_spot  = btc_price_at(btc_klines, ts_sec)
        if btc_spot is None or strike is None:
            # Degrade gracefully: skip BTC-dependent filters but flag row
            d_norm    = None
            prob_bnc  = None
            prob_poly = None
            edge_poly = None
        else:
            d_norm    = distance_norm(btc_spot, strike, t_left, btc_vol)
            prob_bnc  = binance_prob(d_norm)
            # Polymarket proxy: same Bachelier model — represents market-implied
            # probability from an independent data source (Polymarket 15m BTC).
            # In live trading this is the actual Polymarket ask price.
            prob_poly = prob_bnc
            edge_poly = round(prob_poly - ask, 6)

        # Apply BTC-dependent consensus filters
        if d_norm is not None:
            if d_norm < DIST_NORM_MIN:
                continue
            if prob_bnc < PROB_BINANCE_MIN:
                continue
            if prob_poly < PROB_POLY_MIN:
                continue
            if edge_poly < EDGE_POLY_MIN:
                continue

        rows.append({
            "slug":         slug,
            "date":         date_str,
            "strike":       strike,
            "t_left":       t_left,
            "elapsed_sec":  elapsed_sec,
            "ask":          ask,
            "bid":          bid,
            "spread":       spread,
            "btc_spot":     btc_spot,
            "d_norm":       d_norm,
            "prob_bnc":     prob_bnc,
            "prob_poly":    prob_poly,
            "edge_poly":    edge_poly,
            "outcome":      outcome_val,
        })

    return rows


# ── PnL simulation ─────────────────────────────────────────────────────────────

def compute_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-trade PnL columns:
      entry_price = ask + SLIPPAGE
      contracts   = ORDER_SIZE / entry_price
      pnl_dollars = contracts × (outcome − entry_price)
    """
    df = df.copy()
    df["entry_price"] = (df["ask"] + SLIPPAGE).round(6)
    df["contracts"]   = ORDER_SIZE / df["entry_price"]
    df["pnl_dollars"] = df["contracts"] * (df["outcome"] - df["entry_price"])
    df["win"]         = (df["outcome"] == 1).astype(int)
    return df


# ── De-duplicate: one trade per market per day ─────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    When multiple ticks qualify within the T-left window for the same market,
    take the first qualifying tick (highest T-left = earliest entry, more time
    left on the contract → lower entry price, more upside).
    """
    return (
        df.sort_values("t_left", ascending=False)
          .drop_duplicates(subset=["slug"], keep="first")
          .reset_index(drop=True)
    )


# ── Statistics ─────────────────────────────────────────────────────────────────

def _daily_pnl(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["pnl_dollars"].sum().sort_index()


def print_stats(df: pd.DataFrame, days_covered: int, btc_vol: float) -> None:
    if len(df) == 0:
        print("\n  No trades generated under the current filters.")
        return

    total       = len(df)
    wins        = int(df["win"].sum())
    losses      = total - wins
    win_rate    = wins / total
    net_pnl     = df["pnl_dollars"].sum()
    roi_pct     = net_pnl / (total * ORDER_SIZE) * 100
    avg_entry   = df["entry_price"].mean() * 100   # in cents
    avg_t_left  = df["t_left"].mean()
    avg_spread  = df["spread"].mean() * 100         # in cents

    daily = _daily_pnl(df)
    neg_days  = int((daily < 0).sum())
    best_day  = daily.idxmax() if len(daily) else "—"
    best_pnl  = daily.max()    if len(daily) else 0
    worst_day = daily.idxmin() if len(daily) else "—"
    worst_pnl = daily.min()    if len(daily) else 0

    # Top day contributions
    sorted_d  = daily.sort_values(ascending=False)
    top1_pct  = sorted_d.iloc[0]  / net_pnl * 100 if len(sorted_d) >= 1 and net_pnl != 0 else 0
    top3_pct  = sorted_d.iloc[:3].sum() / net_pnl * 100 if len(sorted_d) >= 3 and net_pnl != 0 else 0

    # Daily score: win_rate × (net_pnl / days_covered)
    daily_score = win_rate * 100 * (net_pnl / max(days_covered, 1))

    # Train / valid / test split (60/20/20 by date order)
    dates      = sorted(df["date"].unique())
    n          = len(dates)
    train_end  = dates[int(n * 0.60) - 1] if n >= 5 else dates[-1]
    valid_end  = dates[int(n * 0.80) - 1] if n >= 5 else dates[-1]

    train_df   = df[df["date"] <= train_end]
    valid_df   = df[(df["date"] > train_end) & (df["date"] <= valid_end)]
    test_df    = df[df["date"] > valid_end]

    def _wr(sub): return sub["win"].mean() * 100 if len(sub) > 0 else float("nan")

    W = 62
    print(f"\n{'═'*W}")
    print(f"  Lock Strategy V30  ·  KXBTC15M (15-min BTC)  ·  MID_70_90_CONSENSUS_EDGE")
    print(f"  Order ${ORDER_SIZE:.0f}  ·  Slippage {SLIPPAGE*100:.0f}c  ·  BTC vol {btc_vol:.0%}")
    print(f"{'─'*W}")
    print(f"  Trades          {total:>6d}")
    print(f"  Win Rate        {win_rate*100:>6.2f}%")
    print(f"  Net PnL         ${net_pnl:>+8.2f}")
    print(f"  ROI             {roi_pct:>6.2f}%")
    print(f"  Avg Entry       {avg_entry:>6.2f}c")
    print(f"  Avg T-left      {avg_t_left:>6.1f}s")
    print(f"  Avg Spread      {avg_spread:>6.2f}c")
    print(f"{'─'*W}")
    print(f"  Train WR        {_wr(train_df):>6.2f}%   (60% of dates: up to {train_end})")
    print(f"  Valid WR        {_wr(valid_df):>6.2f}%   (next 20%: up to {valid_end})")
    print(f"  Test WR         {_wr(test_df):>6.2f}%   (final 20%)")
    print(f"{'─'*W}")
    print(f"  Daily Score     {daily_score:>6.2f}")
    print(f"  Top1 Day PnL    {top1_pct:>6.2f}%  of total")
    print(f"  Top3 Day PnL    {top3_pct:>6.2f}%  of total")
    print(f"  Negative Days   {neg_days:>6d}")
    print(f"  Best Day        {best_day}  /  ${best_pnl:+.2f}")
    print(f"  Worst Day       {worst_day}  /  ${worst_pnl:+.2f}")
    print(f"{'─'*W}")
    print(f"  Active Settings (15m-fitted)")
    print(f"    T-left min: {ENTRY_T_MIN}s  |  T-left max: {ENTRY_T_MAX}s")
    print(f"    Ask min: {ASK_MIN*100:.0f}c  |  Ask max: {ASK_MAX*100:.0f}c")
    print(f"    Spread max: {SPREAD_MAX*100:.0f}c")
    print(f"    Prob Polymarket min: {PROB_POLY_MIN*100:.0f}%")
    print(f"    Edge Polymarket min: {EDGE_POLY_MIN*100:.0f}c")
    print(f"    Prob Binance min: {PROB_BINANCE_MIN*100:.0f}%")
    print(f"    Distance norm Binance min: {DIST_NORM_MIN:.2f}")
    print(f"{'═'*W}\n")

    # Per-day breakdown
    print(f"  {'Date':<12}  {'Trades':>6}  {'WR%':>7}  {'PnL':>8}  {'Cum PnL':>9}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*9}")
    cum = 0.0
    for d in sorted(df["date"].unique()):
        sub  = df[df["date"] == d]
        w    = sub["win"].mean() * 100
        p    = sub["pnl_dollars"].sum()
        cum += p
        print(f"  {d:<12}  {len(sub):>6}  {w:>6.1f}%  ${p:>+7.2f}  ${cum:>+8.2f}")
    print()


# ── Parameter sweep ────────────────────────────────────────────────────────────

def _sweep_filters(df_raw: pd.DataFrame) -> None:
    """
    Sweep DIST_NORM_MIN and ASK_MAX to show win rate / trades / net PnL.
    df_raw must include all columns BEFORE filters are applied.
    """
    dist_thresholds = [0.75, 1.00, 1.10, 1.25, 1.40, 1.55, 1.70, 2.00]
    prob_thresholds = [0.80, 0.85, 0.90, 0.92, 0.95]

    print(f"\n{'═'*80}")
    print("  Sweep: Prob Binance min threshold × Distance norm min")
    print(f"{'─'*80}")
    header = f"  {'prob_bnc':>9}"
    for d in dist_thresholds:
        header += f"  {d:>7.2f}σ"
    print(header)
    print("  " + "─" * 78)

    for pb in prob_thresholds:
        row_str = f"  {pb:>8.0%}"
        for d in dist_thresholds:
            mask = (
                (df_raw["d_norm"].notna()) &
                (df_raw["d_norm"] >= d) &
                (df_raw["prob_bnc"] >= pb) &
                (df_raw["prob_bnc"] >= PROB_POLY_MIN) &
                (df_raw["edge_poly"] >= EDGE_POLY_MIN)
            )
            sub = df_raw[mask]
            if len(sub) < 5:
                row_str += f"  {'—':>9}"
            else:
                wr = sub["win"].mean() * 100
                row_str += f"  {wr:>6.1f}%/{len(sub):>2}"
        print(row_str)

    print(f"{'═'*80}")
    print("  format: WR%/n   (entries after deduplication not applied here)")

    # Ask range sweep
    ask_combos = [(0.70, 0.85), (0.70, 0.90), (0.72, 0.88), (0.75, 0.90), (0.70, 0.95)]
    print(f"\n{'═'*72}")
    print("  Sweep: Ask range × Distance norm min  (using prob_bnc ≥ 92%)")
    print(f"{'─'*72}")
    header2 = f"  {'ask range':>12}"
    for d in dist_thresholds:
        header2 += f"  {d:>7.2f}σ"
    print(header2)
    print("  " + "─" * 70)
    for amin, amax in ask_combos:
        row_str = f"  {amin*100:.0f}c–{amax*100:.0f}c     "
        for d in dist_thresholds:
            mask = (
                (df_raw["d_norm"].notna()) &
                (df_raw["ask"] >= amin) &
                (df_raw["ask"] <= amax) &
                (df_raw["d_norm"] >= d) &
                (df_raw["prob_bnc"] >= PROB_BINANCE_MIN) &
                (df_raw["edge_poly"] >= EDGE_POLY_MIN)
            )
            sub = df_raw[mask]
            if len(sub) < 5:
                row_str += f"  {'—':>9}"
            else:
                wr = sub["win"].mean() * 100
                row_str += f"  {wr:>6.1f}%/{len(sub):>2}"
        print(row_str)
    print(f"{'═'*72}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    api_key:    str | None,
    days:       int | None,
    cache_path: str | None,
    from_cache: str | None,
    rate_delay: float,
    btc_vol:    float,
    sweep:      bool,
    no_dedup:   bool,
) -> None:

    if from_cache:
        print(f"Loading cached data from {from_cache} …")
        df_raw = pd.read_csv(from_cache)
        # Restore correct dtypes
        for col in ("d_norm", "prob_bnc", "prob_poly", "edge_poly", "btc_spot"):
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        btc_klines_available = True
    else:
        import pmxt  # noqa: F401 — validated at runtime

        kalshi = _make_kalshi(api_key)

        print(f"Fetching KXBTC15M markets from pmxt …")
        markets = _fetch_all_markets(kalshi, days)
        print(f"Found {len(markets)} settled markets.\n")

        if not markets:
            print("No markets found. Try --days 180 or pass --api-key.")
            return

        # Determine date range for BTC price fetch
        close_times = [int(m.resolution_date.timestamp()) for m in markets if m.resolution_date]
        if not close_times:
            print("No resolution dates found.")
            return
        btc_start = min(close_times) - MARKET_DURATION - 120
        btc_end   = max(close_times) + 120

        print(f"Fetching Coinbase BTC/USD 1m candles "
              f"({datetime.fromtimestamp(btc_start, tz=timezone.utc).date()} → "
              f"{datetime.fromtimestamp(btc_end, tz=timezone.utc).date()}) …")
        btc_klines = fetch_btc_klines(btc_start, btc_end)
        print(f"  {len(btc_klines):,} 1m bars fetched.\n")

        btc_klines_available = bool(btc_klines)

        all_rows: list[dict] = []
        for i, mkt in enumerate(markets):
            outcome = mkt.up or mkt.yes
            ticker  = (outcome.outcome_id if outcome else None) or getattr(mkt, "ticker", "?")
            print(f"  [{i+1}/{len(markets)}] {ticker} …", end=" ", flush=True)
            rows = _sample_market(kalshi, mkt, btc_klines, btc_vol, rate_delay)
            all_rows.extend(rows)
            print(f"{len(rows)} qualifying ticks")

        df_raw = pd.DataFrame(all_rows)

        if cache_path and len(df_raw):
            df_raw.to_csv(cache_path, index=False)
            print(f"\nCached {len(df_raw)} rows → {cache_path}")

    if len(df_raw) == 0:
        print("No qualifying ticks found.")
        return

    # Add PnL columns (requires win column first)
    df_raw = compute_pnl(df_raw)

    if not btc_klines_available or df_raw["d_norm"].isna().all():
        print("\n[warn] No BTC price data available — distance/prob filters not applied.")

    # Sweep before dedup (shows raw filter sensitivity)
    if sweep and "d_norm" in df_raw.columns:
        _sweep_filters(df_raw)

    # Deduplicate: one entry per market (take earliest qualifying T-left)
    if no_dedup:
        df = df_raw.copy()
    else:
        df = deduplicate(df_raw)

    days_covered = df["date"].nunique() if len(df) else 1
    print_stats(df, days_covered, btc_vol)

    # Per-filter attrition summary (using raw rows before dedup)
    if "d_norm" in df_raw.columns:
        n_total   = len(df_raw)
        n_dn      = int((df_raw["d_norm"].notna() & (df_raw["d_norm"] >= DIST_NORM_MIN)).sum())
        n_pb      = int((df_raw["d_norm"].notna() &
                         (df_raw["d_norm"] >= DIST_NORM_MIN) &
                         (df_raw["prob_bnc"] >= PROB_BINANCE_MIN)).sum())
        n_pp      = int((df_raw["d_norm"].notna() &
                         (df_raw["d_norm"] >= DIST_NORM_MIN) &
                         (df_raw["prob_bnc"] >= PROB_BINANCE_MIN) &
                         (df_raw["prob_poly"] >= PROB_POLY_MIN)).sum())
        n_edge    = int((df_raw["d_norm"].notna() &
                         (df_raw["d_norm"] >= DIST_NORM_MIN) &
                         (df_raw["prob_bnc"] >= PROB_BINANCE_MIN) &
                         (df_raw["prob_poly"] >= PROB_POLY_MIN) &
                         (df_raw["edge_poly"] >= EDGE_POLY_MIN)).sum())
        print(f"  Filter attrition (raw ticks, no dedup):")
        print(f"    Ask[70–90c] ∩ Spread≤2c ∩ T-left[10–60s]    {n_total:>6}")
        print(f"    + Distance norm ≥ {DIST_NORM_MIN:.2f}σ                    {n_dn:>6}")
        print(f"    + Prob Binance ≥ {PROB_BINANCE_MIN:.0%}                   {n_pb:>6}")
        print(f"    + Prob Polymarket ≥ {PROB_POLY_MIN:.0%}                {n_pp:>6}")
        print(f"    + Edge ≥ {EDGE_POLY_MIN*100:.0f}c                             {n_edge:>6}")
        print(f"    After dedup (1 trade/market):                {len(df):>6}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lock Strategy V30 backtest — KXBTC15M (15-min BTC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api-key",    default=None,  help="pmxt API key")
    parser.add_argument("--days",       type=int, default=None,
                        help="Limit to last N days of settled markets")
    parser.add_argument("--cache",      default=None,  help="Save qualifying ticks to CSV")
    parser.add_argument("--from-cache", default=None,  help="Skip API fetch; load from CSV")
    parser.add_argument("--rate-delay", type=float, default=0.3,
                        help="Seconds between pmxt API calls (default 0.3)")
    parser.add_argument("--btc-vol",    type=float, default=0.65,
                        help="Annual BTC vol used in Bachelier model (default 0.65)")
    parser.add_argument("--sweep",      action="store_true",
                        help="Print parameter sensitivity sweep tables")
    parser.add_argument("--no-dedup",   action="store_true",
                        help="Count every qualifying tick instead of 1 trade per market")
    args = parser.parse_args()

    run(
        api_key    = args.api_key,
        days       = args.days,
        cache_path = args.cache,
        from_cache = args.from_cache,
        rate_delay = args.rate_delay,
        btc_vol    = args.btc_vol,
        sweep      = args.sweep,
        no_dedup   = args.no_dedup,
    )
