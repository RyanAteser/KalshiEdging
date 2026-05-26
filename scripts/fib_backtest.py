#!/usr/bin/env python3
"""
fib_backtest.py — Fibonacci proximity analysis for DIST trades.

Tests whether DIST entries made when BTC is near a 1-minute Fibonacci
retracement level have worse characteristics (closer BTC moves toward
strike, thinner margin) than entries made away from Fib levels.

Setup:
  - For each DIST trade, fetches 120 x 1-minute Coinbase candles ending
    at entry time (2-hour lookback window)
  - Finds swing high / swing low in that window
  - Computes Fibonacci levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
  - Checks if BTC price at entry is within $10 / $20 / $30 of any level
  - Reports win/loss breakdown and adverse-move stats for near vs far entries

Usage:
    python scripts/fib_backtest.py
    python scripts/fib_backtest.py --buffers 10 20 30 50

Output:
    Console table + fib_backtest_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

# ── Coinbase Exchange public candles ─────────────────────────────────
COINBASE_CANDLES_URL = (
    "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    "?granularity=60&start={start}&end={end}"
)
LOOKBACK_SECONDS = 7200   # 2 hours of 1-minute candles
FIB_RATIOS       = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
FIB_LABELS       = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]
REQUEST_PAUSE    = 0.4    # seconds between Coinbase API calls (rate limit)

# ── Embedded trade ledger ─────────────────────────────────────────────
# Columns: id, side, entry_price, strike, t_left_s, entry_ts, btc_close, pnl
# strike = Kalshi market strike (the threshold the YES/NO bet is about)
# BTC entry price is fetched live from Coinbase candle at entry_ts
DIST_TRADES = [
    (61,  "NO",  0.9420, 76447, 392,  "2026-05-26 16:53:27", 76428, 7.1340),
    (60,  "YES", 0.9360, 76430, 465,  "2026-05-26 16:37:15", 76436, 5.0740),
    (59,  "YES", 0.9230, 76245, 391,  "2026-05-26 16:23:29", 76428, 8.3930),
    (58,  "NO",  0.9700, 76423, 226,  "2026-05-26 16:11:14", 76229, 3.1800),
    (56,  "NO",  0.9010, 76460, 476,  "2026-05-26 15:52:04", 76418, 9.6030),
    (55,  "NO",  0.8900, 76577, 580,  "2026-05-26 15:35:20", 76428, 9.5700),
    (54,  "NO",  0.9780, 76849, 429,  "2026-05-26 15:22:51", 76543, 1.8700),
    (53,  "YES", 0.9220, 76751, 498,  "2026-05-26 14:51:42", 76816, 6.1620),
    (52,  "NO",  0.9490, 77889, 694,  "2026-05-26 14:33:26", 76709, 3.8250),
    (51,  "YES", 0.9460, 77282, 523,  "2026-05-26 14:21:17", 77907, 3.8340),
    (50,  "YES", 0.9660, 76813, 643,  "2026-05-26 13:49:17", 77121, 2.3460),
    (49,  "NO",  0.9530, 77007, 485,  "2026-05-26 13:21:55", 76712, 3.1020),
    (48,  "NO",  0.9650, 77119, 376,  "2026-05-26 12:08:44", 76965, 2.2400),
    (47,  "YES", 0.9610, 77128, 329,  "2026-05-26 10:39:31", 77420, 2.3790),
    (46,  "YES", 0.9830, 76628, 496,  "2026-05-26 10:21:44", 77139, 1.0200),
    (44,  "NO",  0.9930, 76826, 293,  "2026-05-26 07:10:06", 76592, 0.4200),
    (43,  "YES", 0.9930, 76754, 282,  "2026-05-26 06:40:18", 76887, 0.4130),
    (41,  "YES", 0.9860, 76464, 346,  "2026-05-26 03:39:14", 76564, 0.8260),
    (35,  "NO",  0.9780, 76730, 292,  "2026-05-26 01:40:07", 76583, 1.2540),
    (30,  "NO",  0.9330, 77219, 659,  "2026-05-25 22:34:01", 77113, 3.6180),
    (29,  "NO",  0.9950, 77379, 229,  "2026-05-25 20:41:11", 77191, 0.2650),
    (28,  "NO",  0.9960, 77445, 312,  "2026-05-25 19:39:48", 77340, 0.2120),
    (26,  "YES", 0.9900, 77499, 274,  "2026-05-25 14:55:26", 77629, 0.5300),
    (25,  "YES", 0.9270, 77265, 475,  "2026-05-25 14:07:04", 77396, 3.5770),
    (24,  "YES", 0.9700, 77188, 473,  "2026-05-25 06:37:06", 77326, 1.4400),
    (23,  "NO",  0.9870, 77284, 277,  "2026-05-25 06:25:22", 77186, 0.6110),
    (22,  "YES", 0.9790, 77152, 361,  "2026-05-25 04:53:58", 77292, 0.9660),
    (21,  "NO",  0.9730, 77308, 434,  "2026-05-25 02:22:45", 77027, 1.2150),
    (20,  "YES", 0.9000, 76850, 666,  "2026-05-25 02:03:53", 77314, 4.1000),
    (19,  "NO",  0.9950, 77078,  85,  "2026-05-25 01:58:34", 76842, 0.2050),
    (18,  "YES", 0.9940, 76785, 216,  "2026-05-24 23:41:23", 77114, 0.2400),
    (17,  "NO",  0.9790, 76794, 264,  "2026-05-24 22:40:35", 76681, 0.8400),
    (16,  "YES", 0.9260, 76049, 787,  "2026-05-24 22:01:52", 76867, 2.7380),
    (15,  "NO",  0.9880, 76475, 512,  "2026-05-24 21:36:27", 76051, 0.4320),
    (14,  "YES", 0.9710, 76287, 499,  "2026-05-24 16:06:40", 76510, 1.0150),
    (13,  "NO",  0.9830, 76893, 316,  "2026-05-24 13:54:43", 76683, 0.5950),
    (11,  "YES", 0.9890, 76714, 464,  "2026-05-24 06:52:15", 76890, 0.3740),
    (10,  "YES", 0.9840, 76566, 504,  "2026-05-24 05:51:35", 76722, 0.5440),
    ( 9,  "NO",  0.9990, 76818, 128,  "2026-05-24 03:42:51", 76658, 0.0340),
    ( 8,  "NO",  0.9840, 76823, 350,  "2026-05-24 02:24:09", 76645, 0.7200),
    ( 6,  "YES", 0.9660, 76488, 417,  "2026-05-23 23:08:02", 76590, 1.5300),
    ( 5,  "NO",  0.9540, 76753, 273,  "2026-05-23 21:40:26", 76574, 2.0700),
    ( 4,  "NO",  0.8700, 77157, 642,  "2026-05-23 21:04:17", 76859, 5.8500),
    ( 3,  "YES", 0.9890, 76743, 198,  "2026-05-23 20:56:41", 77182, 0.4400),
    ( 2,  "YES", 0.9500, 75930, 704,  "2026-05-23 20:33:15", 76739, 2.0000),
    ( 1,  "YES", 0.9560, 75626, 500,  "2026-05-23 18:21:40", 75738, 1.7160),
]

# ── Coinbase API ──────────────────────────────────────────────────────

def fetch_candles(entry_ts: str) -> Optional[list]:
    """Fetch 120 one-minute candles ending at entry_ts from Coinbase Exchange.

    Returns list of (time, low, high, open, close, volume) sorted oldest-first.
    Returns None on error.
    """
    dt    = datetime.strptime(entry_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end   = int(dt.timestamp())
    start = end - LOOKBACK_SECONDS

    url = COINBASE_CANDLES_URL.format(start=start, end=end)
    try:
        req  = urllib.request.Request(url, headers={"User-Agent": "KalshiEdging/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        raw  = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for {entry_ts}")
        return None
    except Exception as e:
        print(f"  Fetch error for {entry_ts}: {e}")
        return None

    if not isinstance(raw, list) or not raw:
        return None

    # Coinbase returns newest-first; sort oldest-first
    raw.sort(key=lambda c: c[0])
    return raw   # [[time, low, high, open, close, volume], ...]

# ── Fibonacci computation ─────────────────────────────────────────────

def compute_fib_levels(candles: list, side: str) -> tuple[float, float, list[float]]:
    """Return (swing_high, swing_low, [fib_level, ...]) from candle list."""
    highs = [c[2] for c in candles]
    lows  = [c[1] for c in candles]
    swing_high = max(highs)
    swing_low  = min(lows)
    rng = swing_high - swing_low

    if rng < 1.0:
        # Flat market — no meaningful Fib structure
        return swing_high, swing_low, []

    if side == "YES":
        # BTC above strike (bearish retracement): levels drawn from HIGH down
        levels = [swing_high - rng * r for r in FIB_RATIOS]
    else:
        # BTC below strike (bullish retracement): levels drawn from LOW up
        levels = [swing_low + rng * r for r in FIB_RATIOS]

    return swing_high, swing_low, levels


def nearest_fib_distance(btc_price: float, fib_levels: list[float]) -> Optional[float]:
    """Return distance in dollars to the nearest Fibonacci level."""
    if not fib_levels:
        return None
    return min(abs(btc_price - lvl) for lvl in fib_levels)


def adverse_move(side: str, btc_entry: float, btc_close: float) -> float:
    """How far did BTC move in the direction AGAINST the position?
    YES = BTC needs to stay above strike, adverse = price falling.
    NO  = BTC needs to stay below strike, adverse = price rising.
    Returns a positive value (move size) or 0 if move was favourable.
    """
    if side == "YES":
        move = btc_entry - btc_close   # positive if BTC fell
    else:
        move = btc_close - btc_entry   # positive if BTC rose
    return max(move, 0.0)


# ── Main ──────────────────────────────────────────────────────────────

def run(buffers: list[int]) -> None:
    results = []
    total   = len(DIST_TRADES)

    print(f"\nFib Backtest — {total} DIST trades\n{'─'*60}")

    for i, (tid, side, entry_px, strike, t_left, entry_ts, btc_close, pnl) in enumerate(DIST_TRADES):
        print(f"  [{i+1:2d}/{total}] Trade #{tid}  {entry_ts}  {side}", end="  ", flush=True)

        candles = fetch_candles(entry_ts)
        time.sleep(REQUEST_PAUSE)

        if candles is None or len(candles) < 20:
            print("SKIP (no candle data)")
            continue

        btc_entry  = float(candles[-1][4])   # close of last 1m candle = BTC at entry
        swing_high, swing_low, fib_levels = compute_fib_levels(candles, side)
        fib_rng    = swing_high - swing_low
        nearest    = nearest_fib_distance(btc_entry, fib_levels)
        adv_move   = adverse_move(side, btc_entry, btc_close)
        margin     = abs(btc_entry - strike)   # BTC distance from market strike at entry

        near_flags = {buf: (nearest is not None and nearest <= buf) for buf in buffers}

        print(
            f"BTC={btc_entry:.0f}  range={fib_rng:.0f}  "
            f"nearest_fib=${nearest:.1f}  adv_move=${adv_move:.0f}"
        )

        results.append({
            "id":          tid,
            "side":        side,
            "entry_px":    entry_px,
            "strike":      strike,
            "t_left":      t_left,
            "entry_ts":    entry_ts,
            "btc_entry":   btc_entry,
            "btc_close":   btc_close,
            "pnl":         pnl,
            "swing_high":  swing_high,
            "swing_low":   swing_low,
            "fib_range":   fib_rng,
            "nearest_fib": nearest,
            "adverse_move": adv_move,
            "margin":      margin,
            **{f"near_{b}": near_flags[b] for b in buffers},
        })

    if not results:
        print("\nNo results — check API connectivity.")
        return

    # ── Summary stats ─────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  RESULTS — {len(results)} trades analyzed")
    print(f"{'═'*70}\n")

    def _stats(subset: list[dict], label: str) -> None:
        if not subset:
            print(f"  {label}: — (no data)")
            return
        n         = len(subset)
        wins      = sum(1 for r in subset if r["pnl"] > 0)
        avg_adv   = sum(r["adverse_move"] for r in subset) / n
        max_adv   = max(r["adverse_move"] for r in subset)
        avg_marg  = sum(r["margin"] for r in subset) / n
        avg_near  = (sum(r["nearest_fib"] for r in subset if r["nearest_fib"]) /
                     sum(1 for r in subset if r["nearest_fib"]) if any(r["nearest_fib"] for r in subset) else 0)
        win_rate  = wins / n * 100
        print(
            f"  {label:35s} n={n:3d}  win={win_rate:5.1f}%  "
            f"avg_adv=${avg_adv:6.1f}  max_adv=${max_adv:6.1f}  "
            f"avg_margin=${avg_marg:6.1f}  avg_nearest_fib=${avg_near:5.1f}"
        )

    print(f"  {'Category':<35} {'n':>4}  {'win%':>6}  {'avg_adv':>9}  {'max_adv':>9}  {'avg_margin':>11}  {'avg_near_fib':>13}")
    print(f"  {'─'*35}─{'─'*4}──{'─'*6}──{'─'*9}──{'─'*9}──{'─'*11}──{'─'*13}")

    _stats(results, "ALL DIST trades")
    print()

    for buf in buffers:
        near = [r for r in results if r[f"near_{buf}"]]
        far  = [r for r in results if not r[f"near_{buf}"]]
        pct_near = len(near) / len(results) * 100 if results else 0
        print(f"  ── Buffer ${buf} ({pct_near:.1f}% of entries are near-Fib) ──")
        _stats(near, f"  NEAR Fib (≤${buf})")
        _stats(far,  f"  FAR  Fib (>${buf})")
        print()

    # ── Recommendation ────────────────────────────────────────────────
    print(f"{'═'*70}")
    print("  INTERPRETATION GUIDE")
    print(f"{'─'*70}")
    print("  If NEAR-Fib trades show higher avg_adv → BTC moves more against")
    print("  the position when entered near a Fib level → filter those out.")
    print()
    print("  Best buffer = largest $ where (far_win% ≥ near_win% + 2pp)")
    print("                AND filtered-out trades < 40% of total volume.")
    print()

    # Auto-recommend best buffer
    best_buf = None
    best_improvement = 0.0
    for buf in buffers:
        near = [r for r in results if r[f"near_{buf}"]]
        far  = [r for r in results if not r[f"near_{buf}"]]
        if len(near) < 3 or len(far) < 3:
            continue
        near_win = sum(1 for r in near if r["pnl"] > 0) / len(near)
        far_win  = sum(1 for r in far  if r["pnl"] > 0) / len(far)
        near_adv = sum(r["adverse_move"] for r in near) / len(near)
        far_adv  = sum(r["adverse_move"] for r in far) / len(far)
        filtered_pct = len(near) / len(results)
        improvement = (far_adv - near_adv)   # positive = far trades are safer
        if improvement > best_improvement and filtered_pct <= 0.40:
            best_improvement = improvement
            best_buf = buf

    if best_buf:
        near_n = sum(1 for r in results if r[f"near_{best_buf}"])
        print(f"  → Recommended buffer: ${best_buf}  "
              f"(filters {near_n}/{len(results)} = {near_n/len(results)*100:.0f}% of entries)")
    else:
        print("  → No clear buffer advantage found in this sample.")
    print()

    # ── Write CSV ─────────────────────────────────────────────────────
    csv_path = "fib_backtest_results.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results written to {csv_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fibonacci proximity backtest for DIST trades")
    parser.add_argument(
        "--buffers", type=int, nargs="+", default=[10, 20, 30],
        help="Dollar buffers to test (default: 10 20 30)"
    )
    args = parser.parse_args()
    run(args.buffers)
