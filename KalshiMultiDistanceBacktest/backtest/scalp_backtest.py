"""
scalp_backtest.py — Fixed-spread scalp backtest for Kalshi 15-minute markets.

Strategy: "buy at X cents, sell at X+spread cents" — unlimited re-entries per market.

Key differences from ladder_backtest:
  - Multiple trades per market (re-enters immediately after each exit)
  - Fixed buy_c and sell_c rather than per-tier stepping
  - Reports trades_per_market to validate frequency assumptions
  - No momentum filter — pure price-level scalp

Trade lifecycle per market:
  ENTER  when ask crosses INTO [buy_c - tol, buy_c + tol]  (first tick in zone)
  EXIT   when ask >= sell_c  → WIN  (+spread_c cents)
         when ask <= stop_c  → LOSS (-stop_c cents)  [if stop enabled]
         market settles      → pnl = outcome * (100 - buy_c) - (1 - outcome) * buy_c
                               i.e. +YES_pnl or -buy_c depending on resolution

Sweep modes:
  --sweep-buy    : fix spread, sweep buy_c from 5c to 95c in spread_c steps
  --sweep-spread : fix buy_c, sweep spread from 1c to 30c

Usage (via main.py):
  python main.py scalp --asset BTC --buy 60 --spread 5
  python main.py scalp --asset BTC --buy 60 --spread 5 --stop 10
  python main.py scalp --asset BTC --spread 5 --sweep-buy
  python main.py scalp --asset BTC --buy 60 --sweep-spread
  python main.py scalp --asset ETH --spread 5 --sweep-buy --stop 10
"""

from __future__ import annotations

import pandas as pd


# ── Core per-market simulation ─────────────────────────────────────────────

def _simulate_market_scalp(
    ticks: pd.DataFrame,    # sorted by tick_time, ask in [0,1]
    buy: float,             # entry price in [0,1]
    sell: float,            # target exit price in [0,1]
    stop: float,            # stop price in [0,1], or -1 if no stop
    tol: float,             # entry tolerance
) -> list[dict]:
    """Simulate unlimited-reentry scalp on one market. Returns list of trade dicts."""
    trades = []
    in_trade = False
    entry_ask = 0.0
    outcome = int(ticks["outcome"].iloc[-1])

    for ask in ticks["ask"].values:
        if pd.isna(ask):
            continue

        if not in_trade:
            # Enter when ask is in [buy-tol, buy+tol]
            if abs(ask - buy) <= tol:
                in_trade = True
                entry_ask = ask
        else:
            # Check exit conditions
            if ask >= sell:
                trades.append({"result": "hit", "entry": entry_ask})
                in_trade = False
            elif stop >= 0 and ask <= stop:
                trades.append({"result": "stopped", "entry": entry_ask})
                in_trade = False

    # Open trade at market settlement
    if in_trade:
        trades.append({"result": "settled", "entry": entry_ask, "outcome": outcome})

    return trades


def _pnl_for_trade(t: dict, sell_c: int, stop_c: int) -> float:
    """Return pnl in cents for a trade dict."""
    result = t["result"]
    entry_c = round(t["entry"] * 100)
    spread_c = sell_c - entry_c
    if result == "hit":
        return spread_c
    elif result == "stopped":
        return -stop_c
    else:  # settled
        o = t.get("outcome", 0)
        return o * (100 - entry_c) + (1 - o) * (-entry_c)


# ── Single configuration backtest ─────────────────────────────────────────

def run_scalp_backtest(
    df: pd.DataFrame,
    buy_c: int,
    sell_c: int,
    stop_c: int = 0,
    tol_c: int = 2,
) -> dict:
    """
    Run scalp for a single buy/sell/stop combo across all markets.
    Returns a summary dict.
    """
    buy  = buy_c  / 100.0
    sell = sell_c / 100.0
    stop = (buy_c - stop_c) / 100.0 if stop_c > 0 else -1.0
    tol  = tol_c  / 100.0

    all_trades: list[dict] = []
    market_trade_counts: list[int] = []
    markets_with_entry = 0

    for _, mkt in df.groupby("ticker"):
        mkt = mkt.sort_values("tick_time").reset_index(drop=True)
        trades = _simulate_market_scalp(mkt, buy, sell, stop, tol)
        if trades:
            markets_with_entry += 1
            market_trade_counts.append(len(trades))
            all_trades.extend(trades)

    n = len(all_trades)
    if n == 0:
        return {
            "buy_c": buy_c, "sell_c": sell_c, "stop_c": stop_c,
            "total_trades": 0, "markets": 0,
            "trades_per_mkt": 0.0,
            "hit%": 0.0, "stop%": 0.0, "settled%": 0.0,
            "exp_pnl_c": 0.0, "total_pnl_c": 0.0,
        }

    pnls   = [_pnl_for_trade(t, sell_c, stop_c) for t in all_trades]
    hits   = sum(1 for t in all_trades if t["result"] == "hit")
    stops  = sum(1 for t in all_trades if t["result"] == "stopped")
    settld = sum(1 for t in all_trades if t["result"] == "settled")

    avg_per_mkt = sum(market_trade_counts) / len(market_trade_counts) if market_trade_counts else 0.0
    exp_pnl     = sum(pnls) / n

    return {
        "buy_c":          buy_c,
        "sell_c":         sell_c,
        "stop_c":         stop_c,
        "total_trades":   n,
        "markets":        markets_with_entry,
        "trades_per_mkt": round(avg_per_mkt, 2),
        "hit%":           round(hits   / n * 100, 1),
        "stop%":          round(stops  / n * 100, 1),
        "settled%":       round(settld / n * 100, 1),
        "exp_pnl_c":      round(exp_pnl, 2),
        "total_pnl_c":    round(sum(pnls), 1),
    }


# ── Sweep: vary buy_c, fixed spread ───────────────────────────────────────

def run_sweep_buy(
    df: pd.DataFrame,
    spread_c: int,
    stop_c: int = 0,
    tol_c: int = 2,
    min_buy_c: int = 5,
    max_buy_c: int = 90,
) -> pd.DataFrame:
    """Sweep buy_c from min_buy_c to max_buy_c (step = spread_c), fixed spread."""
    rows = []
    step = max(1, spread_c)
    for buy_c in range(min_buy_c, max_buy_c + 1, step):
        sell_c = buy_c + spread_c
        if sell_c > 99:
            break
        r = run_scalp_backtest(df, buy_c, sell_c, stop_c=stop_c, tol_c=tol_c)
        rows.append(r)
    return pd.DataFrame(rows)


# ── Sweep: fixed buy_c, vary spread ───────────────────────────────────────

def run_sweep_spread(
    df: pd.DataFrame,
    buy_c: int,
    stop_c: int = 0,
    tol_c: int = 2,
    max_spread_c: int = 30,
) -> pd.DataFrame:
    """Sweep spread from 1c to max_spread_c, fixed buy_c."""
    rows = []
    for spread_c in range(1, max_spread_c + 1):
        sell_c = buy_c + spread_c
        if sell_c > 99:
            break
        r = run_scalp_backtest(df, buy_c, sell_c, stop_c=stop_c, tol_c=tol_c)
        rows.append(r)
    return pd.DataFrame(rows)


# ── Printing ───────────────────────────────────────────────────────────────

def print_scalp_single(r: dict) -> None:
    stop_str = f" | stop {r['stop_c']}c below entry" if r["stop_c"] else " | no stop"
    print(f"\n  buy={r['buy_c']}c → sell={r['sell_c']}c{stop_str}")
    print(f"  total trades : {r['total_trades']:,}  across {r['markets']} markets")
    print(f"  trades/market: {r['trades_per_mkt']:.2f}")
    print(f"  hit%   : {r['hit%']:.1f}%")
    print(f"  stop%  : {r['stop%']:.1f}%")
    print(f"  settled: {r['settled%']:.1f}%")
    print(f"  exp_pnl: {r['exp_pnl_c']:+.2f}c/trade")
    print(f"  total  : {r['total_pnl_c']:+.1f}c")
    print()


def print_sweep_buy(
    results: pd.DataFrame,
    asset: str,
    spread_c: int,
    stop_c: int,
) -> None:
    if results.empty:
        print("  No data.")
        return
    stop_str = f"stop={stop_c}c" if stop_c else "no stop"
    print(f"\n{'='*92}")
    print(f"  SCALP SWEEP (buy price) — {asset}   spread={spread_c}c  {stop_str}")
    print(f"  Re-entries allowed | exp_pnl_c = avg cents per trade | t/mkt = trades per market")
    print(f"{'='*92}")
    print(
        f"  {'buy':>4}  {'sell':>4}  {'trades':>7}  {'mkts':>5}  {'t/mkt':>6}  "
        f"{'hit%':>6}  {'stop%':>6}  {'setl%':>6}  {'exp_pnl_c':>10}  {'total_pnl':>10}"
    )
    print(f"  {'─'*4}  {'─'*4}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*10}")
    for _, row in results.iterrows():
        flag = " ◀" if row["exp_pnl_c"] > 0 else ""
        print(
            f"  {int(row['buy_c']):>3}c  {int(row['sell_c']):>3}c  "
            f"{int(row['total_trades']):>7,}  {int(row['markets']):>5}  "
            f"{row['trades_per_mkt']:>6.2f}  "
            f"{row['hit%']:>5.1f}%  {row['stop%']:>5.1f}%  {row['settled%']:>5.1f}%  "
            f"{row['exp_pnl_c']:>+9.2f}c  {row['total_pnl_c']:>+9.1f}c{flag}"
        )
    print()


def print_sweep_spread(
    results: pd.DataFrame,
    asset: str,
    buy_c: int,
    stop_c: int,
) -> None:
    if results.empty:
        print("  No data.")
        return
    stop_str = f"stop={stop_c}c" if stop_c else "no stop"
    print(f"\n{'='*92}")
    print(f"  SCALP SWEEP (spread) — {asset}   buy={buy_c}c  {stop_str}")
    print(f"  Re-entries allowed | exp_pnl_c = avg cents per trade | t/mkt = trades per market")
    print(f"{'='*92}")
    print(
        f"  {'sprd':>4}  {'sell':>4}  {'trades':>7}  {'mkts':>5}  {'t/mkt':>6}  "
        f"{'hit%':>6}  {'stop%':>6}  {'setl%':>6}  {'exp_pnl_c':>10}  {'total_pnl':>10}"
    )
    print(f"  {'─'*4}  {'─'*4}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*10}")
    for _, row in results.iterrows():
        spread_c = int(row["sell_c"]) - int(row["buy_c"])
        flag = " ◀" if row["exp_pnl_c"] > 0 else ""
        print(
            f"  {spread_c:>3}c  {int(row['sell_c']):>3}c  "
            f"{int(row['total_trades']):>7,}  {int(row['markets']):>5}  "
            f"{row['trades_per_mkt']:>6.2f}  "
            f"{row['hit%']:>5.1f}%  {row['stop%']:>5.1f}%  {row['settled%']:>5.1f}%  "
            f"{row['exp_pnl_c']:>+9.2f}c  {row['total_pnl_c']:>+9.1f}c{flag}"
        )
    print()
