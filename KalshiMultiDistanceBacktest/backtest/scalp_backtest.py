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

Multi-zone mode (--zones):
  Runs multiple buy/sell/stop zones simultaneously on each market.
  Each zone tracks its own state independently.
  Zones are non-overlapping by design (e.g. 60:65, 75:80, 80:85).
  A zone exit does NOT block the adjacent zone from triggering independently.
  After exit from any zone, a 1-tick cooldown prevents immediate re-entry on
  that same zone (avoids double-counting the exit tick as a new entry).

Sweep modes:
  --sweep-buy    : fix spread, sweep buy_c from 5c to 95c in spread_c steps
  --sweep-spread : fix buy_c, sweep spread from 1c to 30c

Usage (via main.py):
  %PY% main.py scalp --asset BTC --buy 60 --spread 5
  %PY% main.py scalp --asset BTC --buy 60 --spread 5 --stop 10
  %PY% main.py scalp --asset BTC --spread 5 --sweep-buy
  %PY% main.py scalp --asset BTC --buy 60 --sweep-spread
  %PY% main.py scalp --asset ETH --spread 5 --sweep-buy --stop 10
  %PY% main.py scalp --asset BTC --zones 60:65,75:80,80:85 --stop 5
  %PY% main.py scalp --asset BTC --zones 60:65,75:80,80:85 --stop 5 --contracts 1
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


# ── Multi-zone simulation ──────────────────────────────────────────────────

def parse_zones(zones_str: str, stop_c: int, tol_c: int) -> list[dict]:
    """
    Parse '--zones 60:65,75:80,80:85' into zone config dicts.
    Each zone: {buy, sell, stop, tol}  (all in [0,1] floats)
    stop_c applies to all zones; pass 0 for no stop.
    """
    zones = []
    for part in zones_str.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"Zone must be 'buy:sell', got: {part!r}")
        b, s = part.split(":", 1)
        buy_c  = int(b)
        sell_c = int(s)
        spread_c = sell_c - buy_c
        if spread_c <= 0:
            raise ValueError(f"sell must be > buy in zone {part!r}")
        zones.append({
            "buy_c":    buy_c,
            "sell_c":   sell_c,
            "stop_c":   stop_c,
            "spread_c": spread_c,
            "buy":   buy_c  / 100.0,
            "sell":  sell_c / 100.0,
            "stop":  (buy_c - stop_c) / 100.0 if stop_c > 0 else -1.0,
            "tol":   tol_c  / 100.0,
        })
    return zones


def _simulate_market_multizone(
    ticks: pd.DataFrame,
    zones: list[dict],
) -> list[dict]:
    """
    Simulate all zones simultaneously on one market.
    Each zone has independent state. 1-tick cooldown after any exit.
    Returns flat list of trade dicts, each tagged with zone index.
    """
    outcome = int(ticks["outcome"].iloc[-1])
    asks    = ticks["ask"].values

    # Per-zone state
    n = len(zones)
    in_trade    = [False] * n
    entry_ask   = [0.0]   * n
    cooldown    = [0]     * n   # ticks remaining before zone can re-enter

    all_trades: list[dict] = []

    for ask in asks:
        if ask != ask:  # NaN check
            continue

        for i, z in enumerate(zones):
            if cooldown[i] > 0:
                cooldown[i] -= 1
                continue

            if not in_trade[i]:
                if abs(ask - z["buy"]) <= z["tol"]:
                    in_trade[i]  = True
                    entry_ask[i] = ask
            else:
                if ask >= z["sell"]:
                    all_trades.append({
                        "zone": i, "result": "hit",
                        "entry": entry_ask[i],
                        "buy_c": z["buy_c"], "sell_c": z["sell_c"], "stop_c": z["stop_c"],
                    })
                    in_trade[i] = False
                    cooldown[i] = 1
                elif z["stop"] >= 0 and ask <= z["stop"]:
                    all_trades.append({
                        "zone": i, "result": "stopped",
                        "entry": entry_ask[i],
                        "buy_c": z["buy_c"], "sell_c": z["sell_c"], "stop_c": z["stop_c"],
                    })
                    in_trade[i] = False
                    cooldown[i] = 1

    # Any zones still open at settlement
    for i, z in enumerate(zones):
        if in_trade[i]:
            all_trades.append({
                "zone": i, "result": "settled", "outcome": outcome,
                "entry": entry_ask[i],
                "buy_c": z["buy_c"], "sell_c": z["sell_c"], "stop_c": z["stop_c"],
            })

    return all_trades


def _pnl_multitrade(t: dict) -> float:
    """PnL in cents for a multi-zone trade dict."""
    if t["result"] == "hit":
        return t["sell_c"] - t["buy_c"]
    elif t["result"] == "stopped":
        return -t["stop_c"]
    else:
        o = t.get("outcome", 0)
        buy_c = t["buy_c"]
        return o * (100 - buy_c) + (1 - o) * (-buy_c)


def run_multizone_backtest(
    df: pd.DataFrame,
    zones: list[dict],
    contracts: int = 1,
) -> dict:
    """
    Run all zones simultaneously across every market.
    Returns a summary dict + per-zone breakdown.
    """
    all_trades:          list[dict] = []
    market_trade_counts: list[int]  = []
    markets_with_entry = 0

    for _, mkt in df.groupby("ticker"):
        mkt    = mkt.sort_values("tick_time").reset_index(drop=True)
        trades = _simulate_market_multizone(mkt, zones)
        if trades:
            markets_with_entry += 1
            market_trade_counts.append(len(trades))
            all_trades.extend(trades)

    n = len(all_trades)
    if n == 0:
        return {"total_trades": 0, "markets": 0, "trades_per_mkt": 0.0,
                "exp_pnl_c": 0.0, "total_pnl_c": 0.0, "per_zone": []}

    pnls   = [_pnl_multitrade(t) * contracts for t in all_trades]
    hits   = sum(1 for t in all_trades if t["result"] == "hit")
    stops  = sum(1 for t in all_trades if t["result"] == "stopped")
    settld = sum(1 for t in all_trades if t["result"] == "settled")
    avg_per_mkt = sum(market_trade_counts) / len(market_trade_counts)

    # Per-zone breakdown
    per_zone = []
    for i, z in enumerate(zones):
        zt = [t for t in all_trades if t["zone"] == i]
        if not zt:
            per_zone.append(None)
            continue
        zp   = [_pnl_multitrade(t) * contracts for t in zt]
        zh   = sum(1 for t in zt if t["result"] == "hit")
        zs   = sum(1 for t in zt if t["result"] == "stopped")
        per_zone.append({
            "buy_c": z["buy_c"], "sell_c": z["sell_c"], "stop_c": z["stop_c"],
            "trades":    len(zt),
            "hit%":      round(zh / len(zt) * 100, 1),
            "stop%":     round(zs / len(zt) * 100, 1),
            "exp_pnl_c": round(sum(zp) / len(zt), 2),
            "total_pnl_c": round(sum(zp), 1),
        })

    return {
        "total_trades":   n,
        "markets":        markets_with_entry,
        "trades_per_mkt": round(avg_per_mkt, 2),
        "hit%":           round(hits   / n * 100, 1),
        "stop%":          round(stops  / n * 100, 1),
        "settled%":       round(settld / n * 100, 1),
        "exp_pnl_c":      round(sum(pnls) / n, 2),
        "total_pnl_c":    round(sum(pnls), 1),
        "contracts":      contracts,
        "per_zone":       per_zone,
    }


def print_multizone_results(r: dict, asset: str, zones: list[dict]) -> None:
    if r["total_trades"] == 0:
        print("  No trades triggered across any zone.")
        return

    contracts = r["contracts"]
    zone_str  = "  ".join(f"{z['buy_c']}c→{z['sell_c']}c" for z in zones)
    stop_c    = zones[0]["stop_c"]
    stop_str  = f"stop={stop_c}c" if stop_c else "no stop"
    dollar_pnl = r["total_pnl_c"] / 100 * contracts

    print(f"\n{'='*72}")
    print(f"  MULTI-ZONE SCALP — {asset}   {stop_str}   contracts={contracts}")
    print(f"  Zones: {zone_str}")
    print(f"{'='*72}")
    print(f"  total trades    : {r['total_trades']:,}  across {r['markets']} markets")
    print(f"  trades / market : {r['trades_per_mkt']:.2f}  ← combined across all zones")
    print(f"  hit%            : {r['hit%']:.1f}%")
    print(f"  stop%           : {r['stop%']:.1f}%")
    print(f"  settled%        : {r['settled%']:.1f}%")
    print(f"  exp_pnl         : {r['exp_pnl_c']:+.2f}c / trade")
    print(f"  total pnl       : {r['total_pnl_c']:+.1f}c  (${dollar_pnl:+.2f})")
    print()
    print(f"  Per-zone breakdown:")
    print(f"  {'zone':>10}  {'trades':>7}  {'hit%':>6}  {'stop%':>6}  {'exp_pnl_c':>10}  {'total_pnl':>10}")
    print(f"  {'─'*10}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*10}")
    for z_summary in r["per_zone"]:
        if z_summary is None:
            continue
        flag = " ◀" if z_summary["exp_pnl_c"] > 0 else ""
        print(
            f"  {z_summary['buy_c']:>3}c→{z_summary['sell_c']:>3}c  "
            f"{z_summary['trades']:>7,}  "
            f"{z_summary['hit%']:>5.1f}%  {z_summary['stop%']:>5.1f}%  "
            f"{z_summary['exp_pnl_c']:>+9.2f}c  "
            f"{z_summary['total_pnl_c']:>+9.1f}c{flag}"
        )

    # $1 live trading guide
    max_concurrent = 1  # zones are sequential by price, 1 active at a time
    margin_needed  = max(z["buy_c"] for z in zones) / 100 * contracts
    risk_per_trade = stop_c / 100 * contracts if stop_c else "N/A"
    reward_per_trade = min(z["sell_c"] - z["buy_c"] for z in zones) / 100 * contracts

    print()
    print(f"  $1 live trading guide ({contracts} contract/zone):")
    print(f"  Max capital at risk at once : ${margin_needed:.2f}  (worst case = 1 trade open)")
    if stop_c:
        print(f"  Risk per trade              : ${risk_per_trade:.2f}")
        print(f"  Reward per trade            : ${reward_per_trade:.2f}")
        print(f"  R:R ratio                   : 1:{reward_per_trade / risk_per_trade:.1f}")
    print()


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
