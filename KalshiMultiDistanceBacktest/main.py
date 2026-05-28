#!/usr/bin/env python3
"""
Multi-asset exact distance backtest for Kalshi 15-minute markets.

Usage:
  %PY% main.py fetch              # fetch prices + Kalshi ticks for all assets
  %PY% main.py build              # build per-asset datasets
  %PY% main.py backtest           # run distance backtest for all assets
  %PY% main.py ladder             # price-ladder bounce analysis (all assets)
  %PY% main.py scalp              # fixed-spread scalp backtest (multi-entry per market)
  %PY% main.py fetch --asset ETH  # single asset
  %PY% main.py build --asset SOL
  %PY% main.py ladder --asset BTC --step 5   # 5-cent increments
  %PY% main.py ladder --asset ETH --sweep --stop 20   # sweep steps 1-30, stop=20c
  %PY% main.py ladder --asset ETH --step 10 --stop 10 --from-below 10  # momentum entry
  %PY% main.py ladder --asset ETH --sweep --stop 10 --from-below 10    # sweep w/ momentum
  %PY% main.py backtest --asset XRP

  # Scalp examples (buy at X, sell at X+spread, re-enter after each exit)
  %PY% main.py scalp --asset BTC --buy 60 --spread 5          # single config
  %PY% main.py scalp --asset BTC --buy 60 --spread 5 --stop 10
  %PY% main.py scalp --asset BTC --spread 5 --sweep-buy        # sweep all buy prices
  %PY% main.py scalp --asset ETH --buy 60 --sweep-spread       # sweep all spreads
  %PY% main.py scalp --asset BTC --spread 5 --sweep-buy --stop 10
"""

import sys
from pathlib import Path

# Allow imports of sibling packages
sys.path.insert(0, str(Path(__file__).parent))

from config import ASSETS, ENABLED_ASSETS, DAYS


def cmd_fetch(asset_filter=None):
    from data.fetch_prices import save_prices
    from data.fetch_kalshi import save_kalshi_series

    assets = [asset_filter] if asset_filter else ENABLED_ASSETS
    for name in assets:
        if name not in ASSETS:
            print(f"  Unknown asset: {name}  (choices: {list(ASSETS)})")
            continue
        cfg = ASSETS[name]
        print(f"\n=== Fetching price data: {name} ({cfg['coinbase_pair']}) ===")
        save_prices(cfg, days=DAYS)
        print(f"\n=== Fetching Kalshi ticks: {cfg['kalshi_series']} ===")
        save_kalshi_series(cfg, days=DAYS)


def cmd_build(asset_filter=None):
    from data.build_dataset import build_asset

    assets = [asset_filter] if asset_filter else ENABLED_ASSETS
    for name in assets:
        if name not in ASSETS:
            print(f"  Unknown asset: {name}")
            continue
        print(f"\n=== Building dataset: {name} ===")
        try:
            build_asset(name, ASSETS[name])
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")


def cmd_backtest(asset_filter=None):
    import pandas as pd
    from backtest.distance_backtest import run_distance_backtest, print_results

    assets = [asset_filter] if asset_filter else ENABLED_ASSETS
    summary_rows = []

    for name in assets:
        if name not in ASSETS:
            print(f"  Unknown asset: {name}")
            continue
        cfg     = ASSETS[name]
        ds_path = Path("data") / f"dataset_{name.lower()}.parquet"

        if not ds_path.exists():
            print(f"  {name}: dataset not found — run 'python main.py build --asset {name}' first")
            continue

        df = pd.read_parquet(ds_path)
        print(f"\n{'='*60}")
        print(f"  {name}  ({cfg['kalshi_series']})")
        print(f"  entry thresholds: {cfg['thresholds']}")
        print(f"  stop_dist: {cfg['stop_dist']}")
        print(f"{'='*60}")

        results = run_distance_backtest(df, cfg["thresholds"], cfg["stop_dist"])
        print_results(results)

        if not results.empty and results["trades"].sum() > 0:
            from backtest.distance_backtest import _best_row
            best = results.loc[_best_row(results)]
            summary_rows.append({
                "asset":      name,
                "best_dist":  best["entry_dist"],
                "trades":     int(best["trades"]),
                "win_rate":   f"{best['win_rate']:.1f}%",
                "stop_rate":  f"{best['stop_rate']:.1f}%",
                "avg_pnl":    round(best["avg_pnl"], 5),
                "total_pnl":  round(best["total_pnl"], 4),
            })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        print("\n" + "=" * 70)
        print("  CROSS-ASSET SUMMARY  (best threshold per asset)")
        print("=" * 70)
        print(summary.to_string(index=False))
        print()


def cmd_ladder(
    asset_filter=None,
    step_c: int = 10,
    stop_loss_c: int = 0,
    sweep: bool = False,
    from_below_c: int = 0,
    from_above_c: int = 0,
):
    import pandas as pd
    from pathlib import Path
    from backtest.ladder_backtest import (
        run_ladder_backtest, run_ladder_sweep,
        print_ladder_results, print_sweep_results,
    )

    assets = [asset_filter] if asset_filter else ENABLED_ASSETS
    for name in assets:
        if name not in ASSETS:
            print(f"  Unknown asset: {name}")
            continue
        ds_path = Path("data") / f"dataset_{name.lower()}.parquet"
        if not ds_path.exists():
            print(f"  {name}: dataset not found — run build first")
            continue
        df = pd.read_parquet(ds_path)
        print(f"\n{name} ({ASSETS[name]['kalshi_series']}) — "
              f"{len(df):,} ticks, {df['ticker'].nunique()} markets")
        if sweep:
            results = run_ladder_sweep(
                df, max_step_c=step_c, stop_loss_c=stop_loss_c,
                from_below_c=from_below_c, from_above_c=from_above_c,
            )
            print_sweep_results(results, name, stop_loss_c, from_below_c, from_above_c)
        else:
            results = run_ladder_backtest(
                df, step_c=step_c, stop_loss_c=stop_loss_c,
                from_below_c=from_below_c, from_above_c=from_above_c,
            )
            print_ladder_results(results, name, step_c, stop_loss_c, from_below_c, from_above_c)


def cmd_scalp(
    asset_filter=None,
    buy_c: int = 60,
    spread_c: int = 5,
    stop_c: int = 0,
    sweep_buy: bool = False,
    sweep_spread: bool = False,
    zones_str: str = "",
    contracts: int = 1,
    tol_c: int = 2,
):
    import pandas as pd
    from pathlib import Path
    from backtest.scalp_backtest import (
        run_scalp_backtest, run_sweep_buy, run_sweep_spread,
        run_multizone_backtest, parse_zones,
        print_scalp_single, print_sweep_buy, print_sweep_spread,
        print_multizone_results,
    )

    assets = [asset_filter] if asset_filter else ENABLED_ASSETS
    for name in assets:
        if name not in ASSETS:
            print(f"  Unknown asset: {name}")
            continue
        ds_path = Path("data") / f"dataset_{name.lower()}.parquet"
        if not ds_path.exists():
            print(f"  {name}: dataset not found — run build first")
            continue
        df = pd.read_parquet(ds_path)
        print(f"\n{name} ({ASSETS[name]['kalshi_series']}) — "
              f"{len(df):,} ticks, {df['ticker'].nunique()} markets")

        if zones_str:
            zones = parse_zones(zones_str, stop_c=stop_c, tol_c=tol_c)
            r = run_multizone_backtest(df, zones=zones, contracts=contracts)
            print_multizone_results(r, name, zones)
        elif sweep_buy:
            results = run_sweep_buy(df, spread_c=spread_c, stop_c=stop_c, tol_c=tol_c)
            print_sweep_buy(results, name, spread_c, stop_c)
        elif sweep_spread:
            results = run_sweep_spread(df, buy_c=buy_c, stop_c=stop_c, tol_c=tol_c)
            print_sweep_spread(results, name, buy_c, stop_c)
        else:
            sell_c = buy_c + spread_c
            r = run_scalp_backtest(df, buy_c=buy_c, sell_c=sell_c, stop_c=stop_c, tol_c=tol_c)
            print_scalp_single(r)


def main():
    args  = sys.argv[1:]
    cmd   = args[0] if args else "backtest"

    # Parse --asset ETH  or  --asset=ETH
    asset = None
    for i, a in enumerate(args):
        if a.startswith("--asset="):
            asset = a.split("=", 1)[1].upper()
            break
        if a == "--asset" and i + 1 < len(args):
            asset = args[i + 1].upper()
            break

    # Parse --sweep
    sweep = "--sweep" in args

    # Parse --step 5  or  --step=5  (when --sweep, means max_step_c; default 30)
    step_c = 30 if sweep else 10
    for i, a in enumerate(args):
        if a.startswith("--step="):
            step_c = int(a.split("=", 1)[1])
            break
        if a == "--step" and i + 1 < len(args):
            step_c = int(args[i + 1])
            break

    # Parse --stop N  or  --stop=N
    stop_loss_c = 0
    for i, a in enumerate(args):
        if a.startswith("--stop="):
            stop_loss_c = int(a.split("=", 1)[1])
            break
        if a == "--stop" and i + 1 < len(args):
            stop_loss_c = int(args[i + 1])
            break

    # Parse --from-below N  or  --from-below=N
    from_below_c = 0
    for i, a in enumerate(args):
        if a.startswith("--from-below="):
            from_below_c = int(a.split("=", 1)[1])
            break
        if a == "--from-below" and i + 1 < len(args):
            from_below_c = int(args[i + 1])
            break

    # Parse --from-above N  or  --from-above=N
    from_above_c = 0
    for i, a in enumerate(args):
        if a.startswith("--from-above="):
            from_above_c = int(a.split("=", 1)[1])
            break
        if a == "--from-above" and i + 1 < len(args):
            from_above_c = int(args[i + 1])
            break

    # Parse --buy N
    buy_c = 60
    for i, a in enumerate(args):
        if a.startswith("--buy="):
            buy_c = int(a.split("=", 1)[1]); break
        if a == "--buy" and i + 1 < len(args):
            buy_c = int(args[i + 1]); break

    # Parse --spread N
    spread_c = 5
    for i, a in enumerate(args):
        if a.startswith("--spread="):
            spread_c = int(a.split("=", 1)[1]); break
        if a == "--spread" and i + 1 < len(args):
            spread_c = int(args[i + 1]); break

    # Parse --tol N  (entry tolerance in cents, default 2)
    tol_c = 2
    for i, a in enumerate(args):
        if a.startswith("--tol="):
            tol_c = int(a.split("=", 1)[1]); break
        if a == "--tol" and i + 1 < len(args):
            tol_c = int(args[i + 1]); break

    sweep_buy    = "--sweep-buy"    in args
    sweep_spread = "--sweep-spread" in args

    # Parse --zones 60:65,75:80,80:85
    zones_str = ""
    for i, a in enumerate(args):
        if a.startswith("--zones="):
            zones_str = a.split("=", 1)[1]; break
        if a == "--zones" and i + 1 < len(args):
            zones_str = args[i + 1]; break

    # Parse --contracts N  (number of contracts per zone, default 1)
    contracts = 1
    for i, a in enumerate(args):
        if a.startswith("--contracts="):
            contracts = int(a.split("=", 1)[1]); break
        if a == "--contracts" and i + 1 < len(args):
            contracts = int(args[i + 1]); break

    if cmd == "fetch":
        cmd_fetch(asset)
    elif cmd == "build":
        cmd_build(asset)
    elif cmd == "backtest":
        cmd_backtest(asset)
    elif cmd == "ladder":
        cmd_ladder(
            asset, step_c=step_c, stop_loss_c=stop_loss_c,
            sweep=sweep, from_below_c=from_below_c, from_above_c=from_above_c,
        )
    elif cmd == "scalp":
        cmd_scalp(
            asset, buy_c=buy_c, spread_c=spread_c, stop_c=stop_loss_c,
            sweep_buy=sweep_buy, sweep_spread=sweep_spread,
            zones_str=zones_str, contracts=contracts, tol_c=tol_c,
        )
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
