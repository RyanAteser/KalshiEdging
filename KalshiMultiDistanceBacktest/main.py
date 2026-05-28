#!/usr/bin/env python3
"""
Multi-asset exact distance backtest for Kalshi 15-minute markets.

Usage:
  python main.py fetch              # fetch prices + Kalshi ticks for all assets
  python main.py build              # build per-asset datasets
  python main.py backtest           # run distance backtest for all assets
  python main.py ladder             # price-ladder bounce analysis (all assets)
  python main.py fetch --asset ETH  # single asset
  python main.py build --asset SOL
  python main.py ladder --asset BTC --step 5   # 5-cent increments
  python main.py ladder --asset ETH --sweep --stop 20   # sweep steps 1-30, stop=20c
  python main.py ladder --asset ETH --step 10 --stop 10 --from-below 10  # momentum entry
  python main.py ladder --asset ETH --sweep --stop 10 --from-below 10    # sweep w/ momentum
  python main.py backtest --asset XRP
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
                df, max_step_c=step_c, stop_loss_c=stop_loss_c, from_below_c=from_below_c
            )
            print_sweep_results(results, name, stop_loss_c, from_below_c)
        else:
            results = run_ladder_backtest(
                df, step_c=step_c, stop_loss_c=stop_loss_c, from_below_c=from_below_c
            )
            print_ladder_results(results, name, step_c, stop_loss_c, from_below_c)


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

    if cmd == "fetch":
        cmd_fetch(asset)
    elif cmd == "build":
        cmd_build(asset)
    elif cmd == "backtest":
        cmd_backtest(asset)
    elif cmd == "ladder":
        cmd_ladder(asset, step_c=step_c, stop_loss_c=stop_loss_c, sweep=sweep, from_below_c=from_below_c)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
