"""
evaluate.py — Evaluate the trained RL agent on held-out markets.

Reports:
  - Total trades, win rate, avg pnl/trade
  - Sharpe ratio of trade returns
  - Comparison: agent vs fixed scalp (60→65c, stop=5c) on same markets
  - Trade type breakdown: scalps (exited early) vs settlers (held to expiry)
  - Per-trade log (optional --verbose)

Usage:
  %PY% main.py rl-eval --asset BTC
  %PY% main.py rl-eval --asset BTC --model models/btc_scalp_settle
  %PY% main.py rl-eval --asset BTC --verbose
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(
    df: pd.DataFrame,
    asset: str,
    model_path: str = "models",
    val_frac: float = 0.2,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("  ERROR: stable-baselines3 not installed.")
        return

    from .environment import KalshiScalpEnv, load_markets, train_val_split

    # Find model file
    mdl_dir  = Path(model_path)
    mdl_file = mdl_dir / f"best_model"
    if not (mdl_file.with_suffix(".zip")).exists():
        mdl_file = mdl_dir / f"{asset.lower()}_scalp_settle"
    if not (mdl_file.with_suffix(".zip")).exists():
        print(f"  No model found in {model_path} — run rl-train first.")
        return

    print(f"\n  Loading model: {mdl_file}.zip")
    model = PPO.load(str(mdl_file))

    markets = load_markets(df)
    _, val_mkts = train_val_split(markets, val_frac=val_frac, seed=seed)
    print(f"  Evaluating on {len(val_mkts)} held-out markets...")

    env = KalshiScalpEnv(val_mkts, shuffle=False, seed=seed + 1)

    all_trades: list[dict] = []

    for mkt_idx, mkt in enumerate(val_mkts):
        obs, _ = env.reset()
        # Ensure we're on this specific market
        env._ticks       = mkt
        env._tick_idx    = 0
        env._in_position = False
        env._entry_ask   = 0.0
        env._episode_returns = []
        obs = env._obs()

        entry_tick  = None
        entry_price = None
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            tick_before = env._tick_idx
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            row = mkt.iloc[min(tick_before, len(mkt) - 1)]
            ask = float(row["ask"])

            if action == 1 and entry_price is None:
                entry_tick  = tick_before
                entry_price = ask

            if action == 2 and entry_price is not None:
                pnl_c = (ask - entry_price) * 100
                n_ticks_held = tick_before - entry_tick
                all_trades.append({
                    "market":     mkt_idx,
                    "type":       "scalp",
                    "entry_c":    round(entry_price * 100),
                    "exit_c":     round(ask * 100),
                    "pnl_c":      round(pnl_c, 1),
                    "ticks_held": n_ticks_held,
                    "outcome":    int(mkt["outcome"].iloc[-1]),
                })
                entry_price = None
                entry_tick  = None

            # Settlement at end
            if done and entry_price is not None:
                outcome      = int(mkt["outcome"].iloc[-1])
                settle_c     = 100 if outcome == 1 else 0
                pnl_c        = settle_c - entry_price * 100
                n_ticks_held = len(mkt) - entry_tick
                all_trades.append({
                    "market":     mkt_idx,
                    "type":       "settle",
                    "entry_c":    round(entry_price * 100),
                    "exit_c":     settle_c,
                    "pnl_c":      round(pnl_c, 1),
                    "ticks_held": n_ticks_held,
                    "outcome":    outcome,
                })
                entry_price = None

    env.close()
    _print_results(all_trades, asset, len(val_mkts), verbose)


def _print_results(
    trades: list[dict],
    asset: str,
    n_markets: int,
    verbose: bool,
) -> None:
    if not trades:
        print("  Agent made no trades on the validation set.")
        return

    df = pd.DataFrame(trades)
    n  = len(df)
    pnls = df["pnl_c"].values

    wins  = (pnls > 0).sum()
    losses = (pnls < 0).sum()
    mean  = float(np.mean(pnls))
    std   = float(np.std(pnls)) + 1e-8
    sharpe = mean / std * np.sqrt(n)   # annualised-style across trades

    scalps  = df[df["type"] == "scalp"]
    settles = df[df["type"] == "settle"]

    print(f"\n{'='*68}")
    print(f"  RL AGENT EVALUATION — {asset}   ({n_markets} val markets)")
    print(f"{'='*68}")
    print(f"  total trades     : {n}  ({n / n_markets:.2f} per market)")
    print(f"  win rate         : {wins/n*100:.1f}%  ({wins}W / {losses}L)")
    print(f"  avg pnl/trade    : {mean:+.2f}c")
    print(f"  total pnl        : {pnls.sum():+.1f}c  (${pnls.sum()/100:+.2f})")
    print(f"  sharpe (trades)  : {sharpe:+.3f}")
    print()
    print(f"  ── Scalp exits (sold early): {len(scalps)} trades ──")
    if len(scalps):
        sp = scalps["pnl_c"].values
        print(f"     avg pnl   : {np.mean(sp):+.2f}c")
        print(f"     win rate  : {(sp>0).mean()*100:.1f}%")
        print(f"     avg hold  : {scalps['ticks_held'].mean():.1f} ticks")
    print()
    print(f"  ── Settle (held to expiry): {len(settles)} trades ──")
    if len(settles):
        sp = settles["pnl_c"].values
        yes_settles = settles[settles["outcome"] == 1]
        no_settles  = settles[settles["outcome"] == 0]
        print(f"     avg pnl   : {np.mean(sp):+.2f}c")
        print(f"     YES rate  : {len(yes_settles)/len(settles)*100:.1f}%  "
              f"({len(yes_settles)} YES / {len(no_settles)} NO)")
        print(f"     avg entry : {settles['entry_c'].mean():.1f}c")
        print(f"     avg hold  : {settles['ticks_held'].mean():.1f} ticks")
    print()

    # Pnl distribution
    buckets = [-100, -20, -10, -5, 0, 5, 10, 20, 100]
    labels  = ["<-20c", "-20→-10c", "-10→-5c", "-5→0c", "0→5c", "5→10c", "10→20c", ">20c"]
    counts, _ = np.histogram(pnls, bins=buckets)
    print(f"  ── PnL distribution ──")
    for label, count in zip(labels, counts):
        bar = "█" * int(count / max(counts) * 20)
        print(f"     {label:>12} : {bar:<20} {count}")
    print()

    if verbose:
        print(f"  ── Per-trade log ──")
        print(f"  {'mkt':>4}  {'type':>6}  {'entry':>5}  {'exit':>5}  {'pnl':>7}  {'ticks':>6}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*7}  {'─'*6}")
        for _, row in df.iterrows():
            print(
                f"  {int(row['market']):>4}  {row['type']:>6}  "
                f"{int(row['entry_c']):>4}c  {int(row['exit_c']):>4}c  "
                f"{row['pnl_c']:>+6.1f}c  {int(row['ticks_held']):>6}"
            )
        print()
