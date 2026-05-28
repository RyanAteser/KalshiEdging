"""
train.py — Train the Kalshi scalp/settle RL agent using PPO.

The agent learns to:
  - BUY when contract price offers positive expected value
  - SELL early (scalp) when momentum stalls or profit is at risk
  - HOLD TO SETTLE when conviction is high and time remains
  - CUT early when trade is going wrong

Reward: Differential Sharpe Ratio (DSR) on each closed trade.

Usage:
  %PY% main.py rl-train --asset BTC
  %PY% main.py rl-train --asset BTC --timesteps 500000
  %PY% main.py rl-train --asset BTC --timesteps 1000000 --out models/btc_agent
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def train(
    df: pd.DataFrame,
    asset: str,
    timesteps: int = 300_000,
    out_path: str = "models",
    val_frac: float = 0.2,
    seed: int = 42,
    verbose: int = 1,
) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("  ERROR: stable-baselines3 not installed.")
        print("  Run:  pip install stable-baselines3 gymnasium")
        return

    from .environment import KalshiScalpEnv, load_markets, train_val_split

    print(f"\n  Loading markets for {asset}...")
    markets = load_markets(df)
    train_mkts, val_mkts = train_val_split(markets, val_frac=val_frac, seed=seed)
    print(f"  Train: {len(train_mkts)} markets  |  Val: {len(val_mkts)} markets")

    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / f"{asset.lower()}_scalp_settle"

    def make_train_env():
        return Monitor(KalshiScalpEnv(train_mkts, shuffle=True, seed=seed))

    def make_val_env():
        return Monitor(KalshiScalpEnv(val_mkts, shuffle=False, seed=seed + 1))

    train_env = make_vec_env(make_train_env, n_envs=4, seed=seed)
    val_env   = make_val_env()

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 512,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,      # encourage exploration
        policy_kwargs   = dict(net_arch=[128, 128, 64]),
        verbose         = verbose,
        seed            = seed,
        tensorboard_log = str(out_dir / "tb_logs"),
    )

    callbacks = [
        EvalCallback(
            val_env,
            best_model_save_path = str(out_dir),
            log_path             = str(out_dir / "eval_logs"),
            eval_freq            = max(1000, timesteps // 20),
            n_eval_episodes      = len(val_mkts),
            deterministic        = True,
            verbose              = verbose,
        ),
        CheckpointCallback(
            save_freq  = max(5000, timesteps // 10),
            save_path  = str(out_dir / "checkpoints"),
            name_prefix = f"{asset.lower()}_ckpt",
            verbose    = 0,
        ),
    ]

    print(f"\n  Training PPO — {timesteps:,} timesteps")
    print(f"  Model will be saved to: {model_file}")
    print(f"  TensorBoard: tensorboard --logdir {out_dir / 'tb_logs'}")
    print()

    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
    model.save(str(model_file))
    print(f"\n  Saved: {model_file}.zip")

    train_env.close()
    val_env.close()
