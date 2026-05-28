"""
environment.py — Gym environment for Kalshi 15-minute scalp/settle agent.

Each episode = one 15-minute market replayed tick by tick.

State (10 features, all normalised [-1, 1] or [0, 1]):
  0  ask               contract ask price              [0, 1]
  1  mid               (ask+bid)/2                     [0, 1]
  2  time_remaining    1.0 at open → 0.0 at expiry     [0, 1]
  3  btc_dist          (btc - strike) / strike         [-1, 1]  (clipped ±0.1)
  4  ask_mom_fast      ask change over last 3 ticks     norm
  5  ask_mom_slow      ask change over last 10 ticks    norm
  6  btc_mom           btc change over last 10 ticks    norm
  7  gbm_prob          GBM-implied YES probability      [0, 1]
  8  in_position       0 or 1
  9  unrealized_pnl    (current_ask - entry_ask) if long, else 0   norm

Actions:
  0  HOLD   — do nothing
  1  BUY    — enter long at current ask (ignored if already in position)
  2  SELL   — exit at current ask (ignored if not in position)

Reward: DSR update on each closed trade.
        Step reward = 0 while no trade closes.
        At episode end, open positions are settled at binary outcome (0 or 100c).

Termination: last tick of the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .dsr import DSR


class KalshiScalpEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        markets: list[pd.DataFrame],   # list of per-market tick DataFrames
        dsr_eta: float = 0.01,
        max_t_left: float = 900.0,     # 15 minutes in seconds
        shuffle: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.markets    = markets
        self.dsr        = DSR(eta=dsr_eta)
        self.max_t_left = max_t_left
        self.shuffle    = shuffle
        self._rng       = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell

        self._market_idx  = 0
        self._order       = list(range(len(markets)))
        self._tick_idx    = 0
        self._ticks: pd.DataFrame | None = None
        self._in_position = False
        self._entry_ask   = 0.0
        self._episode_returns: list[float] = []

    # ── Gym interface ──────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Advance to next market
        if self._market_idx == 0 and self.shuffle:
            self._order = self._rng.permutation(len(self.markets)).tolist()

        self._ticks       = self.markets[self._order[self._market_idx]]
        self._market_idx  = (self._market_idx + 1) % len(self.markets)
        self._tick_idx    = 0
        self._in_position = False
        self._entry_ask   = 0.0
        self._episode_returns = []

        return self._obs(), {}

    def step(self, action: int):
        tick     = self._current_tick()
        ask      = float(tick["ask"])
        outcome  = int(tick["outcome"])
        is_last  = self._tick_idx >= len(self._ticks) - 1

        reward = 0.0

        # ── Execute action ─────────────────────────────────────────────
        if action == 1 and not self._in_position:       # BUY
            self._in_position = True
            self._entry_ask   = ask

        elif action == 2 and self._in_position:         # SELL early
            pnl = ask - self._entry_ask                 # in [0,1] units
            reward = self.dsr.update(pnl)
            self._episode_returns.append(pnl)
            self._in_position = False
            self._entry_ask   = 0.0

        # ── Forced settlement at last tick ────────────────────────────
        if is_last and self._in_position:
            settle_price = 1.0 if outcome == 1 else 0.0
            pnl = settle_price - self._entry_ask
            reward = self.dsr.update(pnl)
            self._episode_returns.append(pnl)
            self._in_position = False
            self._entry_ask   = 0.0

        self._tick_idx += 1
        terminated = is_last
        truncated  = False

        return self._obs(), reward, terminated, truncated, self._info()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _current_tick(self) -> pd.Series:
        idx = min(self._tick_idx, len(self._ticks) - 1)
        return self._ticks.iloc[idx]

    def _obs(self) -> np.ndarray:
        t   = self._ticks
        idx = min(self._tick_idx, len(t) - 1)
        row = t.iloc[idx]

        ask = float(row.get("ask", 0.5))
        bid = float(row.get("bid", ask))
        mid = (ask + bid) / 2.0

        t_left  = float(row.get("t_left", 0.0))
        t_norm  = float(np.clip(t_left / self.max_t_left, 0.0, 1.0))

        btc    = float(row.get("binance_price", 0.0))
        strike = float(row.get("strike", btc or 1.0))
        dist   = float(np.clip((btc - strike) / max(strike, 1e-6), -0.1, 0.1)) / 0.1

        # Momentum: ask changes over windows
        asks_all = t["ask"].values
        def _mom(window: int) -> float:
            start = max(0, idx - window)
            if idx <= start:
                return 0.0
            return float(asks_all[idx] - asks_all[start]) / 0.1

        ask_mom_fast = float(np.clip(_mom(3),  -1.0, 1.0))
        ask_mom_slow = float(np.clip(_mom(10), -1.0, 1.0))

        btc_arr = t["binance_price"].values
        def _btc_mom(window: int) -> float:
            start = max(0, idx - window)
            ref   = btc_arr[start]
            if ref <= 0:
                return 0.0
            return float(np.clip((btc_arr[idx] - ref) / ref / 0.005, -1.0, 1.0))

        btc_mom = _btc_mom(10)

        gbm_prob = float(np.clip(row.get("prob", 0.5), 0.0, 1.0))

        in_pos     = 1.0 if self._in_position else 0.0
        unreal_pnl = float(np.clip((ask - self._entry_ask) / 0.1, -1.0, 1.0)) \
                     if self._in_position else 0.0

        return np.array([
            ask,
            mid,
            t_norm,
            dist,
            ask_mom_fast,
            ask_mom_slow,
            btc_mom,
            gbm_prob,
            in_pos,
            unreal_pnl,
        ], dtype=np.float32)

    def _info(self) -> dict:
        returns = self._episode_returns
        if len(returns) == 0:
            return {"n_trades": 0, "total_pnl_c": 0.0, "sharpe": 0.0}
        pnls_c = [r * 100 for r in returns]
        mean   = np.mean(pnls_c)
        std    = np.std(pnls_c) + 1e-8
        return {
            "n_trades":    len(returns),
            "total_pnl_c": round(sum(pnls_c), 2),
            "sharpe":      round(mean / std, 3),
        }


# ── Dataset loader ─────────────────────────────────────────────────────────

def load_markets(df: pd.DataFrame, min_ticks: int = 5) -> list[pd.DataFrame]:
    """Split a dataset parquet into per-market DataFrames, sorted by time."""
    markets = []
    for _, mkt in df.groupby("ticker"):
        mkt = mkt.sort_values("tick_time").reset_index(drop=True)
        if len(mkt) >= min_ticks:
            markets.append(mkt)
    return markets


def train_val_split(
    markets: list[pd.DataFrame],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Chronological split — val set is the most recent val_frac of markets."""
    n_val = max(1, int(len(markets) * val_frac))
    return markets[:-n_val], markets[-n_val:]
