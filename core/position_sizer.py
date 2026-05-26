"""
position_sizer.py — Dollar-stake cycle with velocity guard.

Stake cycle ($35 → $100):
  Each consecutive win adds STAKE_WIN_STEP to the active stake (capped at STAKE_MAX).
  Any loss resets to STAKE_MIN immediately.

  Example progression:
    Start:     $35 stake
    1 win  →   $45
    2 wins →   $55  ...
    7 wins →  $100  (capped)
    1 loss →   $35  (reset)

Velocity filter:
  Blocks new entries if MAX_ENTRIES_PER_WINDOW confirmed fills occurred in the
  last VELOCITY_WINDOW_SEC seconds.  Prevents over-trading during rapid market
  moves. Call note_entry() after every confirmed fill.

Shutdown: after MAX_CONSECUTIVE_LOSSES losses in a row.

Kalshi fee formula:
  trading_fee    = ceil(0.07 × contracts × price × (1−price) × 100) / 100
  regulatory_fee = ceil(0.0035 × contracts × price × 100) / 100
  total_cost     = contracts × price + trading_fee + regulatory_fee
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Callable, Optional

from db.db import Database

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_LOSSES  = 2
SAFETY_FACTOR           = 0.95   # use only 95% of balance — covers fees + slippage

STAKE_MIN               = 35.0   # starting dollar stake per trade
STAKE_MAX               = 100.0  # maximum dollar stake per trade
STAKE_WIN_STEP          = 10.0   # stake increment per consecutive win

MAX_ENTRIES_PER_WINDOW  = 3      # velocity: max confirmed entries allowed
VELOCITY_WINDOW_SEC     = 1800   # ... within this rolling window (30 min)


class PositionSizer:

    def __init__(
            self,
            db: Database,
            on_shutdown: Optional[Callable[[], None]] = None,
    ) -> None:
        self._db          = db
        self._on_shutdown = on_shutdown
        self._lock        = threading.Lock()

        self._consecutive_wins   = 0
        self._consecutive_losses = 0
        self._total_trades       = 0
        self._total_wins         = 0
        self._shutdown_fired     = False

        self._current_stake: float = STAKE_MIN
        self._entry_times: deque   = deque()   # timestamps of confirmed fills

        self._load_state_from_db()

    # ── Kalshi fee model ──────────────────────────────────────────────

    @staticmethod
    def _kalshi_cost(contracts: int, price: float) -> float:
        if contracts <= 0 or price <= 0:
            return 0.0

        base = contracts * price

        trading_fee = math.ceil(
            0.07 * contracts * price * (1.0 - price) * 100
        ) / 100.0

        regulatory_fee = math.ceil(
            0.0035 * contracts * price * 100
        ) / 100.0

        return base + trading_fee + regulatory_fee

    # ── Velocity helper (call inside self._lock) ──────────────────────

    def _is_velocity_blocked(self) -> bool:
        cutoff = time.time() - VELOCITY_WINDOW_SEC
        # Evict stale timestamps while we're in here
        while self._entry_times and self._entry_times[0] < cutoff:
            self._entry_times.popleft()
        return len(self._entry_times) >= MAX_ENTRIES_PER_WINDOW

    # ── Public API ────────────────────────────────────────────────────

    def get_qty(self, cash: float, ask_price: float) -> int:
        """Return max contracts affordable within the current dollar stake.

        Returns 0 when:
          - consecutive-loss circuit breaker is active
          - velocity limit is reached
          - cash or price invalid
        """
        with self._lock:
            if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                logger.warning(
                    "Sizer: BLOCKED — %d consecutive losses (limit=%d). "
                    "Restart the bot to reset.",
                    self._consecutive_losses, MAX_CONSECUTIVE_LOSSES,
                )
                return 0

            if self._is_velocity_blocked():
                recent = len(self._entry_times)
                logger.warning(
                    "Sizer: VELOCITY BLOCKED — %d entries in last %.0fm (max=%d)",
                    recent, VELOCITY_WINDOW_SEC / 60, MAX_ENTRIES_PER_WINDOW,
                )
                return 0

            stake = self._current_stake

        if ask_price <= 0 or cash <= 0:
            return 0

        # Cap stake at available cash
        safe_cash = cash * SAFETY_FACTOR
        stake     = min(stake, safe_cash)
        if stake <= 0:
            return 0

        naive_qty = int(math.floor(stake / ask_price))
        if naive_qty < 1:
            return 0

        qty = naive_qty
        while qty > 0:
            cost = self._kalshi_cost(qty, ask_price)
            if cost <= stake:
                break
            qty -= 1

        cost = self._kalshi_cost(qty, ask_price) if qty > 0 else 0.0
        logger.info(
            "Sizer: stake=$%.2f  cash=$%.2f  price=%.4f  qty=%d  est_cost=$%.4f",
            stake, cash, ask_price, qty, cost,
        )
        return max(qty, 0)

    def note_entry(self) -> None:
        """Call after every confirmed fill to track velocity."""
        with self._lock:
            self._entry_times.append(time.time())
        logger.debug("Sizer: entry noted (window count=%d)", len(self._entry_times))

    def record_result(self, pnl: float) -> None:
        with self._lock:
            self._total_trades += 1

            if pnl > 0:
                self._total_wins       += 1
                self._consecutive_wins  += 1
                self._consecutive_losses = 0
                prev_stake = self._current_stake
                self._current_stake = min(
                    self._current_stake + STAKE_WIN_STEP, STAKE_MAX
                )
                logger.info(
                    "WIN  streak=%d  total=%d/%d  stake $%.0f → $%.0f",
                    self._consecutive_wins, self._total_wins, self._total_trades,
                    prev_stake, self._current_stake,
                )
            else:
                self._consecutive_losses += 1
                self._consecutive_wins    = 0
                prev_stake = self._current_stake
                self._current_stake = STAKE_MIN
                logger.warning(
                    "LOSS  consecutive=%d/%d  stake reset $%.0f → $%.0f",
                    self._consecutive_losses, MAX_CONSECUTIVE_LOSSES,
                    prev_stake, self._current_stake,
                )

                if (self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES
                        and not self._shutdown_fired
                        and self._on_shutdown):
                    self._shutdown_fired = True
                    logger.critical(
                        "SHUTDOWN: %d consecutive losses — stopping bot",
                        self._consecutive_losses,
                    )
                    self._on_shutdown()

    @property
    def current_stake(self) -> float:
        with self._lock:
            return self._current_stake

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_trades":       self._total_trades,
                "total_wins":         self._total_wins,
                "consecutive_wins":   self._consecutive_wins,
                "consecutive_losses": self._consecutive_losses,
                "current_stake":      self._current_stake,
                "win_rate":           self._total_wins / self._total_trades
                if self._total_trades else 0,
            }

    # ── DB bootstrap ──────────────────────────────────────────────────

    def _load_state_from_db(self) -> None:
        try:
            sql = """
                SELECT pnl FROM trades
                WHERE side = 'SELL' AND pnl IS NOT NULL
                ORDER BY id DESC LIMIT 50
            """
            rows = self._db.fetchall(sql)
        except Exception as exc:
            logger.warning("PositionSizer: could not load trade history: %s", exc)
            return

        if not rows:
            logger.info("PositionSizer: no history — fresh start at stake=$%.0f", STAKE_MIN)
            return

        trades = [float(r[0]) for r in rows]
        self._total_trades = len(trades)
        self._total_wins   = sum(1 for p in trades if p > 0)

        # Stake and consecutive streaks reset each session — don't carry
        # forward losses from a previous run into a fresh start.
        logger.info(
            "PositionSizer: loaded wins=%d/%d  starting stake=$%.0f",
            self._total_wins, self._total_trades, self._current_stake,
        )
