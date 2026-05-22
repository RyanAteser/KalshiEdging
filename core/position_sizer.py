"""
position_sizer.py — Cash-based position sizing with accurate Kalshi fee math.

Kalshi fee formula (as of 2025):
  trading_fee    = ceil(0.07 * contracts * price * (1 - price) * 100) / 100
  regulatory_fee = ceil(0.0035 * contracts * price * 100) / 100
  total_cost     = contracts * price + trading_fee + regulatory_fee

We solve for max `contracts` where total_cost <= cash × SAFETY_FACTOR.

Shutdown: after MAX_CONSECUTIVE_LOSSES losses in a row.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Callable, Optional

from db.db import Database

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_LOSSES = 2
SAFETY_FACTOR          = 0.95   # use only 95% of balance — covers fees + slippage


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

        self._load_state_from_db()

    # ── Kalshi fee model ──────────────────────────────────────────────

    @staticmethod
    def _kalshi_cost(contracts: int, price: float) -> float:
        """
        Total cost (in dollars) including Kalshi trading + regulatory fees.
        """
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

    # ── Public API ────────────────────────────────────────────────────

    def get_qty(self, cash: float, ask_price: float) -> int:
        """Return max contracts that fit in available cash, fees included.
        Returns 0 if the consecutive-loss circuit breaker is active."""
        with self._lock:
            if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                logger.warning(
                    "Sizer: BLOCKED — %d consecutive losses (limit=%d). "
                    "Restart the bot to reset.",
                    self._consecutive_losses, MAX_CONSECUTIVE_LOSSES,
                )
                return 0

        if ask_price <= 0 or cash <= 0:
            return 0

        safe_cash = cash * SAFETY_FACTOR

        naive_qty = int(math.floor(safe_cash / ask_price))
        if naive_qty < 1:
            return 0

        qty = naive_qty
        while qty > 0:
            cost = self._kalshi_cost(qty, ask_price)
            if cost <= safe_cash:
                break
            qty -= 1

        cost = self._kalshi_cost(qty, ask_price) if qty > 0 else 0.0
        logger.info(
            "Sizer: cash=$%.4f (safe=$%.4f)  price=%.4f  qty=%d  est_cost=$%.4f",
            cash, safe_cash, ask_price, qty, cost,
        )
        return max(qty, 0)

    def record_result(self, pnl: float) -> None:
        with self._lock:
            self._total_trades += 1

            if pnl > 0:
                self._total_wins       += 1
                self._consecutive_wins  += 1
                self._consecutive_losses = 0
                logger.info(
                    "WIN  streak=%d  total=%d/%d",
                    self._consecutive_wins, self._total_wins, self._total_trades,
                )
            else:
                self._consecutive_losses += 1
                self._consecutive_wins    = 0
                logger.warning(
                    "LOSS  consecutive=%d/%d",
                    self._consecutive_losses, MAX_CONSECUTIVE_LOSSES,
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
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_trades":       self._total_trades,
                "total_wins":         self._total_wins,
                "consecutive_wins":   self._consecutive_wins,
                "consecutive_losses": self._consecutive_losses,
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
            logger.info("PositionSizer: no history — fresh start")
            return

        trades = [float(r[0]) for r in rows]
        self._total_trades = len(trades)
        self._total_wins   = sum(1 for p in trades if p > 0)

        # Consecutive streaks are session-only — don't carry forward losses
        # from a previous (possibly broken) run into a fresh session.
        logger.info(
            "PositionSizer: loaded wins=%d/%d (streak resets each session)",
            self._total_wins, self._total_trades,
        )