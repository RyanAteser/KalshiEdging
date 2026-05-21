"""
risk_manager.py — Coordinates signal → execution → DB.

Key behaviours:
  - Fetches live Kalshi cash balance before every buy (converts cents → dollars)
  - Optimistic position lock: sets engine position BEFORE placing order so no
    duplicate signals fire while the HTTP request is in-flight. Unlocked on failure.
  - qty = floor(balance / ask_price) — uses all available cash
  - Handles ENTRY and STOP_LOSS signals
  - Records PnL, feeds PositionSizer for shutdown logic
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from pykalshi import KalshiClient

from db.db import Database
from core.config import Config
from core.execution_engine import ExecutionEngine
from core.models import Signal, SignalType
from core.position_sizer import PositionSizer
from core.signal_engine_router import SignalEngineRouter
from core.shadow_tracker import ShadowTracker
from core.shadow_vol_tracker import ShadowVolTracker
from core import event_bus
from core.event_bus import TradeEvent, SignalEvent

logger = logging.getLogger(__name__)


class RiskManager:

    FAILED_ORDER_COOLDOWN = 30.0   # seconds to wait after a failed order

    def __init__(
            self,
            db: Database,
            signal_engine: SignalEngineRouter,
            execution_engine: ExecutionEngine,
            config: Config,
            client: KalshiClient,
            on_shutdown: Optional[Callable[[], None]] = None,
    ) -> None:
        self._db             = db
        self._signal_engine  = signal_engine
        self._execution      = execution_engine
        self._config         = config
        self._client         = client
        self._sizer          = PositionSizer(db=db, on_shutdown=on_shutdown)
        self._market_locks: dict[str, threading.Lock] = {}
        self._global_lock    = threading.Lock()
        # Cooldown timestamps per ticker — block entry attempts for N seconds after failure
        self._order_cooldown: dict[str, float] = {}
        # Poller reference — set later via set_poller() to avoid circular init
        self._poller = None
        # Shadow tracker — set later via set_shadow_tracker()
        self._shadow: Optional[ShadowTracker] = None
        # Volume spike shadow tracker — set later via set_shadow_vol_tracker()
        self._shadow_vol: Optional[ShadowVolTracker] = None
        # Local position guard: tickers we just bought, before Kalshi API catches up
        self._local_open_tickers: set[str] = set()

    def set_poller(self, poller) -> None:
        """Inject portfolio_poller reference so buys can arm its grace period."""
        self._poller = poller

    def set_shadow_tracker(self, shadow: ShadowTracker) -> None:
        self._shadow = shadow

    def set_shadow_vol_tracker(self, shadow_vol: ShadowVolTracker) -> None:
        self._shadow_vol = shadow_vol

    def shadow_tick(self, ticker: str, bid: Optional[float], ask: Optional[float]) -> None:
        if self._shadow is not None:
            self._shadow.process_tick(ticker, bid, ask)

    def shadow_vol_tick(
        self, ticker: str, bid: Optional[float], ask: Optional[float], vol: Optional[float]
    ) -> None:
        if self._shadow_vol is not None:
            self._shadow_vol.process_tick(ticker, bid, ask, vol)

    def _get_lock(self, ticker: str) -> threading.Lock:
        with self._global_lock:
            if ticker not in self._market_locks:
                self._market_locks[ticker] = threading.Lock()
            return self._market_locks[ticker]

    # ── Signal dispatch ───────────────────────────────────────────────

    def handle_signal(
            self,
            signal: Signal,
            best_bid: Optional[float],
            best_ask: Optional[float],
    ) -> None:
        lock = self._get_lock(signal.ticker)
        with lock:
            if signal.signal_type == SignalType.ENTRY:
                self._handle_entry(signal, best_bid, best_ask)
            elif signal.signal_type in (SignalType.STOP_LOSS, SignalType.EXIT):
                self._handle_exit(signal, best_bid, best_ask)

    # ── Entry ─────────────────────────────────────────────────────────

    def _handle_entry(
            self,
            signal: Signal,
            best_bid: Optional[float],
            best_ask: Optional[float],
    ) -> None:
        existing = self._db.get_open_position(signal.market_id)
        if existing:
            logger.debug("[%s] Entry skipped: position already open in DB", signal.ticker)
            return

        # Local guard — Kalshi /positions API lags several seconds after fill.
        # Without this, the poller phantom-closes and we double-buy.
        # If the DB shows no open position, the guard is stale (position settled via
        # poller without going through _handle_exit). Clear it so re-entry works.
        if signal.ticker in self._local_open_tickers:
            logger.debug(
                "[%s] Clearing stale local guard — DB shows no open position (settled via poller)",
                signal.ticker,
            )
            self._local_open_tickers.discard(signal.ticker)

        # Check failed-order cooldown
        cooldown_until = self._order_cooldown.get(signal.ticker, 0)
        if time.time() < cooldown_until:
            remaining = cooldown_until - time.time()
            logger.debug(
                "[%s] Entry skipped: in cooldown for %.1fs",
                signal.ticker, remaining,
            )
            return

        if best_ask is None:
            logger.info("[%s] Entry skipped: no ask", signal.ticker)
            return

        # Determine the actual ask price for the side we're buying
        kalshi_side = signal.metadata.get("side", "YES") if signal.metadata else "YES"
        if kalshi_side == "NO":
            # NO ask = 1 - YES bid. Engine already computed this as signal.price
            side_ask = signal.price
        else:
            side_ask = best_ask

        # Sanity check — reject clearly bad prices only
        if side_ask is None or side_ask <= 0 or side_ask > 0.99:
            logger.warning(
                "[%s] Entry skipped: invalid %s ask=%.4f",
                signal.ticker, kalshi_side, side_ask or 0,
                                            )
            return

        # ── Fetch live cash balance ───────────────────────────────────
        cash = self._get_cash_balance()
        if cash is None:
            logger.error("[%s] Entry skipped: could not fetch balance", signal.ticker)
            return

        qty = self._sizer.get_qty(cash=cash, ask_price=side_ask)
        if qty < 1:
            logger.warning(
                "[%s] Entry skipped: insufficient cash ($%.2f) for 1 %s contract at %.4f  "
                "— cooldown 30s",
                signal.ticker, cash, kalshi_side, side_ask,
            )
            # Insufficient cash cooldown — prevents per-tick balance polling spam
            self._order_cooldown[signal.ticker] = time.time() + 30.0
            self._signal_engine.mark_cooldown(signal.ticker, 30.0)
            return

        # ── Optimistic lock ───────────────────────────────────────────
        self._signal_engine.mark_position_open(
            ticker=signal.ticker,
            position_id=-1,
            entry_price=side_ask,
            side=kalshi_side,
        )

        # Persist signal
        self._db.insert_signal(
            market_id=signal.market_id,
            signal_type=signal.signal_type.value,
            price=signal.price,
            metadata=signal.metadata,
        )
        event_bus.push_signal(SignalEvent(
            ticker=signal.ticker,
            signal_type=signal.signal_type.value,
            price=signal.price,
        ))

        result = self._execution.buy(
            ticker=signal.ticker,
            price=side_ask,
            best_ask=side_ask,
            qty=qty,
            kalshi_side=kalshi_side,
        )

        if not result.success:
            logger.error("[%s] Buy failed: %s", signal.ticker, result.error)

            # Before releasing the lock, verify with Kalshi whether the position
            # actually exists. It's possible an earlier retry succeeded and only
            # a later retry saw the error. If position exists, keep the lock.
            if self._position_exists_on_kalshi(signal.ticker):
                logger.warning(
                    "[%s] Order reported failed BUT position exists on Kalshi — "
                    "keeping lock, recording as filled",
                    signal.ticker,
                )
                # Record the position in the DB with best_ask as entry (estimate)
                position_id = self._db.open_position(
                    market_id=signal.market_id,
                    entry_price=best_ask,
                    quantity=qty,
                    stop_loss=round((best_ask or signal.price) - 0.02, 6),
                )
                self._signal_engine.mark_position_open(
                    ticker=signal.ticker,
                    position_id=position_id,
                    entry_price=best_ask,
                    side=kalshi_side,
                )
                return

            # Genuinely failed — set cooldown in both risk_manager and engine
            self._order_cooldown[signal.ticker] = time.time() + self.FAILED_ORDER_COOLDOWN
            self._signal_engine.mark_position_closed(signal.ticker)
            self._local_open_tickers.discard(signal.ticker)

            # Tell the engine to stop spamming signals during cooldown
            self._signal_engine.mark_cooldown(signal.ticker, self.FAILED_ORDER_COOLDOWN)

            logger.info(
                "[%s] Cooldown set for %.0fs",
                signal.ticker, self.FAILED_ORDER_COOLDOWN,
            )
            return

        filled_price = result.filled_price or signal.price

        self._db.insert_trade(
            market_id=signal.market_id,
            side="BUY",
            price=filled_price,
            quantity=result.filled_qty,
        )

        computed_stop = self._signal_engine.get_stop_price() or round(filled_price - 0.02, 6)
        position_id = self._db.open_position(
            market_id=signal.market_id,
            entry_price=filled_price,
            quantity=result.filled_qty,
            stop_loss=computed_stop,
        )

        # Update engine with real position_id now that we have it
        self._signal_engine.mark_position_open(
            ticker=signal.ticker,
            position_id=position_id,
            entry_price=filled_price,
            side=kalshi_side,
        )

        # Log feature snapshot for ML training data
        try:
            features = self._signal_engine.get_last_features(signal.ticker)
            if features:
                self._db.log_ev_entry(
                    ticker=signal.ticker,
                    market_id=signal.market_id,
                    position_id=position_id,
                    side=kalshi_side,
                    entry_price=filled_price,
                    features=features,
                )
        except Exception as exc:
            logger.warning("[%s] EV feature log failed: %s", signal.ticker, exc)

        # CRITICAL: Tell poller a position was just opened so it waits out the
        # Kalshi /positions API propagation delay before assuming it's "gone"
        if self._poller is not None:
            try:
                self._poller.note_position_opened(signal.ticker)
            except Exception as exc:
                logger.warning("Could not notify poller: %s", exc)

        # Mark locally so we don't double-buy during Kalshi API lag
        self._local_open_tickers.add(signal.ticker)

        if self._shadow is not None:
            self._shadow.open(signal.ticker, kalshi_side, filled_price)
        if self._shadow_vol is not None:
            self._shadow_vol.open(signal.ticker, kalshi_side, filled_price)

        event_bus.push_trade(TradeEvent(
            ticker=signal.ticker, side="BUY",
            price=filled_price, qty=result.filled_qty,
        ))
        logger.info(
            "[%s] BOUGHT: entry=%.4f  qty=%d  cash_used=$%.2f  position_id=%d",
            signal.ticker, filled_price, result.filled_qty,
            filled_price * result.filled_qty, position_id,
            )

    # ── Exit ──────────────────────────────────────────────────────────

    def _handle_exit(
            self,
            signal: Signal,
            best_bid: Optional[float],
            best_ask: Optional[float],
    ) -> None:
        position = self._db.get_open_position(signal.market_id)
        if not position:
            logger.debug("[%s] Exit skipped: no open position", signal.ticker)
            return

        pos_id      = position[0]
        entry_price = position[2]
        quantity    = position[3]

        kalshi_side = signal.metadata.get("side", "YES") if signal.metadata else "YES"

        # For NO positions: signal.price is already the NO price (1 - YES ask).
        # For YES positions: signal.price is the YES price.
        # sell() needs the price of the SIDE being sold.
        if kalshi_side == "NO":
            # NO bid = 1 - YES ask. Use live best_ask for most accurate price.
            no_bid = round(1.0 - (best_ask or (1.0 - signal.price)), 6)
            sell_price = max(no_bid * 0.95, 0.01)   # 5% below NO bid as limit
        else:
            sell_price = signal.price * 0.95   # 5% below YES price as limit

        self._db.insert_signal(
            market_id=signal.market_id,
            signal_type=signal.signal_type.value,
            price=signal.price,
            metadata=signal.metadata,
        )

        result = self._execution.sell(
            ticker=signal.ticker,
            price=sell_price,
            best_bid=best_bid,
            quantity=quantity,
            kalshi_side=kalshi_side,
        )

        if not result.success:
            err_str = str(result.error or "").lower()
            if "market_closed" in err_str or "market closed" in err_str:
                # Market settled before we could sell — treat as a settlement win.
                # The poller would eventually do this, but doing it now re-arms the
                # engine immediately so we can trade the next market without delay.
                logger.warning(
                    "[%s] Sell blocked: market already closed — recording as settlement",
                    signal.ticker,
                )
                self._db.close_position(pos_id)
                self._signal_engine.mark_position_closed(signal.ticker)
                self._local_open_tickers.discard(signal.ticker)
                pnl = (1.0 - entry_price) * quantity
                self._sizer.record_result(pnl)
                if self._shadow is not None:
                    self._shadow.close_all(signal.ticker, 1.0, "settlement")
                if self._shadow_vol is not None:
                    self._shadow_vol.close_all(signal.ticker, 1.0, "settlement")
                event_bus.push_trade(TradeEvent(
                    ticker=signal.ticker, side="SELL",
                    price=1.0, qty=quantity, pnl=pnl,
                ))
            else:
                logger.error("[%s] Sell failed: %s", signal.ticker, result.error)
            return

        exit_price = result.filled_price or sell_price
        # PnL: for both sides, profit = (exit - entry) * qty
        # YES: exit > entry = win. NO: exit > entry = win (entry/exit are NO prices).
        pnl = (exit_price - entry_price) * result.filled_qty

        self._db.insert_trade(
            market_id=signal.market_id,
            side="SELL",
            price=exit_price,
            quantity=result.filled_qty,
            pnl=pnl,
        )
        self._db.close_position(pos_id)
        self._signal_engine.mark_position_closed(signal.ticker)
        self._local_open_tickers.discard(signal.ticker)   # clear local guard
        self._sizer.record_result(pnl)

        # Close ML training log for this position
        try:
            exit_reason = signal.metadata.get("reason", signal.signal_type.value) if signal.metadata else signal.signal_type.value
            self._db.close_ev_log(pos_id, exit_price, exit_reason, pnl)
        except Exception as exc:
            logger.warning("[%s] EV feature log close failed: %s", signal.ticker, exc)
        if self._shadow is not None:
            self._shadow.close_all(signal.ticker, exit_price, "real_exit")
        if self._shadow_vol is not None:
            self._shadow_vol.close_all(signal.ticker, exit_price, "real_exit")

        event_bus.push_trade(TradeEvent(
            ticker=signal.ticker, side="SELL",
            price=exit_price, qty=result.filled_qty, pnl=pnl,
        ))
        logger.info(
            "[%s] SOLD: exit=%.4f  pnl=%+.4f  (%s)",
            signal.ticker, exit_price, pnl,
            "WIN" if pnl > 0 else "LOSS",
        )

    # ── Balance fetch ─────────────────────────────────────────────────

    def _get_cash_balance(self) -> Optional[float]:
        """
        Fetch available cash from Kalshi portfolio.
        Kalshi returns balance as an integer in CENTS (e.g. 1000 = $10.00).
        """
        try:
            resp = self._client.portfolio.get_balance()

            if isinstance(resp, dict):
                raw = (resp.get("balance")
                       or resp.get("cash_balance")
                       or resp.get("available_balance"))
            else:
                raw = (getattr(resp, "balance", None)
                       or getattr(resp, "cash_balance", None)
                       or getattr(resp, "available_balance", None))

            if raw is None:
                logger.warning("Balance response missing: %s", resp)
                return None

            # Kalshi ALWAYS returns balance as integer cents (e.g. 52 = $0.52, 1000 = $10.00)
            # Never skip the division — a raw value of 52 means $0.52, not $52.
            dollars = float(raw) / 100.0
            logger.info("Cash balance: $%.2f (raw cents=%s)", dollars, raw)
            return dollars

        except Exception as exc:
            logger.error("Failed to fetch cash balance: %s", exc)
            return None

    # ── Position existence check ──────────────────────────────────────

    def _position_exists_on_kalshi(self, ticker: str) -> bool:
        """
        Query Kalshi portfolio to see if a position actually exists for this ticker.
        Used to reconcile when an order reports failure but may have succeeded.
        """
        try:
            resp = self._client.portfolio.get_positions()

            if hasattr(resp, "market_positions"):
                positions = resp.market_positions or []
            elif hasattr(resp, "positions"):
                positions = resp.positions or []
            elif isinstance(resp, dict):
                positions = resp.get("market_positions") or resp.get("positions") or []
            else:
                positions = list(resp) if resp else []

            for pos in positions:
                if isinstance(pos, dict):
                    pt = pos.get("ticker") or pos.get("market_ticker")
                    qty = pos.get("position") or pos.get("yes_position") or 0
                else:
                    pt = getattr(pos, "ticker", None) or getattr(pos, "market_ticker", None)
                    qty = getattr(pos, "position", 0) or getattr(pos, "yes_position", 0)

                if pt == ticker and abs(float(qty or 0)) > 0:
                    logger.info("[%s] Position found on Kalshi: qty=%s", ticker, qty)
                    return True

            return False
        except Exception as exc:
            logger.warning("[%s] Position check failed: %s — assuming no position",
                           ticker, exc)
            return False