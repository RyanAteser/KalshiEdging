"""
signal_engine_router.py — Thin wrapper around EVSignalEngine.

Retains the same public interface that worker.py, risk_manager.py, and
portfolio_poller.py depend on. Old engine references have been removed;
this now routes exclusively to the EV grid filter strategy.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, TYPE_CHECKING

from core.config import Config
from core.models import Signal

if TYPE_CHECKING:
    from core.btc_feed import BtcFeed
    from core.binance_feed import BinanceFeed
    from core.binance_futures_feed import BinanceFuturesFeed
    from core.signal_engine_ev import EVSignalEngine

logger = logging.getLogger(__name__)


class SignalEngineRouter:
    """
    Routes all ticks to EVSignalEngine.

    Call set_ev_engine() after all data feeds are running, before workers start.
    Until then, process_tick() returns None safely.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._lock   = threading.Lock()
        self._ev: Optional["EVSignalEngine"] = None
        logger.info("SignalEngineRouter initialised — waiting for EV engine")

    def set_ev_engine(
        self,
        btc_feed: "BtcFeed",
        binance_feed: "BinanceFeed",
        binance_futures_feed: "BinanceFuturesFeed",
    ) -> None:
        """Wire the EV engine once all data feeds are ready."""
        from core.signal_engine_ev import EVSignalEngine
        with self._lock:
            self._ev = EVSignalEngine(
                self._config, btc_feed, binance_feed, binance_futures_feed,
            )
        logger.info("EV Grid Filter engine armed")

    # ── Backward-compat stubs (called by worker.py, safe no-ops now) ─

    def set_t2t_engine(self, btc_feed: "BtcFeed") -> None:
        pass

    def update_t2t_context(self, ticker: str, btc_target, close_ts) -> None:
        pass

    # ── Dashboard label ───────────────────────────────────────────────

    @property
    def active_key(self) -> str:
        return "ev_grid"

    @property
    def active_label(self) -> str:
        return "EV Grid Filter"

    # ── Core routing ──────────────────────────────────────────────────

    def process_tick(
        self,
        ticker: str,
        market_id: int,
        price: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
        volume: Optional[float] = None,
    ) -> Optional[Signal]:
        with self._lock:
            ev = self._ev
        if ev is None:
            return None
        return ev.process_tick(ticker, market_id, price, best_bid, best_ask, volume)

    def get_or_create_state(self, ticker: str, market_id: int):
        with self._lock:
            ev = self._ev
        if ev is None:
            return None
        return ev.get_or_create_state(ticker, market_id)

    def mark_position_open(
        self,
        ticker: str,
        position_id: int,
        entry_price: float,
        side: Optional[str] = None,
    ) -> None:
        with self._lock:
            ev = self._ev
        if ev:
            ev.mark_position_open(ticker, position_id, entry_price, side=side)

    def mark_position_closed(self, ticker: str) -> None:
        with self._lock:
            ev = self._ev
        if ev:
            ev.mark_position_closed(ticker)

    def mark_cooldown(self, ticker: str, duration: float = 30.0) -> None:
        with self._lock:
            ev = self._ev
        if ev:
            ev.mark_cooldown(ticker, duration)

    def get_last_features(self, ticker: str) -> Optional[dict]:
        """Return ML training feature snapshot from the last entry signal."""
        with self._lock:
            ev = self._ev
        if ev is None:
            return None
        return ev.get_last_features(ticker)

    def get_position_snapshot(self, ticker: str) -> Optional[dict]:
        with self._lock:
            ev = self._ev
        if ev is None:
            return None
        return ev.get_position_snapshot(ticker)

    def get_stop_price(self) -> Optional[float]:
        with self._lock:
            ev = self._ev
        if ev is None:
            return None
        return ev.get_stop_price()
