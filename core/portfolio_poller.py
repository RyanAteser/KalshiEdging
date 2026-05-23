"""
portfolio_poller.py — Polls Kalshi positions and syncs engine state BIDIRECTIONALLY.

CRITICAL BEHAVIOR:
  1. Kalshi has position, engine doesn't → adopt it (orphan)
  2. Engine has position, Kalshi doesn't → close it... BUT ONLY after a 20s
     grace period to prevent the Kalshi API race condition where positions
     API lags behind the orders API immediately after a fill.

Without the grace period, the poller sees an empty positions list 0-5 seconds
after a buy, marks the position closed in the engine/DB, and the stop loss
never fires.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional, Set

from pykalshi import KalshiClient

from core.signal_engine_router import SignalEngineRouter
from core.position_sizer import PositionSizer
from db.db import Database
from core import event_bus
from core.event_bus import TradeEvent

logger = logging.getLogger(__name__)

POLL_INTERVAL       = 5.0
MAX_ERRORS          = 5
RECONCILE_GRACE_SEC = 20.0   # Wait this long after open before trusting "position gone"


class PortfolioPoller(threading.Thread):

    def __init__(
            self,
            client:        KalshiClient,
            signal_engine: SignalEngineRouter,
            db:            Database,
            sizer:         Optional[PositionSizer] = None,
    ) -> None:
        super().__init__(daemon=True, name="portfolio-poller")
        self._client        = client
        self._signal_engine = signal_engine
        self._db            = db
        self._sizer         = sizer
        self._stop_event    = threading.Event()
        self._error_count   = 0

        # Tracks when each position was opened (for grace period)
        self._position_opened_at: dict[str, float] = {}
        self._logged_raw_format   = False   # log raw response once for debugging
        self._shadow = None

    def stop(self) -> None:
        self._stop_event.set()

    def note_position_opened(self, ticker: str) -> None:
        """Call from risk_manager after a successful buy to arm grace period."""
        self._position_opened_at[ticker] = time.time()
        logger.debug("[%s] Grace period armed for %.0fs", ticker, RECONCILE_GRACE_SEC)

    def set_shadow_tracker(self, shadow) -> None:
        self._shadow = shadow

    def run(self) -> None:
        logger.info(
            "Portfolio poller started (%.1fs interval, grace=%.0fs)",
            POLL_INTERVAL, RECONCILE_GRACE_SEC,
        )
        while not self._stop_event.is_set():
            try:
                self._check()
                self._error_count = 0
            except Exception as exc:
                msg = str(exc)
                # 401 is an auth/permissions issue — retrying won't fix it.
                # Log a warning and back off, but don't count toward fatal limit.
                if "401" in msg or "authentication" in msg.lower():
                    logger.warning("Poll auth error (not counting as fatal): %s", exc)
                else:
                    self._error_count += 1
                    logger.warning("Poll failed (%d/%d): %s", self._error_count, MAX_ERRORS, exc)
                    if self._error_count >= MAX_ERRORS:
                        logger.error("Poller stopping — too many errors")
                        break
            self._stop_event.wait(POLL_INTERVAL)
        logger.info("Portfolio poller stopped")

    # ── Core check ───────────────────────────────────────────────────

    def _check(self) -> None:
        engine_open: Set[str]          = self._get_engine_open_tickers()
        kalshi_open: Dict[str, dict]   = self._get_kalshi_open_positions()
        kalshi_tickers                  = set(kalshi_open.keys())

        now = time.time()

        # ── 1. Kalshi has position, engine doesn't → ADOPT (orphan) ──
        orphans = kalshi_tickers - engine_open
        for ticker in orphans:
            self._adopt_orphan(ticker, kalshi_open[ticker])

        # ── 2. Engine has position, Kalshi doesn't → CLOSE (maybe) ───
        missing = engine_open - kalshi_tickers
        for ticker in missing:
            opened_at = self._position_opened_at.get(ticker, 0)
            age = now - opened_at

            # CRITICAL grace period check — Kalshi /positions endpoint
            # lags 5-15s behind /orders. Without this, we close positions
            # that just filled and stop loss never fires.
            if age < RECONCILE_GRACE_SEC:
                logger.debug(
                    "[%s] Still within grace period (%.1fs / %.0fs) — NOT closing",
                    ticker, age, RECONCILE_GRACE_SEC,
                )
                continue

            self._close_missing(ticker)

    # ── Orphan adoption ─────────────────────────────────────────────

    def _adopt_orphan(self, ticker: str, pos_data: dict) -> None:
        entry_px = pos_data.get("avg_price", 0.0)
        qty      = pos_data.get("qty", 0)
        side     = pos_data.get("side", "YES")

        logger.warning(
            "[%s] ORPHAN DETECTED on Kalshi — registering for stop loss. "
            "qty=%d side=%s entry≈%.4f",
            ticker, qty, side, entry_px,
        )

        try:
            market_id = self._db.get_market_id(ticker)
            existing  = self._db.get_open_position(market_id) if market_id else None

            if not existing:
                # No DB record → this position was NOT opened by this bot.
                # Skip adoption so we don't interfere with another system's trades.
                logger.info(
                    "[%s] Orphan has no DB record — likely from another system. Skipping.",
                    ticker,
                )
                return

            pos_id = existing["id"]

            # Use the router's public interface to register the orphan
            self._signal_engine.mark_position_open(
                ticker=ticker,
                position_id=pos_id,
                entry_price=entry_px or 0.65,
                side=side,
            )

            # Arm grace period so we don't immediately try to close
            self._position_opened_at[ticker] = time.time()

            logger.info(
                "[%s] Adopted: id=%d side=%s entry=%.4f qty=%d",
                ticker, pos_id, side, entry_px or 0.65, qty,
            )

        except Exception as exc:
            logger.error("[%s] Failed to adopt orphan: %s", ticker, exc)

    # ── Closing ─────────────────────────────────────────────────────

    def _close_missing(self, ticker: str) -> None:
        logger.info("[%s] Gone from Kalshi (past grace) — closing engine state", ticker)
        self._signal_engine.mark_position_closed(ticker)
        self._position_opened_at.pop(ticker, None)

        try:
            market_id = self._db.get_market_id(ticker)
            if not market_id:
                return
            position = self._db.get_open_position(market_id)
            if not position:
                return

            pos_id      = position["id"]
            entry_price = position["entry_price"]
            qty         = position["quantity"]

            # Settled = win ($1 payout)
            pnl = (1.0 - entry_price) * qty
            self._db.close_position(pos_id, exit_price=1.0, exit_reason="settlement", pnl=pnl)
            if self._sizer:
                self._sizer.record_result(pnl)
            if self._shadow is not None:
                self._shadow.close_all(ticker, 1.0, "settlement")

            logger.info(
                "[%s] Closed: entry=%.4f qty=%d pnl=+%.4f (assumed settlement)",
                ticker, entry_price, qty, pnl,
            )
            event_bus.push_trade(TradeEvent(
                ticker=ticker, side="SELL",
                price=1.0, qty=qty, pnl=pnl,
            ))

        except Exception as exc:
            logger.warning("[%s] DB close error: %s", ticker, exc)

    # ── Engine state ──────────────────────────────────────────────────

    def _get_engine_open_tickers(self) -> Set[str]:
        open_tickers: Set[str] = set()
        ev = getattr(self._signal_engine, "_ev", None)
        if ev is None:
            return open_tickers
        for ticker, st in ev._states.items():
            if st.has_position:
                open_tickers.add(ticker)
        return open_tickers

    # ── Kalshi positions with diagnostic logging ────────────────────

    def _get_kalshi_open_positions(self) -> Dict[str, dict]:
        """
        Returns {ticker: {qty, avg_price, side}} for all non-zero positions.
        """
        out: Dict[str, dict] = {}

        try:
            resp = self._client.portfolio.get_positions()
        except Exception as exc:
            raise RuntimeError(f"get_positions() failed: {exc}") from exc

        # Extract list of positions from response
        if hasattr(resp, "market_positions"):
            positions = resp.market_positions or []
        elif hasattr(resp, "positions"):
            positions = resp.positions or []
        elif isinstance(resp, dict):
            positions = resp.get("market_positions") or resp.get("positions") or []
        else:
            positions = list(resp) if resp else []

        # ── Log raw structure ONCE so we can see what Kalshi actually returns
        if not self._logged_raw_format and positions:
            first = positions[0]
            if isinstance(first, dict):
                logger.warning("Kalshi position RAW (dict) keys=%s  sample=%s",
                               list(first.keys()), first)
            else:
                attrs = [a for a in dir(first) if not a.startswith("_")]
                logger.warning("Kalshi position RAW (obj) attrs=%s", attrs)
                try:
                    logger.warning("  repr=%s", repr(first))
                except Exception:
                    pass
            self._logged_raw_format = True

        for pos in positions:
            def _g(*attrs):
                """Try multiple attribute/key names, return first non-None."""
                for attr in attrs:
                    if isinstance(pos, dict):
                        v = pos.get(attr)
                    else:
                        v = getattr(pos, attr, None)
                    if v is not None:
                        return v
                return None

            ticker = _g("ticker", "market_ticker")

            # Try EVERY known field name for position size
            raw_qty = _g(
                "position_fp",     # ← Kalshi's actual field (string like '0.00')
                "position", "yes_position", "quantity", "qty",
                "size", "contracts", "net_position",
            )

            # Coerce to number — Kalshi might return int, str, or float
            try:
                qty_f = float(raw_qty) if raw_qty is not None else 0.0
            except (TypeError, ValueError):
                qty_f = 0.0

            # Log every parsed position for debugging (DEBUG level only)
            logger.debug(
                "Parsed position: ticker=%s raw_qty=%r qty_f=%.2f",
                ticker, raw_qty, qty_f,
            )

            if not ticker or qty_f == 0:
                continue

            # Extract average price — multiple possible field names
            avg_raw = _g(
                "market_exposure_dollars",   # ← Kalshi's actual field (total $ spent)
                "average_price", "avg_price", "market_exposure",
                "avg_entry_price", "entry_price",
            )
            try:
                avg_val = float(avg_raw) if avg_raw is not None else 0.0
                # market_exposure_dollars is always total $ exposure — always divide by qty.
                # The old heuristic (divide only if avg_val > qty) was wrong when total
                # exposure < qty numerically (e.g. $9.80 for 10 contracts → 0.98/contract).
                if avg_val > 0 and abs(qty_f) > 0:
                    per_contract = avg_val / abs(qty_f)
                    # Normalize: cents → dollars if > 1 (some API versions return cents)
                    avg_price = per_contract / 100.0 if per_contract > 1.0 else per_contract
                else:
                    avg_price = 0.0
            except (TypeError, ValueError):
                avg_price = 0.0

            side = "YES" if qty_f > 0 else "NO"

            out[ticker] = {
                "qty":       int(abs(qty_f)),
                "avg_price": avg_price,
                "side":      side,
            }

        if out:
            logger.debug("Kalshi open positions: %s", list(out.keys()))
        return out