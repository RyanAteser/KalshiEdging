"""
market_rotator.py — Rolls BTC 15m markets forward as they close.

Primary trigger: wall-clock time. When now >= close_ts the market
is over regardless of price. Rolls immediately.

Secondary triggers (catch edge cases):
  - YES settled: ask >= 99c (YES won)
  - NO  settled: ask <= 1c  (NO won, YES worthless)
  - Stale:       no tick in 120s
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from pykalshi import KalshiClient
from pykalshi.models import MarketStatus

from core.config import Config
from core.risk_manager import RiskManager
from core.signal_engine_router import SignalEngineRouter
from core.worker import MarketWorker
from db.db import Database

logger = logging.getLogger(__name__)

CHECK_INTERVAL   = 3.0     # check every 3s — fast enough to catch close on time
STALE_SECONDS    = 120.0
MIN_AHEAD_SEC    = 15      # accept markets closing >= 15s from now
ROLL_BEFORE_SEC  = 5       # start rolling 5s before close_ts
BTC_SERIES       = "KXBTC15M"


# ── Helpers ───────────────────────────────────────────────────────────

def _parse_ts(val) -> int:
    if val is None:
        return 0
    try:
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val)
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S"):
            try:
                return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                pass
    except Exception:
        pass
    return 0


def _get_close_ts(market) -> int:
    for attr in ("close_ts", "close_time", "close_time_ts", "close_timestamp",
                 "closeTime", "close_time_seconds", "close"):
        val = market.get(attr) if isinstance(market, dict) else getattr(market, attr, None)
        ts = _parse_ts(val)
        if ts > 0:
            return ts
    return 0


class MarketRotator(threading.Thread):

    YES_SETTLED_THRESHOLD = 0.99   # ask >= this → YES won
    NO_SETTLED_THRESHOLD  = 0.01   # ask <= this → NO won (YES worthless)

    def __init__(
            self,
            client: KalshiClient,
            db: Database,
            signal_engine: SignalEngineRouter,
            risk_manager: RiskManager,
            config: Config,
            workers: list[MarketWorker],
            workers_lock: threading.Lock,
            on_remove: Optional[Callable[[str], None]] = None,
            on_add:    Optional[Callable[[str, int], None]] = None,
    ) -> None:
        super().__init__(daemon=True, name="market-rotator")
        self._client        = client
        self._db            = db
        self._signal_engine = signal_engine
        self._risk_manager  = risk_manager
        self._config        = config
        self._workers       = workers
        self._workers_lock  = workers_lock
        self._on_remove     = on_remove
        self._on_add        = on_add
        self._stop_event    = threading.Event()

        # Per-ticker tracking
        self._last_ask:       dict[str, float] = {}
        self._last_bid:       dict[str, float] = {}
        self._last_tick_time: dict[str, float] = {}
        self._settled_streak: dict[str, int]   = {}
        self._close_ts:       dict[str, int]   = {}   # close timestamp per ticker

    def stop(self) -> None:
        self._stop_event.set()

    def register_market(self, ticker: str, close_ts: int) -> None:
        """Called after a worker starts so rotator knows when to roll it."""
        self._close_ts[ticker] = close_ts
        logger.info(
            "[rotator] Registered %s  close_ts=%s (%s UTC)",
            ticker, close_ts,
            datetime.fromtimestamp(close_ts, tz=timezone.utc).strftime("%H:%M:%S")
            if close_ts else "unknown",
        )

    def update_price(self, ticker: str, price=None, ask=None, bid=None) -> None:
        now = time.time()
        if price is not None:
            self._last_tick_time[ticker] = now
        if ask is not None:
            self._last_ask[ticker]       = ask
            self._last_tick_time[ticker] = now
        if bid is not None:
            self._last_bid[ticker]       = bid
            self._last_tick_time[ticker] = now

    def run(self) -> None:
        logger.info("Market rotator started (BTC 15m, check=%.0fs)", CHECK_INTERVAL)

        import core.event_bus as event_bus
        def _on_market(event):
            self.update_price(event.ticker, price=event.price,
                              ask=event.ask, bid=event.bid)
        event_bus.subscribe_market(_on_market)

        while not self._stop_event.is_set():
            try:
                self._check_and_rotate()
            except Exception:
                logger.exception("Rotator error")
            self._stop_event.wait(CHECK_INTERVAL)

    # ── Check ─────────────────────────────────────────────────────────

    def _check_and_rotate(self) -> None:
        now = time.time()

        with self._workers_lock:
            current_workers = list(self._workers)

        for worker in current_workers:
            ticker    = worker._ticker
            ask       = self._last_ask.get(ticker)
            bid       = self._last_bid.get(ticker)
            last_tick = self._last_tick_time.get(ticker, now)
            close_ts  = self._close_ts.get(ticker, 0)

            reason = None

            # ── PRIMARY: time-based ────────────────────────────────────
            if close_ts > 0 and now >= (close_ts - ROLL_BEFORE_SEC):
                reason = f"close_ts reached ({datetime.fromtimestamp(close_ts, tz=timezone.utc).strftime('%H:%M:%S')} UTC)"

            # ── SECONDARY: YES settled (ask >= 99c) ───────────────────
            elif ask is not None and ask >= self.YES_SETTLED_THRESHOLD:
                streak = self._settled_streak.get(ticker, 0) + 1
                self._settled_streak[ticker] = streak
                if streak >= 2:
                    reason = f"YES settled (ask={ask:.2f})"
            elif ask is not None and ask <= self.NO_SETTLED_THRESHOLD and ask > 0:
                # NO settled: YES is near-worthless
                streak = self._settled_streak.get(ticker, 0) + 1
                self._settled_streak[ticker] = streak
                if streak >= 2:
                    reason = f"NO settled (ask={ask:.2f})"
            else:
                self._settled_streak[ticker] = 0

            # ── TERTIARY: stale ────────────────────────────────────────
            if reason is None and (now - last_tick) > STALE_SECONDS:
                reason = f"stale {now - last_tick:.0f}s"

            if reason:
                logger.info("[%s] Rolling → %s", ticker, reason)
                self._roll(ticker)
                return   # one roll per cycle

    # ── Roll ──────────────────────────────────────────────────────────

    def _roll(self, old_ticker: str) -> None:
        next_market = self._find_next_market(exclude=old_ticker)

        if next_market is None:
            logger.warning(
                "No next BTC market found — backoff 30s before next check. "
                "Kalshi may not have published the next 15m market yet."
            )
            self._settled_streak[old_ticker] = 0
            # Backoff — don't try to roll this ticker for 30 seconds
            # Extend close_ts forward so time trigger doesn't keep firing
            self._close_ts[old_ticker] = int(time.time()) + 30
            return

        new_ticker = next_market["ticker"]
        new_close  = next_market["close_ts"]

        # Stop old worker
        old_worker = None
        with self._workers_lock:
            for w in self._workers:
                if w._ticker == old_ticker:
                    old_worker = w
                    break

        if old_worker:
            old_worker.stop()
            old_worker.join(timeout=3.0)
            with self._workers_lock:
                self._workers[:] = [w for w in self._workers if w._ticker != old_ticker]

        # Cleanup old ticker state
        for d in (self._last_ask, self._last_bid, self._last_tick_time,
                  self._settled_streak, self._close_ts):
            d.pop(old_ticker, None)

        if self._on_remove:
            self._on_remove(old_ticker)

        # Start new worker
        market_id  = self._db.upsert_market(new_ticker, new_ticker, close_ts=float(new_close) if new_close else None)
        new_worker = MarketWorker(
            client=self._client,
            ticker=new_ticker,
            market_id=market_id,
            db=self._db,
            signal_engine=self._signal_engine,
            risk_manager=self._risk_manager,
            config=self._config,
        )

        with self._workers_lock:
            self._workers.append(new_worker)

        self.register_market(new_ticker, new_close)
        new_worker.start()

        if self._on_add:
            self._on_add(new_ticker, market_id)

        logger.info(
            "Rotated %s → %s  (closes %s UTC)",
            old_ticker, new_ticker,
            datetime.fromtimestamp(new_close, tz=timezone.utc).strftime("%H:%M:%S")
            if new_close else "?",
        )

    # ── Find next market ──────────────────────────────────────────────

    def _find_next_market(self, exclude: str) -> Optional[dict]:
        now = time.time()
        try:
            resp = self._client.get_markets(
                status=MarketStatus.OPEN,
                series_ticker=BTC_SERIES,
                limit=200,
            )
        except Exception as exc:
            logger.warning("Rotator fetch failed: %s", exc)
            return None

        if hasattr(resp, "markets"):
            markets = resp.markets or []
        elif isinstance(resp, dict):
            markets = resp.get("markets") or []
        else:
            markets = list(resp) if resp else []

        candidates = []
        for m in markets:
            t = (m.get("ticker") if isinstance(m, dict) else getattr(m, "ticker", None))
            if not t or t == exclude:
                continue
            cts = _get_close_ts(m)
            if cts <= 0:
                continue
            ahead = cts - now
            if ahead < MIN_AHEAD_SEC:
                continue   # too close to closing — skip
            candidates.append((t, cts))

        if not candidates:
            logger.warning(
                "No candidates found (excluded=%s, total_markets=%d)",
                exclude, len(markets),
            )
            return None

        candidates.sort(key=lambda x: x[1])
        ticker, cts = candidates[0]
        logger.debug("Next market: %s  close_ts=%d  ahead=%.0fs", ticker, cts, cts - now)
        return {"ticker": ticker, "close_ts": cts}