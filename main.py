"""
main.py — Entry point for the Kalshi EV Grid Filter trading system.

Startup sequence:
  1. Load config from environment
  2. Initialize DB + schema
  3. Start Coinbase spot feed (mid price + CVD)
  4. Start BTC candle feed (Coinbase, for candle_boost)
  5. Wire feeds into EV signal engine via router
  6. Fetch active KXBTC15M market from Kalshi
  7. Start MarketWorker + MarketRotator (rolls to next market on expiry)
  8. Start portfolio poller
  9. Block until KeyboardInterrupt, then graceful shutdown
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time

from dotenv import load_dotenv
from pykalshi import KalshiClient

from core.btc_feed import BtcFeed
import core.coinbase_spot_feed as coinbase_spot_feed_module
from core.config import load_config
from core.execution_engine import ExecutionEngine
from core.market_fetcher import fetch_active_sports_markets
from core.market_rotator import MarketRotator
from core.portfolio_poller import PortfolioPoller
from core.position_sizer import PositionSizer
from core.risk_manager import RiskManager
from core.shadow_tracker import ShadowTracker
from core.shadow_vol_tracker import ShadowVolTracker
from core.signal_engine_router import SignalEngineRouter
from core.worker import MarketWorker
from db.db import Database

load_dotenv()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(
                open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
            ),
            logging.FileHandler("kalshi_trader.log", encoding="utf-8"),
        ],
    )


def main() -> None:
    config = load_config()
    setup_logging(config.log_level)

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting Kalshi EV Grid Filter | paper_mode=%s  max_markets=%d  "
        "grid=[%.2f, %.2f]  min_ev=%.4f",
        config.paper_trade, config.max_markets,
        config.ev_grid_min, config.ev_grid_max, config.ev_min_entry,
    )

    # ── Database ──────────────────────────────────────────────────────
    db = Database()
    db.create_schema()
    session_id = db.start_session(config)

    # ── Kalshi client ─────────────────────────────────────────────────
    client = KalshiClient.from_env()

    # ── Data feeds ────────────────────────────────────────────────────
    btc_feed     = BtcFeed()
    btc_feed.start()
    logger.info("BTC candle feed started (Coinbase 15m)")

    coinbase_spot_feed = coinbase_spot_feed_module.get_instance()
    logger.info("Coinbase spot feed starting...")

    # Give feeds a moment to receive first data
    time.sleep(2)

    # ── Signal engine ─────────────────────────────────────────────────
    router = SignalEngineRouter(config)
    router.set_ev_engine(btc_feed, coinbase_spot_feed)

    # ── Execution + risk ──────────────────────────────────────────────
    execution_engine = ExecutionEngine(client, config)
    risk_manager     = RiskManager(db, router, execution_engine, config, client)
    risk_manager.set_shadow_tracker(ShadowTracker(db))
    risk_manager.set_shadow_vol_tracker(ShadowVolTracker(db))

    # ── Portfolio poller ──────────────────────────────────────────────
    sizer  = PositionSizer(db=db)
    poller = PortfolioPoller(client=client, signal_engine=router, db=db, sizer=sizer)
    poller.start()
    risk_manager.set_poller(poller)

    # ── Fetch markets ─────────────────────────────────────────────────
    markets = []
    while not markets:
        markets = fetch_active_sports_markets(client, config)
        if not markets:
            logger.warning("No live KXBTC15M markets found — retrying in 60s...")
            time.sleep(60)

    # ── Upsert markets + start workers ────────────────────────────────
    workers: list[MarketWorker] = []
    workers_lock = threading.Lock()

    for m in markets:
        ticker    = m["ticker"]
        event     = m.get("event", "")
        market_id = db.upsert_market(
            ticker, event,
            btc_target=m.get("btc_target"),
            close_ts=float(m.get("close_ts")) if m.get("close_ts") else None,
        )

        worker = MarketWorker(
            client=client,
            ticker=ticker,
            market_id=market_id,
            db=db,
            signal_engine=router,
            risk_manager=risk_manager,
            config=config,
        )
        with workers_lock:
            workers.append(worker)
        worker.start()
        logger.info("Started worker: %s (market_id=%d)", ticker, market_id)

    logger.info("All %d workers running.", len(workers))

    # ── Market rotator — rolls to the next 15m market on expiry ──────
    rotator = MarketRotator(
        client=client,
        db=db,
        signal_engine=router,
        risk_manager=risk_manager,
        config=config,
        workers=workers,
        workers_lock=workers_lock,
    )
    for m in markets:
        close_ts = m.get("close_ts", 0)
        if close_ts:
            rotator.register_market(m["ticker"], close_ts)
    rotator.start()
    logger.info("Market rotator started.")

    # ── Graceful shutdown ─────────────────────────────────────────────
    def shutdown(sig, frame) -> None:
        logger.info("Shutdown signal received. Stopping...")
        rotator.stop()
        poller.stop()
        with workers_lock:
            current = list(workers)
        for w in current:
            w.stop()
        btc_feed.stop()
        coinbase_spot_feed_module.get_instance().stop()
        for w in current:
            w.join(timeout=3.0)
        poller.join(timeout=5.0)
        rotator.join(timeout=5.0)
        db.end_session(session_id)
        logger.info("Shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        with workers_lock:
            alive = sum(1 for w in workers if w.is_alive())
        logger.debug("Workers alive: %d/%d", alive, len(workers))
        time.sleep(30)


if __name__ == "__main__":
    main()
