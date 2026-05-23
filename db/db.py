"""
db.py — Database layer. Supports SQLite (default) and PostgreSQL via ENV.
All writes are synchronous and thread-safe via a single connection lock.
All timestamps stored as unix floats (REAL) via time.time().
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL", "sqlite:///kalshi_trader.db")


def _is_postgres() -> bool:
    return DB_URL.startswith("postgresql") or DB_URL.startswith("postgres")


class Database:
    """
    Thread-safe database wrapper. Supports SQLite and PostgreSQL.
    Uses a single shared connection with a reentrant lock for SQLite,
    or a connection-per-call pattern for PostgreSQL.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._postgres = _is_postgres()
        self._sqlite_conn: Optional[sqlite3.Connection] = None

        if not self._postgres:
            self._sqlite_conn = sqlite3.connect(
                DB_URL.replace("sqlite:///", ""),
                check_same_thread=False,
            )
            self._sqlite_conn.row_factory = sqlite3.Row
            self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
            self._sqlite_conn.execute("PRAGMA foreign_keys=ON")
            logger.info("SQLite database initialized at %s", DB_URL)
        else:
            try:
                import psycopg2  # type: ignore
                self._psycopg2 = psycopg2
                logger.info("PostgreSQL database initialized at %s", DB_URL)
            except ImportError:
                raise RuntimeError("psycopg2 not installed. Run: pip install psycopg2-binary")

    def get_market_id(self, ticker: str) -> Optional[int]:
        """Retrieve internal market ID for a given ticker."""
        row = self.fetchone("SELECT id FROM markets WHERE ticker = ?", (ticker,))
        return row[0] if row else None

    @contextmanager
    def _cursor(self) -> Generator[Any, None, None]:
        """Yields a cursor, handles commit/rollback, thread-safe."""
        with self._lock:
            if self._postgres:
                conn = self._psycopg2.connect(DB_URL)
                try:
                    cur = conn.cursor()
                    yield cur
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
            else:
                cur = self._sqlite_conn.cursor()
                try:
                    yield cur
                    self._sqlite_conn.commit()
                except Exception:
                    self._sqlite_conn.rollback()
                    raise

    def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a single write statement."""
        with self._cursor() as cur:
            cur.execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Any]:
        """Fetch a single row."""
        with self._cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[Any]:
        """Fetch all rows."""
        with self._cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        """Batch write."""
        with self._cursor() as cur:
            cur.executemany(sql, params_list)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create all tables if they don't exist, then run migrations."""
        statements = [
            """
            CREATE TABLE IF NOT EXISTS markets (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT NOT NULL UNIQUE,
                event            TEXT,
                series           TEXT,
                btc_target       REAL,
                open_ts          REAL,
                close_ts         REAL,
                result           INTEGER,
                settlement_price REAL,
                created_at       TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ticks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id  INTEGER NOT NULL REFERENCES markets(id),
                ts         REAL    NOT NULL,
                best_bid   REAL,
                best_ask   REAL,
                last_price REAL,
                volume     REAL,
                spread     REAL,
                mid        REAL,
                btc_price  REAL,
                cvd        REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id   INTEGER NOT NULL REFERENCES markets(id),
                ts          REAL    NOT NULL,
                signal_type TEXT    NOT NULL,
                side        TEXT,
                price       REAL    NOT NULL,
                p_model     REAL,
                ev          REAL,
                engine      TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS positions (
                id                         INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id                  INTEGER NOT NULL REFERENCES markets(id),
                side                       TEXT    NOT NULL DEFAULT 'YES',
                entry_price                REAL    NOT NULL,
                entry_ts                   REAL    NOT NULL,
                exit_price                 REAL,
                exit_ts                    REAL,
                quantity                   INTEGER NOT NULL,
                status                     TEXT    NOT NULL DEFAULT 'OPEN',
                stop_loss                  REAL    NOT NULL,
                exit_reason                TEXT,
                pnl                        REAL,
                outcome                    INTEGER,
                order_id                   TEXT,
                seconds_in_market_at_entry REAL,
                tick_count_at_entry        INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id   INTEGER NOT NULL REFERENCES markets(id),
                position_id INTEGER REFERENCES positions(id),
                side        TEXT    NOT NULL,
                kalshi_side TEXT,
                price       REAL    NOT NULL,
                quantity    INTEGER NOT NULL,
                ts          REAL    NOT NULL,
                pnl         REAL,
                order_id    TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ev_features (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id        INTEGER REFERENCES positions(id),
                ticker             TEXT    NOT NULL,
                market_id          INTEGER,
                side               TEXT    NOT NULL,
                entry_ts           REAL    NOT NULL,
                exit_ts            REAL,
                entry_price        REAL    NOT NULL,
                exit_price         REAL,
                spread_at_entry    REAL,
                btc_price_at_entry REAL,
                btc_target         REAL,
                btc_distance_pct   REAL,
                seconds_elapsed    REAL,
                seconds_remaining  REAL,
                tick_count         INTEGER,
                base_p             REAL,
                delta_weight       REAL,
                delta_atr          REAL,
                ob_imbalance       REAL,
                cross_asset_boost  REAL,
                tf_confirm_boost   REAL,
                volume_boost       REAL,
                candle_boost       REAL,
                price_spike_boost  REAL,
                cvd_boost          REAL,
                btc_distance       REAL,
                time_pressure      REAL,
                p_model            REAL,
                ev                 REAL,
                exit_reason        TEXT,
                pnl                REAL,
                outcome            INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS btc_candles (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL    NOT NULL UNIQUE,
                open      REAL    NOT NULL,
                high      REAL    NOT NULL,
                low       REAL    NOT NULL,
                close     REAL    NOT NULL,
                volume    REAL,
                timeframe TEXT    NOT NULL DEFAULT '15m'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS session_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                start_ts     REAL    NOT NULL,
                end_ts       REAL,
                ev_min_entry REAL,
                ev_grid_min  REAL,
                ev_grid_max  REAL,
                ev_fee_rate  REAL,
                paper_trade  INTEGER DEFAULT 1,
                trades       INTEGER DEFAULT 0,
                wins         INTEGER DEFAULT 0,
                losses       INTEGER DEFAULT 0,
                total_pnl    REAL    DEFAULT 0.0
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT    NOT NULL,
                side             TEXT    NOT NULL,
                threshold        REAL    NOT NULL,
                entry_price      REAL    NOT NULL,
                entry_ts         TEXT    NOT NULL,
                exit_price       REAL,
                exit_ts          TEXT,
                exit_reason      TEXT,
                pnl_per_contract REAL,
                created_at       TEXT    NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS shadow_vol_trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT    NOT NULL,
                side             TEXT    NOT NULL,
                multiplier       REAL    NOT NULL,
                entry_price      REAL    NOT NULL,
                entry_ts         TEXT    NOT NULL,
                exit_price       REAL,
                exit_ts          TEXT,
                exit_reason      TEXT,
                pnl_per_contract REAL,
                created_at       TEXT    NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ev_feature_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id       INTEGER,
                ticker            TEXT    NOT NULL,
                market_id         INTEGER,
                side              TEXT    NOT NULL,
                entry_ts          REAL    NOT NULL,
                exit_ts           REAL,
                entry_price       REAL    NOT NULL,
                exit_price        REAL,
                base_p            REAL,
                delta_weight      REAL,
                delta_atr         REAL,
                ob_imbalance      REAL,
                cross_asset_boost REAL,
                tf_confirm_boost  REAL,
                volume_boost      REAL,
                candle_boost      REAL,
                price_spike_boost REAL,
                cvd_boost         REAL,
                p_model           REAL,
                ev                REAL,
                btc_distance      REAL,
                time_pressure     REAL,
                exit_reason       TEXT,
                pnl               REAL,
                outcome           INTEGER
            )
            """,
        ]

        # PostgreSQL uses SERIAL instead of AUTOINCREMENT
        if self._postgres:
            statements = [
                s.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
                for s in statements
            ]

        for stmt in statements:
            self.execute(stmt.strip())

        # ── Migrations: add new columns to pre-existing tables ────────
        # Markets new cols
        for col in (
            "series TEXT",
            "btc_target REAL",
            "open_ts REAL",
            "close_ts REAL",
            "result INTEGER",
            "settlement_price REAL",
        ):
            try:
                self.execute(f"ALTER TABLE markets ADD COLUMN {col}")
            except Exception:
                pass  # column already exists

        # Ticks new cols (ts is the new unix-float column; old tables had 'timestamp TEXT')
        for col in ("ts REAL", "spread REAL", "mid REAL", "btc_price REAL", "cvd REAL"):
            try:
                self.execute(f"ALTER TABLE ticks ADD COLUMN {col}")
            except Exception:
                pass

        # Signals new cols
        for col in ("side TEXT", "p_model REAL", "ev REAL", "engine TEXT"):
            try:
                self.execute(f"ALTER TABLE signals ADD COLUMN {col}")
            except Exception:
                pass

        # Positions new cols
        for col in (
            "side TEXT",
            "entry_ts REAL",
            "exit_price REAL",
            "exit_ts REAL",
            "exit_reason TEXT",
            "pnl REAL",
            "outcome INTEGER",
            "order_id TEXT",
            "seconds_in_market_at_entry REAL",
            "tick_count_at_entry INTEGER",
        ):
            try:
                self.execute(f"ALTER TABLE positions ADD COLUMN {col}")
            except Exception:
                pass

        # Trades new cols
        for col in ("position_id INTEGER", "kalshi_side TEXT", "order_id TEXT"):
            try:
                self.execute(f"ALTER TABLE trades ADD COLUMN {col}")
            except Exception:
                pass

        # Legacy ev_feature_log migrations
        for col in ("btc_distance REAL", "time_pressure REAL"):
            try:
                self.execute(f"ALTER TABLE ev_feature_log ADD COLUMN {col}")
            except Exception:
                pass

        # Index on ticks.ts — only valid once the ts column exists
        try:
            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_ticks_market_ts ON ticks(market_id, ts)"
            )
        except Exception:
            pass  # ts column not yet present on very old databases

        logger.info("Database schema initialized.")

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def upsert_market(
        self,
        ticker: str,
        event: str,
        *,
        btc_target: Optional[float] = None,
        open_ts: Optional[float] = None,
        close_ts: Optional[float] = None,
    ) -> int:
        """Insert market if not exists, return its id.
        If market already exists, update btc_target/close_ts if provided."""
        row = self.fetchone("SELECT id FROM markets WHERE ticker = ?", (ticker,))
        if row:
            market_id = row[0]
            # Update optional fields if provided
            if btc_target is not None:
                self.execute(
                    "UPDATE markets SET btc_target = ? WHERE id = ?",
                    (btc_target, market_id),
                )
            if close_ts is not None:
                self.execute(
                    "UPDATE markets SET close_ts = ? WHERE id = ?",
                    (close_ts, market_id),
                )
            return market_id

        self.execute(
            """INSERT INTO markets (ticker, event, btc_target, open_ts, close_ts, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticker, event, btc_target, open_ts, close_ts, datetime.utcnow().isoformat()),
        )
        row = self.fetchone("SELECT id FROM markets WHERE ticker = ?", (ticker,))
        return row[0]

    def update_market_result(
        self, ticker: str, result: int, settlement_price: Optional[float] = None
    ) -> None:
        """Record final market result and settlement price."""
        self.execute(
            "UPDATE markets SET result = ?, settlement_price = ? WHERE ticker = ?",
            (result, settlement_price, ticker),
        )

    # ------------------------------------------------------------------
    # Ticks
    # ------------------------------------------------------------------

    def insert_tick(
        self,
        market_id: int,
        best_bid: Optional[float],
        best_ask: Optional[float],
        last_price: Optional[float],
        volume: Optional[float],
        btc_price: Optional[float] = None,
        cvd: Optional[float] = None,
    ) -> None:
        spread = None
        mid    = None
        if best_bid is not None and best_ask is not None:
            spread = round(best_ask - best_bid, 6)
            mid    = (best_ask + best_bid) / 2.0

        self.execute(
            """INSERT INTO ticks
               (market_id, ts, best_bid, best_ask, last_price, volume, spread, mid, btc_price, cvd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (market_id, time.time(), best_bid, best_ask, last_price, volume,
             spread, mid, btc_price, cvd),
        )

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def insert_signal(
        self,
        market_id: int,
        signal_type: str,
        price: float,
        *,
        side: Optional[str] = None,
        p_model: Optional[float] = None,
        ev: Optional[float] = None,
        engine: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert a signal row. metadata param kept for backward compat but ignored."""
        # Extract structured fields from metadata if callers still pass it the old way
        if metadata and side is None:
            side = metadata.get("side")
        if metadata and p_model is None:
            p_model = metadata.get("p_model")
        if metadata and ev is None:
            ev = metadata.get("ev")
        if metadata and engine is None:
            engine = metadata.get("engine")

        self.execute(
            """INSERT INTO signals (market_id, ts, signal_type, side, price, p_model, ev, engine)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (market_id, time.time(), signal_type, side, price, p_model, ev, engine),
        )

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def insert_trade(
        self,
        market_id: int,
        side: str,
        price: float,
        quantity: int,
        pnl: Optional[float] = None,
        *,
        position_id: Optional[int] = None,
        kalshi_side: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO trades
                   (market_id, position_id, side, kalshi_side, price, quantity, ts, pnl, order_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (market_id, position_id, side, kalshi_side, price, quantity,
                 time.time(), pnl, order_id),
            )
            if self._postgres:
                cur.execute("SELECT lastval()")
                return cur.fetchone()[0]
            return cur.lastrowid

    def update_trade_pnl(self, trade_id: int, pnl: float) -> None:
        self.execute("UPDATE trades SET pnl = ? WHERE id = ?", (pnl, trade_id))

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def open_position(
        self,
        market_id: int,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        *,
        side: str = "YES",
        seconds_in_market: Optional[float] = None,
        tick_count: Optional[int] = None,
        order_id: Optional[str] = None,
    ) -> int:
        sql = """
        INSERT INTO positions
            (market_id, side, entry_price, entry_ts, quantity, status, stop_loss,
             seconds_in_market_at_entry, tick_count_at_entry, order_id)
        VALUES (?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
        """
        params = (
            market_id, side, entry_price, time.time(), quantity, stop_loss,
            seconds_in_market, tick_count, order_id,
        )
        with self._cursor() as cur:
            cur.execute(sql, params)
            if self._postgres:
                cur.execute("SELECT lastval()")
                return cur.fetchone()[0]
            return cur.lastrowid

    def close_position(
        self,
        position_id: int,
        *,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        pnl: Optional[float] = None,
    ) -> None:
        outcome = None
        if pnl is not None:
            outcome = 1 if pnl > 0 else 0

        self.execute(
            """UPDATE positions
               SET status = 'CLOSED', exit_ts = ?, exit_price = ?,
                   exit_reason = ?, pnl = ?, outcome = ?
               WHERE id = ?""",
            (time.time(), exit_price, exit_reason, pnl, outcome, position_id),
        )

    def get_open_position(self, market_id: int) -> Optional[dict]:
        """Return open position as a dict, or None if not found."""
        row = self.fetchone(
            """SELECT id, market_id, side, entry_price, quantity, stop_loss,
                      entry_ts, tick_count_at_entry
               FROM positions
               WHERE market_id = ? AND status = 'OPEN'
               LIMIT 1""",
            (market_id,),
        )
        if row is None:
            return None
        return {
            "id":                 row[0],
            "market_id":         row[1],
            "side":               row[2],
            "entry_price":       row[3],
            "quantity":          row[4],
            "stop_loss":         row[5],
            "entry_ts":          row[6],
            "tick_count_at_entry": row[7],
        }

    # ------------------------------------------------------------------
    # EV features (primary ML training table)
    # ------------------------------------------------------------------

    def log_ev_features(
        self,
        ticker: str,
        market_id: int,
        position_id: int,
        side: str,
        entry_price: float,
        features: dict,
        *,
        btc_price: Optional[float] = None,
    ) -> int:
        """
        Record a confirmed trade entry with all EV feature values.
        Writes to the new ev_features table.
        Returns the row id.
        """
        # Compute btc_distance_pct if possible
        btc_target = features.get("btc_target")
        btc_distance_pct = None
        if btc_price is not None and btc_target and btc_target > 0:
            btc_distance_pct = (btc_price - btc_target) / btc_target

        sql = """
        INSERT INTO ev_features (
            position_id, ticker, market_id, side, entry_ts, entry_price,
            spread_at_entry, btc_price_at_entry, btc_target, btc_distance_pct,
            seconds_elapsed, seconds_remaining, tick_count,
            base_p, delta_weight, delta_atr, ob_imbalance,
            cross_asset_boost, tf_confirm_boost, volume_boost,
            candle_boost, price_spike_boost, cvd_boost,
            btc_distance, time_pressure, p_model, ev
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?
        )
        """
        params = (
            position_id, ticker, market_id, side, time.time(), entry_price,
            features.get("spread"),
            btc_price,
            btc_target,
            btc_distance_pct,
            features.get("seconds_elapsed"),
            features.get("seconds_remaining"),
            features.get("tick_count"),
            features.get("base_p"),
            features.get("delta_weight"),
            features.get("delta_atr"),
            features.get("ob_imbalance"),
            features.get("cross_asset_boost"),
            features.get("tf_confirm_boost"),
            features.get("volume_boost"),
            features.get("candle_boost"),
            features.get("price_spike_boost"),
            features.get("cvd_boost"),
            features.get("btc_distance"),
            features.get("time_pressure"),
            features.get("p_model"),
            features.get("ev"),
        )
        with self._cursor() as cur:
            cur.execute(sql, params)
            if self._postgres:
                cur.execute("SELECT lastval()")
                return cur.fetchone()[0]
            return cur.lastrowid

    def close_ev_features(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ) -> None:
        """Update the ev_features row for this position with exit data."""
        outcome = 1 if pnl > 0 else 0
        self.execute(
            """UPDATE ev_features
               SET exit_ts = ?, exit_price = ?, exit_reason = ?, pnl = ?, outcome = ?
               WHERE position_id = ?""",
            (time.time(), exit_price, exit_reason, pnl, outcome, position_id),
        )

    # ── Backward compat aliases ───────────────────────────────────────

    def log_ev_entry(
        self,
        ticker: str,
        market_id: int,
        position_id: int,
        side: str,
        entry_price: float,
        features: dict,
    ) -> int:
        """Backward compat alias for log_ev_features (no btc_price param)."""
        return self.log_ev_features(
            ticker, market_id, position_id, side, entry_price, features
        )

    def close_ev_log(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ) -> None:
        """Backward compat alias for close_ev_features."""
        self.close_ev_features(position_id, exit_price, exit_reason, pnl)

    # ------------------------------------------------------------------
    # BTC candles
    # ------------------------------------------------------------------

    def insert_btc_candle(
        self,
        ts: float,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: Optional[float] = None,
        timeframe: str = "15m",
    ) -> None:
        """Insert a BTC candle row; silently ignores duplicates (ts is UNIQUE)."""
        self.execute(
            """INSERT OR IGNORE INTO btc_candles (ts, open, high, low, close, volume, timeframe)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ts, open, high, low, close, volume, timeframe),
        )

    # ------------------------------------------------------------------
    # Session log
    # ------------------------------------------------------------------

    def start_session(self, config) -> int:
        """Record a new bot session. Returns session id."""
        sql = """
        INSERT INTO session_log
            (start_ts, ev_min_entry, ev_grid_min, ev_grid_max, ev_fee_rate, paper_trade)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            time.time(),
            getattr(config, "ev_min_entry", None),
            getattr(config, "ev_grid_min", None),
            getattr(config, "ev_grid_max", None),
            getattr(config, "ev_fee_rate", None),
            1 if getattr(config, "paper_trade", True) else 0,
        )
        with self._cursor() as cur:
            cur.execute(sql, params)
            if self._postgres:
                cur.execute("SELECT lastval()")
                return cur.fetchone()[0]
            return cur.lastrowid

    def end_session(
        self,
        session_id: int,
        trades: int = 0,
        wins: int = 0,
        losses: int = 0,
        total_pnl: float = 0.0,
    ) -> None:
        """Update session record with final stats."""
        self.execute(
            """UPDATE session_log
               SET end_ts = ?, trades = ?, wins = ?, losses = ?, total_pnl = ?
               WHERE id = ?""",
            (time.time(), trades, wins, losses, total_pnl, session_id),
        )

    # ------------------------------------------------------------------
    # Shadow trades
    # ------------------------------------------------------------------

    def open_shadow_position(
        self, ticker: str, side: str, entry_price: float, threshold: float
    ) -> int:
        now = datetime.utcnow().isoformat()
        sql = """
        INSERT INTO shadow_trades (ticker, side, threshold, entry_price, entry_ts, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._cursor() as cur:
            cur.execute(sql, (ticker, side, threshold, entry_price, now, now))
            if self._postgres:
                cur.execute("SELECT lastval()")
                return cur.fetchone()[0]
            return cur.lastrowid

    def close_shadow_position(
        self, row_id: int, exit_price: float, reason: str, pnl: float
    ) -> None:
        self.execute(
            """UPDATE shadow_trades
               SET exit_price=?, exit_ts=?, exit_reason=?, pnl_per_contract=?
               WHERE id=?""",
            (exit_price, datetime.utcnow().isoformat(), reason, pnl, row_id),
        )

    def get_open_shadow_positions(self, ticker: str) -> list:
        return self.fetchall(
            "SELECT id, threshold, entry_price, side FROM shadow_trades"
            " WHERE ticker=? AND exit_price IS NULL",
            (ticker,),
        )

    # ------------------------------------------------------------------
    # Shadow vol trades
    # ------------------------------------------------------------------

    def open_shadow_vol_position(
        self, ticker: str, side: str, entry_price: float, multiplier: float
    ) -> int:
        now = datetime.utcnow().isoformat()
        sql = """
        INSERT INTO shadow_vol_trades (ticker, side, multiplier, entry_price, entry_ts, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._cursor() as cur:
            cur.execute(sql, (ticker, side, multiplier, entry_price, now, now))
            if self._postgres:
                cur.execute("SELECT lastval()")
                return cur.fetchone()[0]
            return cur.lastrowid

    def close_shadow_vol_position(
        self, row_id: int, exit_price: float, reason: str, pnl: float
    ) -> None:
        self.execute(
            """UPDATE shadow_vol_trades
               SET exit_price=?, exit_ts=?, exit_reason=?, pnl_per_contract=?
               WHERE id=?""",
            (exit_price, datetime.utcnow().isoformat(), reason, pnl, row_id),
        )

    def get_open_shadow_vol_positions(self, ticker: str) -> list:
        return self.fetchall(
            "SELECT id, multiplier, entry_price, side FROM shadow_vol_trades"
            " WHERE ticker=? AND exit_price IS NULL",
            (ticker,),
        )
