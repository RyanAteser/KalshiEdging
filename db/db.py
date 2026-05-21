"""
db.py — Database layer. Supports SQLite (default) and PostgreSQL via ENV.
All writes are synchronous and thread-safe via a single connection lock.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
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
        """Create all tables if they don't exist."""
        statements = [
            """
            CREATE TABLE IF NOT EXISTS markets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL UNIQUE,
                event       TEXT,
                created_at  TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ticks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id   INTEGER NOT NULL REFERENCES markets(id),
                timestamp   TEXT NOT NULL,
                best_bid    REAL,
                best_ask    REAL,
                last_price  REAL,
                volume      REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id   INTEGER NOT NULL REFERENCES markets(id),
                timestamp   TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                price       REAL NOT NULL,
                metadata    TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id   INTEGER NOT NULL REFERENCES markets(id),
                side        TEXT NOT NULL,
                price       REAL NOT NULL,
                quantity    INTEGER NOT NULL,
                timestamp   TEXT NOT NULL,
                pnl         REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id   INTEGER NOT NULL REFERENCES markets(id),
                entry_price REAL NOT NULL,
                quantity    INTEGER NOT NULL,
                status      TEXT NOT NULL DEFAULT 'OPEN',
                stop_loss   REAL NOT NULL,
                created_at  TEXT NOT NULL,
                closed_at   TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT NOT NULL,
                side             TEXT NOT NULL,
                threshold        REAL NOT NULL,
                entry_price      REAL NOT NULL,
                entry_ts         TEXT NOT NULL,
                exit_price       REAL,
                exit_ts          TEXT,
                exit_reason      TEXT,
                pnl_per_contract REAL,
                created_at       TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ev_feature_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id       INTEGER,
                ticker            TEXT NOT NULL,
                market_id         INTEGER,
                side              TEXT NOT NULL,
                entry_ts          REAL NOT NULL,
                exit_ts           REAL,
                entry_price       REAL NOT NULL,
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
                exit_reason       TEXT,
                pnl               REAL,
                outcome           INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS shadow_vol_trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT NOT NULL,
                side             TEXT NOT NULL,
                multiplier       REAL NOT NULL,
                entry_price      REAL NOT NULL,
                entry_ts         TEXT NOT NULL,
                exit_price       REAL,
                exit_ts          TEXT,
                exit_reason      TEXT,
                pnl_per_contract REAL,
                created_at       TEXT NOT NULL
            )
            """,
        ]
        # PostgreSQL uses SERIAL instead of AUTOINCREMENT
        if self._postgres:
            statements = [s.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY") for s in statements]

        for stmt in statements:
            self.execute(stmt.strip())

        logger.info("Database schema initialized.")

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def upsert_market(self, ticker: str, event: str) -> int:
        """Insert market if not exists, return its id."""
        row = self.fetchone("SELECT id FROM markets WHERE ticker = ?", (ticker,))
        if row:
            return row[0]
        self.execute(
            "INSERT INTO markets (ticker, event, created_at) VALUES (?, ?, ?)",
            (ticker, event, datetime.utcnow().isoformat()),
        )
        row = self.fetchone("SELECT id FROM markets WHERE ticker = ?", (ticker,))
        return row[0]

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
    ) -> None:
        self.execute(
            "INSERT INTO ticks (market_id, timestamp, best_bid, best_ask, last_price, volume) VALUES (?, ?, ?, ?, ?, ?)",
            (market_id, datetime.utcnow().isoformat(), best_bid, best_ask, last_price, volume),
        )

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def insert_signal(
        self,
        market_id: int,
        signal_type: str,
        price: float,
        metadata: Optional[dict] = None,
    ) -> None:
        self.execute(
            "INSERT INTO signals (market_id, timestamp, signal_type, price, metadata) VALUES (?, ?, ?, ?, ?)",
            (market_id, datetime.utcnow().isoformat(), signal_type, price, json.dumps(metadata or {})),
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
    ) -> int:
        self.execute(
            "INSERT INTO trades (market_id, side, price, quantity, timestamp, pnl) VALUES (?, ?, ?, ?, ?, ?)",
            (market_id, side, price, quantity, datetime.utcnow().isoformat(), pnl),
        )
        row = self.fetchone("SELECT last_insert_rowid()")
        return row[0]

    def update_trade_pnl(self, trade_id: int, pnl: float) -> None:
        self.execute("UPDATE trades SET pnl = ? WHERE id = ?", (pnl, trade_id))

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def open_position(
            self, market_id: int, entry_price: float, quantity: int, stop_loss: float,
    ) -> int:
        sql = """
        INSERT INTO positions (market_id, entry_price, quantity, status, stop_loss, created_at) 
        VALUES (?, ?, ?, 'OPEN', ?, ?)
    """
        params = (market_id, entry_price, quantity, stop_loss, datetime.utcnow().isoformat())

        with self._cursor() as cur:
            cur.execute(sql, params)
            if self._postgres:
                # Note: For Postgres, you'd ideally use "INSERT ... RETURNING id"
                # and cur.fetchone(), but as a quick fix:
                cur.execute("SELECT lastval()")
                return cur.fetchone()[0]
            else:
                return cur.lastrowid # sqlite3's built-in way to get the ID

    def close_position(self, position_id: int) -> None:
        self.execute(
            "UPDATE positions SET status = 'CLOSED', closed_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), position_id),
        )

    def get_open_position(self, market_id: int) -> Optional[Any]:
        return self.fetchone(
            "SELECT * FROM positions WHERE market_id = ? AND status = 'OPEN' LIMIT 1",
            (market_id,),
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

    # ------------------------------------------------------------------
    # EV feature log (ML training data)
    # ------------------------------------------------------------------

    def log_ev_entry(
        self,
        ticker: str,
        market_id: int,
        position_id: int,
        side: str,
        entry_price: float,
        features: dict,
    ) -> int:
        """
        Record a confirmed trade entry with all 10 EV feature values.
        Returns the ev_feature_log row id.
        """
        import time as _time
        sql = """
        INSERT INTO ev_feature_log (
            position_id, ticker, market_id, side, entry_ts, entry_price,
            base_p, delta_weight, delta_atr, ob_imbalance,
            cross_asset_boost, tf_confirm_boost, volume_boost,
            candle_boost, price_spike_boost, cvd_boost,
            p_model, ev
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            position_id, ticker, market_id, side, _time.time(), entry_price,
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
            features.get("p_model"),
            features.get("ev"),
        )
        with self._cursor() as cur:
            cur.execute(sql, params)
            if self._postgres:
                cur.execute("SELECT lastval()")
                return cur.fetchone()[0]
            return cur.lastrowid

    def close_ev_log(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ) -> None:
        """
        Update the ev_feature_log row for this position with exit data.
        outcome = 1 if pnl > 0, else 0 (binary label for ML classifier).
        """
        import time as _time
        outcome = 1 if pnl > 0 else 0
        self.execute(
            """UPDATE ev_feature_log
               SET exit_ts=?, exit_price=?, exit_reason=?, pnl=?, outcome=?
               WHERE position_id=?""",
            (_time.time(), exit_price, exit_reason, pnl, outcome, position_id),
        )
