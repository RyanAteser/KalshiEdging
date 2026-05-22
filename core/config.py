"""
config.py — All configuration loaded from environment variables.
Never hardcode secrets. Use .env file + python-dotenv for local dev.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Kalshi API
    kalshi_api_key_id: str
    kalshi_private_key_path: str

    # Database
    database_url: str

    # Legacy strategy params (kept for any remaining references)
    entry_threshold: float
    stop_loss: float
    max_spread: float
    min_liquidity_dollars: float
    position_size: int

    # EV Grid Filter strategy
    ev_grid_min:  float   # lower price bound for entry (e.g. 0.50)
    ev_grid_max:  float   # upper price bound for entry (e.g. 0.80)
    ev_min_entry: float   # minimum EV to open a position (e.g. 0.005)
    ev_min_exit:  float   # auto-exit when EV drops below this (e.g. -0.003)
    ev_fee_rate:  float   # Kalshi fee approximation for EV formula (e.g. 0.007)

    # Concurrency
    max_markets: int             # max simultaneous market workers
    worker_restart_delay: float  # seconds before restarting crashed worker

    # Misc
    log_level: str
    paper_trade: bool            # if True, skip real order placement


def load_config() -> Config:
    """Load and validate config from environment. Raises on missing required keys."""

    def require(key: str) -> str:
        val = os.getenv(key)
        if not val:
            raise EnvironmentError(f"Required environment variable '{key}' is not set.")
        return val

    def get_float(key: str, default: float) -> float:
        return float(os.getenv(key, str(default)))

    def get_int(key: str, default: int) -> int:
        return int(os.getenv(key, str(default)))

    def get_bool(key: str, default: bool) -> bool:
        return os.getenv(key, str(default)).lower() in ("1", "true", "yes")

    return Config(
        kalshi_api_key_id=require("KALSHI_API_KEY_ID"),
        kalshi_private_key_path=require("KALSHI_PRIVATE_KEY_PATH"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///kalshi_trader.db"),
        entry_threshold=get_float("ENTRY_THRESHOLD", 0.50),
        stop_loss=get_float("STOP_LOSS", 0.48),
        max_spread=get_float("MAX_SPREAD", 0.10),
        min_liquidity_dollars=get_float("MIN_LIQUIDITY_DOLLARS", 5.0),
        position_size=get_int("POSITION_SIZE", 1),
        ev_grid_min=get_float("EV_GRID_MIN", 0.50),
        ev_grid_max=get_float("EV_GRID_MAX", 0.80),
        ev_min_entry=get_float("EV_MIN_ENTRY", 0.005),
        ev_min_exit=get_float("EV_MIN_EXIT", -0.003),
        ev_fee_rate=get_float("EV_FEE_RATE", 0.007),
        max_markets=get_int("MAX_MARKETS", 10),
        worker_restart_delay=get_float("WORKER_RESTART_DELAY", 5.0),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        paper_trade=get_bool("PAPER_TRADE", True),
    )
