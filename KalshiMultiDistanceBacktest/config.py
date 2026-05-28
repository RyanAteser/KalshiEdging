import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

KALSHI_API_KEY_ID       = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "keys/kalshi_private.pem")
KALSHI_ENV              = os.getenv("KALSHI_ENV", "prod")

# ── Per-asset configuration ───────────────────────────────────────────────────
# kalshi_series:    Kalshi market series ticker — VERIFY against live API
#                   pattern: KX{SYMBOL}15M
# coinbase_pair:    Coinbase Exchange candles product ID
# binance_symbol:   Binance klines symbol (fallback)
# kraken_pair:      Kraken OHLC pair (fallback)
# annual_vol:       Annualised volatility for GBM probability model
# min_strike:       Minimum plausible strike value — used to filter bad parses
# thresholds:       Entry distance thresholds to sweep (in USD)
# stop_dist:        Stop-loss distance (exit when dist drops below this)
# price_file:       Filename for cached 1m OHLCV parquet

ASSETS = {
    "ETH": {
        "kalshi_series":  "KXETH15M",
        "coinbase_pair":  "ETH-USD",
        "binance_symbol": "ETHUSDT",
        "kraken_pair":    "ETHUSD",
        "annual_vol":     0.80,
        "min_strike":     100.0,
        "thresholds":     [5, 8, 10, 12, 15, 20, 25, 30],
        "stop_dist":      2.0,
        "price_file":     "prices_eth_1m.parquet",
    },
    "SOL": {
        "kalshi_series":  "KXSOL15M",
        "coinbase_pair":  "SOL-USD",
        "binance_symbol": "SOLUSDT",
        "kraken_pair":    "SOLUSD",
        "annual_vol":     1.10,
        "min_strike":     5.0,
        "thresholds":     [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50],
        "stop_dist":      0.02,
        "price_file":     "prices_sol_1m.parquet",
    },
    "XRP": {
        "kalshi_series":  "KXXRP15M",
        "coinbase_pair":  "XRP-USD",
        "binance_symbol": "XRPUSDT",
        "kraken_pair":    "XRPUSD",
        "annual_vol":     0.90,
        "min_strike":     0.05,
        "thresholds":     [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.013],
        "stop_dist":      0.0003,
        "price_file":     "prices_xrp_1m.parquet",
    },
    "DOGE": {
        "kalshi_series":  "KXDOGE15M",
        "coinbase_pair":  "DOGE-USD",
        "binance_symbol": "DOGEUSDT",
        "kraken_pair":    "XDGUSD",
        "annual_vol":     1.20,
        "min_strike":     0.005,
        "thresholds":     [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0007, 0.001],
        "stop_dist":      0.00003,
        "price_file":     "prices_doge_1m.parquet",
    },
    "BNB": {
        "kalshi_series":  "KXBNB15M",
        "coinbase_pair":  "BNB-USD",      # may not be listed; falls back to Binance
        "binance_symbol": "BNBUSDT",
        "kraken_pair":    "BNBUSD",       # may not be listed on Kraken
        "annual_vol":     0.80,
        "min_strike":     10.0,
        "thresholds":     [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0],
        "stop_dist":      0.50,
        "price_file":     "prices_bnb_1m.parquet",
    },
    "HYPE": {
        "kalshi_series":  "KXHYPE15M",
        "coinbase_pair":  "HYPE-USD",     # may not be listed; falls back to Binance
        "binance_symbol": "HYPEUSDT",
        "kraken_pair":    "HYPEUSD",      # likely unavailable
        "annual_vol":     2.00,
        "min_strike":     1.0,
        "thresholds":     [0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50],
        "stop_dist":      0.05,
        "price_file":     "prices_hype_1m.parquet",
    },
}

# Which assets to run (can be overridden via env: ASSETS=ETH,SOL)
_env_assets = os.getenv("ASSETS", "")
ENABLED_ASSETS = [a.strip().upper() for a in _env_assets.split(",") if a.strip()] \
                 or list(ASSETS.keys())

DAYS = 90
