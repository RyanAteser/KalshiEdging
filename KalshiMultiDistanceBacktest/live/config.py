"""
Live trader configuration — tune these before going live.
"""

# ── Entry conditions ──────────────────────────────────────────────────────────
Z_THRESHOLD    = 1.0    # enter YES when z > Z_THRESHOLD, NO when z < -Z_THRESHOLD
T_ENTER_MAX    = 89     # only enter when t_left ≤ this many seconds
T_ENTER_MIN    = 5      # never enter with < 5s left (no liquidity)

# ── Position sizing ───────────────────────────────────────────────────────────
CONTRACTS      = 100    # contracts per trade (100 contracts @ 95c = $95 risked, ~$4-9 profit)
MAX_OPEN       = 5      # max simultaneous open positions

# ── Fee model ─────────────────────────────────────────────────────────────────
FEE_RATE       = 0.07   # 7% of profit per side
FEE_CAP        = 0.07   # capped at $0.07 per contract

# ── Volatility / z-score ──────────────────────────────────────────────────────
BTC_HISTORY_LEN  = 30   # seconds of BTC price history for EWMA vol
EWMA_DECAY       = 0.05
MIN_SIGMA_FLOOR  = 0.00015

# ── Market scanning ───────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC  = 5    # how often to scan Kalshi for new markets (seconds)
PRICE_INTERVAL_SEC = 1    # how often to refresh BTC price

# ── Kalshi series to trade ────────────────────────────────────────────────────
# These are the up/down 15-minute markets
KALSHI_SERIES = "KXBTC15M"

# ── Safety ───────────────────────────────────────────────────────────────────
PAPER_MODE     = True   # set False ONLY when ready for real money
MAX_DAILY_LOSS = 50.0   # stop trading if daily PnL drops below -$50
