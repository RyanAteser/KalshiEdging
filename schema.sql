-- kalshi_trader schema v2
-- All timestamps stored as unix floats (REAL) for easy time arithmetic.
-- SQLite: AUTOINCREMENT. PostgreSQL: substitute SERIAL PRIMARY KEY.

-- ── Markets ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS markets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT NOT NULL UNIQUE,
    event            TEXT,
    series           TEXT,
    btc_target       REAL,
    open_ts          REAL,
    close_ts         REAL,
    result           INTEGER,       -- 1=YES won, 0=NO won, NULL=pending/unknown
    settlement_price REAL,          -- 0.0 or 1.0 after settlement
    created_at       TEXT NOT NULL
);

-- ── Ticks ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ticks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id  INTEGER NOT NULL REFERENCES markets(id),
    ts         REAL    NOT NULL,    -- unix float
    best_bid   REAL,
    best_ask   REAL,
    last_price REAL,
    volume     REAL,
    spread     REAL,                -- best_ask - best_bid
    mid        REAL,                -- (best_ask + best_bid) / 2
    btc_price  REAL,                -- Coinbase spot mid at tick time
    cvd        REAL                 -- Coinbase cumulative volume delta
);

CREATE INDEX IF NOT EXISTS idx_ticks_market_ts ON ticks(market_id, ts);

-- ── Signals ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id   INTEGER NOT NULL REFERENCES markets(id),
    ts          REAL    NOT NULL,
    signal_type TEXT    NOT NULL,   -- ENTRY | EXIT | STOP_LOSS
    side        TEXT,               -- YES | NO
    price       REAL    NOT NULL,
    p_model     REAL,
    ev          REAL,
    engine      TEXT                -- ev_grid
);

-- ── Positions ─────────────────────────────────────────────────────────
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
    exit_reason                TEXT,   -- settlement | ev_flip | stop_loss | market_closed
    pnl                        REAL,
    outcome                    INTEGER,    -- 1=win 0=loss NULL=open
    order_id                   TEXT,
    seconds_in_market_at_entry REAL,
    tick_count_at_entry        INTEGER
);

-- ── Trades ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id   INTEGER NOT NULL REFERENCES markets(id),
    position_id INTEGER REFERENCES positions(id),
    side        TEXT    NOT NULL,   -- BUY | SELL
    kalshi_side TEXT,               -- YES | NO
    price       REAL    NOT NULL,
    quantity    INTEGER NOT NULL,
    ts          REAL    NOT NULL,
    pnl         REAL,
    order_id    TEXT
);

-- ── EV features (primary ML training table) ───────────────────────────
-- One row per real trade entry. Outcome/pnl filled in at settlement.
CREATE TABLE IF NOT EXISTS ev_features (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id        INTEGER REFERENCES positions(id),
    ticker             TEXT    NOT NULL,
    market_id          INTEGER,
    side               TEXT    NOT NULL,
    entry_ts           REAL    NOT NULL,
    exit_ts            REAL,

    -- Entry market context
    entry_price        REAL    NOT NULL,
    exit_price         REAL,
    spread_at_entry    REAL,        -- ask - bid at signal time
    btc_price_at_entry REAL,        -- Coinbase spot mid at signal time
    btc_target         REAL,        -- market strike price
    btc_distance_pct   REAL,        -- (btc_price - btc_target) / btc_target (raw %)
    seconds_elapsed    REAL,        -- seconds since market open when signal fired
    seconds_remaining  REAL,        -- seconds until market close when signal fired
    tick_count         INTEGER,     -- ticks this market had seen before entry

    -- Signal engine features
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
    btc_distance       REAL,        -- capped feature value used in p_model
    time_pressure      REAL,        -- capped feature value used in p_model
    p_model            REAL,
    ev                 REAL,

    -- Outcome (filled at close)
    exit_reason        TEXT,
    pnl                REAL,
    outcome            INTEGER      -- 1=win 0=loss (binary ML label)
);

-- ── BTC candles ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS btc_candles (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL    NOT NULL UNIQUE,  -- candle open timestamp (unix)
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL,
    timeframe TEXT    NOT NULL DEFAULT '15m'
);

-- ── Session log ───────────────────────────────────────────────────────
-- One row per bot run. Lets you correlate performance to config changes.
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
);

-- ── Legacy tables (kept for backward compat, no longer written to) ────
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
);

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
);

-- ev_feature_log kept for old data — new writes go to ev_features
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
);
