"""
z_score.py — Dynamic z-score gate using realized BTC volatility.

Core formula (log-normal / GBM):
    returns = diff(log(prices))
    sigma   = EWMA std of returns / sqrt(tick_interval)   # per-second vol
    z       = log(btc_price / strike) / (sigma * sqrt(t_remaining_sec))

Interpretation:
    z = 3.5  →  P(loss) ≈ 0.02%  under current realized vol
    z = 4.0  →  P(loss) ≈ 0.003%
    z = 5.0  →  P(loss) ≈ essentially zero

Why EWMA instead of simple std:
    Simple std weights a 3-minute-old price move equally with one from
    2 seconds ago. EWMA decays older returns exponentially, making the
    vol estimate respond faster to regime changes (e.g., BTC suddenly
    going quiet or suddenly going chaotic).

Why a sigma floor:
    If BTC goes dead calm for 60 seconds, realized vol collapses and
    every contract looks like z=10. The floor (min_sigma) prevents
    false certainty during artificially quiet periods.
"""

from __future__ import annotations

import numpy as np

# Tunable constants
Z_MIN_THRESHOLD   = 3.5       # minimum z to allow any entry
EWMA_DECAY        = 0.05      # higher = more weight on recent returns
MIN_SIGMA_FLOOR   = 0.00015   # ~0.015%/sec ≈ 90% annualised vol floor
MIN_HISTORY_LEN   = 20        # need at least 20 prices to compute vol
TICK_INTERVAL_SEC = 1.0       # seconds between BTC price samples


def compute_z_score(
    btc_price: float,
    strike: float,
    t_remaining_sec: float,
    price_history: list[float],
    tick_interval_sec: float = TICK_INTERVAL_SEC,
    min_sigma: float = MIN_SIGMA_FLOOR,
    ewma_decay: float = EWMA_DECAY,
) -> tuple[float, float]:
    """
    Returns (z_score, realized_vol_per_sec).
    Returns (0.0, min_sigma) if insufficient history.

    z_score > Z_MIN_THRESHOLD is required for entry.
    Log realized_vol_per_sec to DB on every trade for calibration.
    """
    if len(price_history) < MIN_HISTORY_LEN:
        return 0.0, min_sigma

    prices  = np.array(price_history, dtype=float)
    returns = np.diff(np.log(prices))

    # EWMA variance — most recent returns weighted highest
    n       = len(returns)
    weights = np.exp(-ewma_decay * np.arange(n)[::-1])
    weights /= weights.sum()
    ewma_var = float(np.sum(weights * returns ** 2)) / tick_interval_sec

    realized_vol = max(float(np.sqrt(ewma_var)), min_sigma)

    # Log-normal z-score (driftless)
    log_dist = float(np.log(btc_price / strike))
    if t_remaining_sec <= 0 or realized_vol <= 0:
        return 0.0, realized_vol

    z = log_dist / (realized_vol * float(np.sqrt(t_remaining_sec)))
    return z, realized_vol


def p_win_from_z(z: float) -> float:
    """
    Theoretical win probability from z-score under log-normal assumption.
    Note: BTC has fat tails — treat this as an optimistic upper bound.
    Real-world P(win) at z=3.5 is closer to 99.7–99.9%, not 99.98%.
    """
    from scipy.stats import norm
    return float(norm.cdf(z))
