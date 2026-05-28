"""
dsr.py — Differential Sharpe Ratio reward signal.

The DSR gives a dense per-trade reward that approximates the marginal
contribution of each new return to the running Sharpe ratio.

Reference: Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement"

Formula:
    A_t = η·r_t + (1−η)·A_{t-1}       (EWM of returns)
    B_t = η·r_t² + (1−η)·B_{t-1}      (EWM of squared returns)
    DSR_t = (A_{t-1}·r_t − 0.5·B_{t-1}·r_t²) / (B_{t-1} − A_{t-1}²)^{3/2}

Usage:
    dsr = DSR(eta=0.01)
    reward = dsr.update(pnl_cents / 100.0)   # pass normalised return
    dsr.reset()                               # call between episodes
"""

from __future__ import annotations
import math


class DSR:
    def __init__(self, eta: float = 0.01, eps: float = 1e-8):
        self.eta  = eta
        self.eps  = eps
        self.A    = 0.0   # EWM of returns
        self.B    = 0.0   # EWM of squared returns

    def reset(self) -> None:
        self.A = 0.0
        self.B = 0.0

    def update(self, r: float) -> float:
        """Feed one completed-trade return, get DSR reward back."""
        A_prev = self.A
        B_prev = self.B

        self.A = self.eta * r       + (1 - self.eta) * A_prev
        self.B = self.eta * r ** 2  + (1 - self.eta) * B_prev

        denom = (B_prev - A_prev ** 2) ** 1.5
        if denom < self.eps:
            return 0.0

        return (A_prev * r - 0.5 * B_prev * r ** 2) / denom

    @property
    def sharpe(self) -> float:
        """Current running Sharpe estimate."""
        var = self.B - self.A ** 2
        if var < self.eps:
            return 0.0
        return self.A / math.sqrt(var)
