"""
ml/predictor.py — Load trained model and produce p_model from live tick state.

Drop-in replacement for the hand-crafted _compute_p_model() in signal_engine_ev.py.
The bot calls MLPredictor.predict() with a snapshot of current market state and
gets back a calibrated P(Up) between 0 and 1.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parent / "btc_15m_model.pkl"


class MLPredictor:
    """
    Wraps the trained XGBoost model.

    Usage:
        predictor = MLPredictor()          # loads ml/btc_15m_model.pkl
        p = predictor.predict(state_dict)  # returns float in [0, 1]
    """

    def __init__(self, model_path: str | Path = _DEFAULT_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run: python ml/train.py --prices <path-to-zip>"
            )
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        self._model        = artifact["model"]
        self._feature_cols = artifact["feature_cols"]
        self._entry_window = artifact.get("entry_window", 300)
        self._snapshots    = artifact.get("snapshots", [0, 30, 60, 120, 180, 300])
        logger.info("MLPredictor loaded: %d features", len(self._feature_cols))

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, state: "MLState") -> float:
        """
        Given an MLState (accumulated per-tick data), return P(Up).

        Returns 0.5 (neutral) if not enough ticks yet.
        """
        if len(state.micro_history) < 5:
            return 0.5   # not enough data yet

        feat_dict = state.to_feature_dict(self._snapshots)
        row = self._build_row(feat_dict)

        try:
            proba = self._model.predict_proba(row)[0, 1]
            return float(np.clip(proba, 0.01, 0.99))
        except Exception as exc:
            logger.warning("MLPredictor.predict failed: %s — returning 0.5", exc)
            return 0.5

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_row(self, feat_dict: dict) -> np.ndarray:
        """Convert feature dict to model input row, filling missing with 0."""
        row = np.array(
            [float(feat_dict.get(col, 0.0) or 0.0) for col in self._feature_cols],
            dtype=np.float32,
        )
        return row.reshape(1, -1)


class MLState:
    """
    Accumulates per-tick data for one market so MLPredictor can extract features.

    Call update() on every tick. The predictor reads this object.
    """

    def __init__(self) -> None:
        self.micro_history: list[float] = []   # up_microprice per tick
        self.obi_history:   list[float] = []   # up_ob_imbalance per tick
        self.bid_history:   list[float] = []
        self.ask_history:   list[float] = []
        self.depth_up:      list[float] = []
        self.depth_down:    list[float] = []
        self._tick = 0

    def update(
        self,
        best_bid: Optional[float],
        best_ask: Optional[float],
        ob_imbalance: Optional[float] = None,
        sum_bid_size: Optional[float] = None,
        sum_ask_size: Optional[float] = None,
    ) -> None:
        if best_bid is None or best_ask is None:
            return

        microprice = (best_bid + best_ask) / 2.0
        if ob_imbalance is None:
            # Approximate from bid/ask
            ob_imbalance = (best_bid - best_ask) / (best_bid + best_ask + 1e-9)

        self.micro_history.append(microprice)
        self.obi_history.append(ob_imbalance)
        self.bid_history.append(best_bid)
        self.ask_history.append(best_ask)
        if sum_bid_size is not None:
            self.depth_up.append(sum_bid_size)
        if sum_ask_size is not None:
            self.depth_down.append(sum_ask_size)
        self._tick += 1

    def to_feature_dict(self, snapshots: list[int]) -> dict:
        micro = np.array(self.micro_history)
        obi   = np.array(self.obi_history)
        bids  = np.array(self.bid_history)
        asks  = np.array(self.ask_history)

        feat: dict = {}

        # Snapshot features
        for t in snapshots:
            p = f"t{t}"
            idx = min(t, len(micro) - 1)
            feat[f"{p}_up_micro"]    = micro[idx]
            feat[f"{p}_up_obi"]      = obi[idx]
            feat[f"{p}_up_bid"]      = bids[idx]
            feat[f"{p}_up_ask"]      = asks[idx]
            if self.depth_up and self.depth_down and t < len(self.depth_up):
                up_d   = self.depth_up[min(t, len(self.depth_up) - 1)]
                down_d = self.depth_down[min(t, len(self.depth_down) - 1)]
                feat[f"{p}_depth_ratio"] = up_d / (down_d + 1e-9)
            else:
                feat[f"{p}_depth_ratio"] = 1.0

        # Momentum
        n = len(micro)
        if n >= 60:
            feat["mom_60"]  = float(micro[min(59, n-1)]  - micro[0])
            feat["obi_60"]  = float(obi[:60].mean())
        if n >= 120:
            feat["mom_120"] = float(micro[min(119, n-1)] - micro[0])
            feat["obi_120"] = float(obi[:120].mean())
        if n >= 300:
            feat["mom_300"] = float(micro[min(299, n-1)] - micro[0])
            feat["obi_300"] = float(obi[:300].mean())

        # Rolling stats
        feat["micro_mean"]   = float(micro.mean())
        feat["micro_std"]    = float(micro.std())
        feat["micro_max"]    = float(micro.max())
        feat["micro_min"]    = float(micro.min())
        feat["micro_range"]  = float(micro.max() - micro.min())
        feat["obi_mean"]     = float(obi.mean())
        feat["obi_std"]      = float(obi.std())
        feat["obi_pos_frac"] = float((obi > 0).mean())

        # Depth ratio
        if self.depth_up and self.depth_down:
            up_arr   = np.array(self.depth_up)
            down_arr = np.array(self.depth_down)
            feat["depth_ratio_mean"] = float((up_arr / (down_arr + 1e-9)).mean())
        else:
            feat["depth_ratio_mean"] = 1.0

        # Opening spread
        feat["opening_spread"] = float(asks[0] - bids[0]) if len(asks) > 0 else 0.02

        return feat
