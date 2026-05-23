"""
ml/predictor.py — Load trained model and produce p_model from live tick state.

Drop-in replacement for the hand-crafted _compute_p_model() in signal_engine_ev.py.
Supports both prices-only and prices+orderbook models transparently.
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

        predictor = MLPredictor()          # loads ml/btc_15m_model.pkl
        p = predictor.predict(state)       # returns float P(Up) in [0.01, 0.99]
    """

    def __init__(self, model_path: str | Path = _DEFAULT_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run: python ml/train.py --prices <zip> --orderbook <zip>"
            )
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        self._model        = artifact["model"]
        self._feature_cols = artifact["feature_cols"]
        self._snapshots    = artifact.get("snapshots", [0, 30, 60, 120, 180, 300])
        self._has_ob       = artifact.get("has_orderbook_features", False)
        logger.info(
            "MLPredictor loaded: %d features  orderbook=%s",
            len(self._feature_cols), self._has_ob,
        )

    def predict(self, state: "MLState") -> float:
        """Return P(Up) given accumulated tick history. Returns 0.5 if insufficient data."""
        if len(state.micro_history) < 5:
            return 0.5
        feat_dict = state.to_feature_dict(self._snapshots)
        row = np.array(
            [float(feat_dict.get(col, 0.0) or 0.0) for col in self._feature_cols],
            dtype=np.float32,
        ).reshape(1, -1)
        try:
            return float(np.clip(self._model.predict_proba(row)[0, 1], 0.01, 0.99))
        except Exception as exc:
            logger.warning("MLPredictor.predict failed: %s — returning 0.5", exc)
            return 0.5


class MLState:
    """
    Accumulates per-tick data for one market.

    Call update() on every tick. Feed into MLPredictor.predict().

    Orderbook fields (top_bid_size etc.) are optional — pass None if unavailable.
    The model was trained with them; passing 0 when missing introduces a small
    bias but the prices-based features dominate and the model still outperforms
    the hand-crafted formula.
    """

    def __init__(self) -> None:
        # Prices-derived (always available)
        self.micro_history: list[float] = []
        self.obi_history:   list[float] = []
        self.bid_history:   list[float] = []
        self.ask_history:   list[float] = []
        self.depth_up:      list[float] = []    # sum_bid_size
        self.depth_down:    list[float] = []    # sum_ask_size

        # Orderbook-derived (optional — from Kalshi REST order book snapshot)
        self.top_bid_size:  list[float] = []
        self.top_ask_size:  list[float] = []
        self.n_bids:        list[float] = []
        self.n_asks:        list[float] = []
        self.spread_history: list[float] = []

    def update(
        self,
        best_bid: Optional[float],
        best_ask: Optional[float],
        ob_imbalance:  Optional[float] = None,
        sum_bid_size:  Optional[float] = None,
        sum_ask_size:  Optional[float] = None,
        top_bid_size:  Optional[float] = None,
        top_ask_size:  Optional[float] = None,
        n_bids:        Optional[int]   = None,
        n_asks:        Optional[int]   = None,
    ) -> None:
        if best_bid is None or best_ask is None:
            return

        mid = (best_bid + best_ask) / 2.0
        if ob_imbalance is None:
            # Approximate: positive = more bid pressure
            total = best_bid + best_ask
            ob_imbalance = (best_bid - best_ask) / (total + 1e-9) if total > 0 else 0.0

        self.micro_history.append(mid)
        self.obi_history.append(ob_imbalance)
        self.bid_history.append(best_bid)
        self.ask_history.append(best_ask)
        self.spread_history.append(best_ask - best_bid)

        if sum_bid_size is not None:
            self.depth_up.append(sum_bid_size)
        if sum_ask_size is not None:
            self.depth_down.append(sum_ask_size)
        if top_bid_size is not None:
            self.top_bid_size.append(top_bid_size)
        if top_ask_size is not None:
            self.top_ask_size.append(top_ask_size)
        if n_bids is not None:
            self.n_bids.append(float(n_bids))
        if n_asks is not None:
            self.n_asks.append(float(n_asks))

    def to_feature_dict(self, snapshots: list[int]) -> dict:
        micro = np.array(self.micro_history)
        obi   = np.array(self.obi_history)
        bids  = np.array(self.bid_history)
        asks  = np.array(self.ask_history)
        sprd  = np.array(self.spread_history)
        n     = len(micro)

        # Optional orderbook arrays
        top_b  = np.array(self.top_bid_size)  if self.top_bid_size  else None
        top_a  = np.array(self.top_ask_size)  if self.top_ask_size  else None
        dep_u  = np.array(self.depth_up)      if self.depth_up      else None
        dep_d  = np.array(self.depth_down)    if self.depth_down    else None
        nb     = np.array(self.n_bids)        if self.n_bids        else None
        na     = np.array(self.n_asks)        if self.n_asks        else None

        feat: dict = {}

        # ── Snapshot features ─────────────────────────────────────────────────
        for t in snapshots:
            idx = min(t, n - 1)
            p   = f"t{t}"

            feat[f"{p}_up_micro"] = float(micro[idx])
            feat[f"{p}_up_obi"]   = float(obi[idx])
            feat[f"{p}_up_bid"]   = float(bids[idx])
            feat[f"{p}_up_ask"]   = float(asks[idx])

            # Depth ratio (prices-side)
            if dep_u is not None and dep_d is not None and idx < len(dep_u):
                feat[f"{p}_depth_ratio"] = float(dep_u[idx] / (dep_d[idx] + 1e-9))
            else:
                feat[f"{p}_depth_ratio"] = 1.0

            # Orderbook features (0 = neutral default when not available)
            if top_b is not None and top_a is not None and idx < len(top_b):
                feat[f"{p}_top_ratio"] = float(top_b[idx] / (top_a[idx] + 1e-9))
                feat[f"{p}_sum_ratio"] = float(
                    dep_u[idx] / (dep_d[idx] + 1e-9) if dep_u is not None and idx < len(dep_u) else 1.0
                )
            else:
                feat[f"{p}_top_ratio"] = 1.0
                feat[f"{p}_sum_ratio"] = 1.0

            feat[f"{p}_spread"] = float(sprd[idx]) if idx < len(sprd) else 0.02
            feat[f"{p}_n_bids"] = float(nb[idx]) if nb is not None and idx < len(nb) else 0.0
            feat[f"{p}_n_asks"] = float(na[idx]) if na is not None and idx < len(na) else 0.0

        # ── Momentum ──────────────────────────────────────────────────────────
        for steps in [60, 120, 300]:
            if n >= steps:
                feat[f"mom_{steps}"] = float(micro[steps - 1] - micro[0])
                feat[f"obi_{steps}"] = float(obi[:steps].mean())

        # Orderbook momentum
        if top_b is not None and top_a is not None:
            top_ratio_arr = top_b / (top_a + 1e-9)
            for steps in [60, 120, 300]:
                if len(top_ratio_arr) >= steps:
                    feat[f"top_ratio_mom_{steps}"] = float(
                        top_ratio_arr[steps - 1] - top_ratio_arr[0]
                    )

        # ── Rolling stats ─────────────────────────────────────────────────────
        feat.update({
            "micro_mean":   float(micro.mean()),
            "micro_std":    float(micro.std()),
            "micro_max":    float(micro.max()),
            "micro_min":    float(micro.min()),
            "micro_range":  float(micro.max() - micro.min()),
            "obi_mean":     float(obi.mean()),
            "obi_std":      float(obi.std()),
            "obi_pos_frac": float((obi > 0).mean()),
        })

        # Depth ratio mean
        if dep_u is not None and dep_d is not None:
            feat["depth_ratio_mean"] = float((dep_u / (dep_d + 1e-9)).mean())
        else:
            feat["depth_ratio_mean"] = 1.0

        # Orderbook rolling stats
        if top_b is not None and top_a is not None:
            tr = top_b / (top_a + 1e-9)
            feat.update({
                "top_ratio_mean": float(tr.mean()),
                "top_ratio_std":  float(tr.std()),
                "top_ratio_max":  float(tr.max()),
            })
            if dep_u is not None and dep_d is not None:
                sr = dep_u / (dep_d + 1e-9)
                feat.update({
                    "sum_ratio_mean": float(sr.mean()),
                    "sum_ratio_std":  float(sr.std()),
                })
        else:
            feat.update({
                "top_ratio_mean": 1.0, "top_ratio_std": 0.0, "top_ratio_max": 1.0,
                "sum_ratio_mean": 1.0, "sum_ratio_std": 0.0,
            })

        feat["spread_mean"] = float(sprd.mean())
        feat["spread_std"]  = float(sprd.std())

        # Opening spread
        feat["opening_spread"] = float(sprd[0]) if len(sprd) > 0 else 0.02

        return feat
