"""
signal_engine_ev.py — EV-based grid filter signal engine.

Strategy:
  1. Compute p_model from 9 additive market microstructure + cross-asset signals
  2. Calculate EV = p_model × (1/ask - 1) - (1 - p_model) - fee
  3. Enter when EV > MIN_EV and ask is in grid range [GRID_MIN, GRID_MAX]
  4. Hold to expiry — no intra-market exit. The portfolio poller detects
     settlement (position disappears from Kalshi) and closes engine state.

p_model components (additive, each capped):
  base_p            — market mid price as baseline probability
  delta_weight      — N-tick price momentum (direction × magnitude)
  delta_atr         — 1-tick move / ATR: NEGATED (mean-reverting on 1m Kalshi ticks)
  ob_imbalance      — bid/ask drift asymmetry (proxy for book pressure)
  cross_asset_boost — BTC spot direction (Coinbase)
  tf_confirm_boost  — 5-tick vs 20-tick momentum agreement
  volume_boost      — volume surge: NEGATED (high vol = more noise, fade trend)
  candle_boost      — BTC 15m candle direction (Coinbase)
  price_spike_boost — spike >3¢ in 5 ticks: NEGATED (fade mean-reverting spikes)
  cvd_boost         — Coinbase spot CVD (cumulative volume delta)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional, TYPE_CHECKING

from core.models import Signal, SignalType
from core.config import Config

if TYPE_CHECKING:
    from core.btc_feed import BtcFeed
    from core.coinbase_spot_feed import CoinbaseSpotFeed

logger = logging.getLogger(__name__)

FIXED_RISK          = 0.02   # stop = entry_price - FIXED_RISK (stored, not enforced)
POST_CLOSE_COOLDOWN = 60.0   # seconds to block re-entry after a position closes
MARKET_OPEN_LOCKOUT = 90.0   # seconds to block entry after a new market is first seen
ATR_WINDOW          = 14     # ticks for rolling ATR
MOMENTUM_WINDOW_SHORT = 5      # ticks for short-term tf_confirm
MOMENTUM_WINDOW_LONG  = 20     # ticks for long-term tf_confirm
VOL_WINDOW            = 20     # ticks for volume rolling average
PRICE_SPIKE_WINDOW    = 5      # ticks for spike detection lookback
OB_IMBALANCE_WINDOW   = 10     # ticks for bid/ask drift rolling window
MIN_HISTORY_TICKS   = MOMENTUM_WINDOW_SHORT  # must have this many ticks before entering


class EVMarketState:
    """Per-ticker state for the EV engine."""

    def __init__(self, ticker: str, market_id: int) -> None:
        self.ticker    = ticker
        self.market_id = market_id

        # Position
        self.has_position    = False
        self.position_ticker: Optional[str]   = None
        self.position_side:   Optional[str]   = None
        self.entry_price:     Optional[float] = None
        self.stop_price:      Optional[float] = None
        self.position_id:     Optional[int]   = None
        self.pending_entry    = False

        # Price / bid / ask / volume histories
        self.price_history:  deque = deque(maxlen=MOMENTUM_WINDOW_LONG + 2)
        self.bid_history:    deque = deque(maxlen=OB_IMBALANCE_WINDOW)
        self.ask_history:    deque = deque(maxlen=OB_IMBALANCE_WINDOW)
        self.volume_history: deque = deque(maxlen=VOL_WINDOW)
        self.atr_history:    deque = deque(maxlen=ATR_WINDOW)

        # BTC 15m market context — updated each REST snapshot
        self.btc_target:  Optional[float] = None
        self.close_ts:    float = 0.0    # unix timestamp of 15m candle close

        # Cooldown / lockouts
        self.cooldown_until:    float = 0.0
        self.market_open_until: float = time.time() + MARKET_OPEN_LOCKOUT

        # Simulation time override (backtest only — replaces time.time() in time-sensitive features)
        self.sim_time: Optional[float] = None

        # Feature snapshot at last entry signal — used for ML training data logging
        self.last_entry_features: Optional[dict] = None


class EVSignalEngine:
    """
    EV-based grid filter with confidence boosts.

    Instantiated once via SignalEngineRouter. All methods are thread-safe
    through a single per-engine lock (coarse-grained, consistent with the
    existing engine pattern).
    """

    def __init__(
        self,
        config: Config,
        btc_feed: "BtcFeed",
        coinbase_spot_feed: "CoinbaseSpotFeed",
    ) -> None:
        self._config  = config
        self._btc     = btc_feed
        self._bfeed   = coinbase_spot_feed
        self._lock    = threading.Lock()
        self._states: dict[str, EVMarketState] = {}
        self._prev_coinbase_mid: Optional[float] = None

    # ── State management ──────────────────────────────────────────────

    def get_or_create_state(self, ticker: str, market_id: int) -> EVMarketState:
        if ticker not in self._states:
            with self._lock:
                if ticker not in self._states:
                    self._states[ticker] = EVMarketState(ticker, market_id)
        return self._states[ticker]

    def mark_position_open(
        self,
        ticker: str,
        position_id: int,
        entry_price: float,
        side: Optional[str] = None,
    ) -> None:
        st = self._states.get(ticker)
        if st is None:
            return
        with self._lock:
            st.has_position    = True
            st.pending_entry   = False
            st.position_ticker = ticker
            st.entry_price     = entry_price
            st.stop_price      = round(entry_price - FIXED_RISK, 6)
            st.position_id     = position_id
            st.position_side   = side or "YES"
        logger.info(
            "[EV] IN: %s @ %.4f  stop=%.4f  side=%s  id=%d",
            ticker, entry_price, st.stop_price, st.position_side, position_id,
        )

    def mark_position_closed(self, ticker: str) -> None:
        st = self._states.get(ticker)
        if st is None:
            return
        with self._lock:
            st.has_position    = False
            st.pending_entry   = False
            st.position_ticker = None
            st.entry_price     = None
            st.stop_price      = None
            st.position_id     = None
            st.position_side   = None
            st.cooldown_until  = time.time() + POST_CLOSE_COOLDOWN
        logger.info("[EV] CLOSED: %s — cooling down %.0fs", ticker, POST_CLOSE_COOLDOWN)

    def update_market_context(
        self, ticker: str, btc_target: Optional[float], close_ts: Optional[float]
    ) -> None:
        """Set BTC strike price and close timestamp for this market. Called each REST snapshot."""
        if ticker not in self._states:
            return
        st = self._states[ticker]
        with self._lock:
            if btc_target:
                st.btc_target = btc_target
            if close_ts:
                st.close_ts = float(close_ts)

    def mark_cooldown(self, ticker: str, duration: float = 30.0) -> None:
        st = self._states.get(ticker)
        if st:
            with self._lock:
                st.cooldown_until = time.time() + duration
            logger.info("[EV] Cooldown: %s blocked for %.0fs", ticker, duration)

    def get_stop_price(self, ticker: Optional[str] = None) -> Optional[float]:
        with self._lock:
            if ticker:
                st = self._states.get(ticker)
                return st.stop_price if st and st.has_position else None
            for st in self._states.values():
                if st.has_position and st.stop_price is not None:
                    return st.stop_price
        return None

    def get_last_features(self, ticker: str) -> Optional[dict]:
        """Return the feature snapshot from the last entry signal, or None."""
        st = self._states.get(ticker)
        if st is None:
            return None
        with self._lock:
            return st.last_entry_features

    def get_position_snapshot(self, ticker: str) -> Optional[dict]:
        st = self._states.get(ticker)
        if not st or not st.has_position:
            return None
        with self._lock:
            return {
                "ticker":      ticker,
                "side":        st.position_side,
                "entry_price": st.entry_price,
                "position_id": st.position_id,
            }

    # ── Main tick processing ──────────────────────────────────────────

    def process_tick(
        self,
        ticker: str,
        market_id: int,
        price: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
        volume: Optional[float] = None,
        sim_time: Optional[float] = None,
    ) -> Optional[Signal]:
        st = self.get_or_create_state(ticker, market_id)

        with self._lock:
            if sim_time is not None:
                st.sim_time = sim_time
            self._update_history(st, price, best_bid, best_ask, volume)

            if st.has_position:
                return None   # hold to expiry — poller handles settlement close

            now = time.time()
            if st.pending_entry or now < st.cooldown_until:
                return None
            if now < st.market_open_until:
                logger.debug(
                    "[EV] %s open-lockout: %.0fs remaining",
                    ticker, st.market_open_until - now,
                )
                return None

            return self._check_entry(st, ticker, market_id, price, best_bid, best_ask)

    # ── History updates ───────────────────────────────────────────────

    def _update_history(
        self,
        st: EVMarketState,
        price: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
        volume: Optional[float],
    ) -> None:
        if st.price_history:
            st.atr_history.append(abs(price - st.price_history[-1]))
        st.price_history.append(price)

        if best_bid is not None:
            st.bid_history.append(best_bid)
        if best_ask is not None:
            st.ask_history.append(best_ask)
        if volume is not None and volume > 0:
            st.volume_history.append(volume)

    # ── EV computation ────────────────────────────────────────────────

    def _compute_p_model(
        self,
        st: EVMarketState,
        price: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
        volume: Optional[float],
    ) -> tuple[float, dict]:
        """Return (p_model, features_dict). features_dict captured for ML training data."""
        mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else price

        base_p            = mid
        delta_weight      = self._feat_delta_weight(st, cap=0.03)
        ob_imbalance      = self._feat_ob_imbalance(st, cap=0.02)
        cross_asset_boost = self._feat_cross_asset(cap=0.02)
        tf_confirm_boost  = self._feat_tf_confirm(st, cap=0.015)
        candle_boost      = self._feat_candle(cap=0.01)
        cvd_boost         = self._feat_cvd(cap=0.02)
        btc_distance      = self._feat_btc_distance(st, cap=0.06)
        time_pressure     = self._feat_time_pressure(st, btc_distance, cap=0.06)
        # Zeroed out — backtest data showed consistent negative correlation
        # regardless of sign; pure noise on 1m KXBTC15M ticks.
        delta_atr         = 0.0
        volume_boost      = 0.0
        price_spike_boost = 0.0

        p = (base_p + delta_weight + ob_imbalance +
             cross_asset_boost + tf_confirm_boost +
             candle_boost + cvd_boost +
             btc_distance + time_pressure)
        p = max(0.01, min(0.99, p))

        features = {
            "base_p":            base_p,
            "delta_weight":      delta_weight,
            "delta_atr":         delta_atr,
            "ob_imbalance":      ob_imbalance,
            "cross_asset_boost": cross_asset_boost,
            "tf_confirm_boost":  tf_confirm_boost,
            "volume_boost":      volume_boost,
            "candle_boost":      candle_boost,
            "price_spike_boost": price_spike_boost,
            "cvd_boost":         cvd_boost,
            "btc_distance":      btc_distance,
            "time_pressure":     time_pressure,
            "p_model":           p,
        }

        logger.debug(
            "[EV] %s p=%.4f base=%.3f Δw=%.4f ob=%.4f "
            "cross=%.4f tf=%.4f cndl=%.4f cvd=%.4f "
            "btc_dist=%.4f t_press=%.4f",
            st.ticker, p, base_p, delta_weight, ob_imbalance,
            cross_asset_boost, tf_confirm_boost,
            candle_boost, cvd_boost,
            btc_distance, time_pressure,
        )
        return p, features

    @staticmethod
    def _ev_yes(p_model: float, ask: float, fee_rate: float) -> float:
        """EV = p × (1/ask - 1) - (1 - p) - fee"""
        if ask <= 0 or ask >= 1:
            return -999.0
        fee = fee_rate * ask * (1.0 - ask)
        return p_model * (1.0 / ask - 1.0) - (1.0 - p_model) - fee

    @staticmethod
    def _ev_no(p_model: float, no_ask: float, fee_rate: float) -> float:
        """Same formula for the NO side (p_no = 1 - p_model)."""
        p_no = 1.0 - p_model
        if no_ask <= 0 or no_ask >= 1:
            return -999.0
        fee = fee_rate * no_ask * (1.0 - no_ask)
        return p_no * (1.0 / no_ask - 1.0) - (1.0 - p_no) - fee

    # ── Feature functions ─────────────────────────────────────────────

    def _feat_delta_weight(self, st: EVMarketState, cap: float) -> float:
        """N-tick price momentum: (now - N ticks ago) × weight."""
        hist = list(st.price_history)
        n = 10
        if len(hist) < n + 1:
            return 0.0
        delta = hist[-1] - hist[-(n + 1)]
        return max(-cap, min(cap, delta / n))

    def _feat_delta_atr(self, st: EVMarketState, cap: float) -> float:
        """1-tick delta normalized by rolling ATR — NEGATED: short-term overshoot mean-reverts."""
        hist = list(st.price_history)
        atrs = list(st.atr_history)
        if len(hist) < 2 or len(atrs) < 3:
            return 0.0
        delta = hist[-1] - hist[-2]
        atr   = sum(atrs) / len(atrs)
        if atr < 1e-6:
            return 0.0
        return max(-cap, min(cap, -delta / atr * 0.5))

    def _feat_ob_imbalance(self, st: EVMarketState, cap: float) -> float:
        """
        Proxy for order book pressure using bid/ask price drift.
        Rising bids faster than asks = buying pressure = positive.
        """
        bids = list(st.bid_history)
        asks = list(st.ask_history)
        if len(bids) < 4 or len(asks) < 4:
            return 0.0
        half = len(bids) // 2
        bid_recent = sum(bids[half:]) / len(bids[half:])
        bid_older  = sum(bids[:half])  / len(bids[:half])
        ask_recent = sum(asks[half:]) / len(asks[half:])
        ask_older  = sum(asks[:half])  / len(asks[:half])
        bid_drift  = bid_recent - bid_older
        ask_drift  = ask_recent - ask_older
        raw = (bid_drift - ask_drift) * 5.0
        return max(-cap, min(cap, raw))

    def _feat_cross_asset(self, cap: float) -> float:
        """BTC spot direction: current mid vs previous mid."""
        mid = self._bfeed.mid_price
        if mid is None:
            return 0.0
        if self._prev_coinbase_mid is None:
            self._prev_coinbase_mid = mid
            return 0.0
        delta = mid - self._prev_coinbase_mid
        self._prev_coinbase_mid = mid
        raw = delta / 500.0   # $500 move → full cap
        return max(-cap, min(cap, raw))

    def _feat_tf_confirm(self, st: EVMarketState, cap: float) -> float:
        """Multi-timeframe agreement: 5-tick and 20-tick momentum same direction."""
        hist = list(st.price_history)
        if len(hist) < MOMENTUM_WINDOW_LONG + 1:
            return 0.0
        short_mom = hist[-1] - hist[-(MOMENTUM_WINDOW_SHORT + 1)]
        long_mom  = hist[-1] - hist[-(MOMENTUM_WINDOW_LONG + 1)]
        if short_mom > 0 and long_mom > 0:
            return cap * 0.5 * min(1.0, abs(short_mom) / 0.05)
        if short_mom < 0 and long_mom < 0:
            return -cap * 0.5 * min(1.0, abs(short_mom) / 0.05)
        return 0.0

    def _feat_volume(
        self, st: EVMarketState, volume: Optional[float], cap: float
    ) -> float:
        """Volume surge vs rolling average — NEGATED: high volume signals noise, fade direction."""
        if volume is None or len(st.volume_history) < 5:
            return 0.0
        avg = sum(st.volume_history) / len(st.volume_history)
        if avg < 1e-6:
            return 0.0
        ratio = volume / avg
        if ratio < 1.1:
            return 0.0
        hist = list(st.price_history)
        if len(hist) < 2:
            return 0.0
        direction = 1.0 if hist[-1] >= hist[-2] else -1.0
        return max(-cap, min(cap, -direction * min(1.0, ratio - 1.0) * cap))

    def _feat_candle(self, cap: float) -> float:
        """BTC 15m Coinbase candle: bullish body → positive, bearish → negative."""
        candles = self._btc.latest_candles
        if not candles:
            return 0.0
        c = candles[0]
        rng = c.range_size
        if rng < 1e-4:
            return 0.0
        body_ratio = c.body_size / rng
        direction  = 1.0 if c.is_bullish else -1.0
        return max(-cap, min(cap, direction * body_ratio * cap))

    def _feat_spike(self, st: EVMarketState, cap: float) -> float:
        """Price spike >3¢ in last 5 ticks — NEGATED: fade spike direction (mean reversion)."""
        hist = list(st.price_history)
        if len(hist) < PRICE_SPIKE_WINDOW + 1:
            return 0.0
        delta = hist[-1] - hist[-(PRICE_SPIKE_WINDOW + 1)]
        if abs(delta) < 0.03:
            return 0.0
        direction = 1.0 if delta > 0 else -1.0
        return max(-cap, min(cap, -direction * min(1.0, abs(delta) / 0.05) * cap))

    def _feat_cvd(self, cap: float) -> float:
        """Binance spot CVD normalized to [-1, 1], scaled by cap."""
        cvd = self._bfeed.cvd
        return max(-cap, min(cap, cvd * cap))

    def _feat_btc_distance(self, st: EVMarketState, cap: float) -> float:
        """
        Primary BTC 15m signal: how far is the current BTC price from the market's
        strike (target) price?

        Positive = BTC is ABOVE the target → YES outcome is more likely.
        Negative = BTC is BELOW the target → YES is less likely.

        A 1% distance (e.g. $650 when BTC=$65,000) maps to ±0.05 boost.
        Cap is intentionally larger than other features — this is the core signal
        for a binary BTC price prediction market.
        """
        btc_price = self._bfeed.mid_price
        if btc_price is None or not st.btc_target or st.btc_target <= 0:
            return 0.0
        distance = (btc_price - st.btc_target) / st.btc_target
        return max(-cap, min(cap, distance * 5.0))

    def _feat_time_pressure(self, st: EVMarketState, btc_distance: float, cap: float) -> float:
        """
        Amplify the btc_distance signal as close time approaches.

        Logic: if BTC is $500 above target with 14 minutes left, there's still
        time for a reversal — moderate confidence. If there's 2 minutes left and
        BTC is the same $500 above, the outcome is much more certain.

        time_fraction: 0 = 15 min left (market just opened), 1 = at close
        boost = btc_distance * time_fraction * scale
        """
        if not st.close_ts:
            return 0.0
        now = st.sim_time if st.sim_time is not None else time.time()
        remaining = st.close_ts - now
        if remaining <= 0:
            return 0.0
        market_duration = 900.0   # 15 minutes in seconds
        time_fraction = max(0.0, 1.0 - (remaining / market_duration))
        # Only add pressure signal in last 50% of the window (after 7.5 min)
        if time_fraction < 0.5:
            return 0.0
        raw = btc_distance * (time_fraction - 0.5) * 2.0 * cap
        return max(-cap, min(cap, raw))

    # ── Entry / exit logic ────────────────────────────────────────────

    def _check_entry(
        self,
        st: EVMarketState,
        ticker: str,
        market_id: int,
        price: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
    ) -> Optional[Signal]:
        cfg = self._config
        if best_ask is None or best_bid is None:
            return None

        # Need minimum history so momentum features have data
        if len(st.price_history) < MIN_HISTORY_TICKS:
            return None

        yes_in_grid = cfg.ev_grid_min <= best_ask <= cfg.ev_grid_max
        no_ask = round(1.0 - best_bid, 6) if best_bid else None
        no_in_grid  = no_ask is not None and cfg.ev_grid_min <= no_ask <= cfg.ev_grid_max

        if not yes_in_grid and not no_in_grid:
            return None

        volume          = st.volume_history[-1] if st.volume_history else None
        p_model, feats  = self._compute_p_model(st, price, best_bid, best_ask, volume)
        fee             = cfg.ev_fee_rate

        ev_yes = self._ev_yes(p_model, best_ask, fee) if yes_in_grid else -999.0
        ev_no  = self._ev_no(p_model, no_ask, fee)    if no_in_grid  else -999.0

        min_ev = cfg.ev_min_entry

        if ev_yes >= min_ev and ev_yes >= ev_no:
            side, entry_px, ev = "YES", best_ask, ev_yes
        elif ev_no >= min_ev:
            side, entry_px, ev = "NO", no_ask, ev_no
        else:
            return None

        st.pending_entry      = True
        st.position_side      = side
        # Snapshot features for ML training log — read by risk_manager after fill
        now_ts = st.sim_time if st.sim_time is not None else time.time()
        seconds_remaining = max(0.0, st.close_ts - now_ts) if st.close_ts else None
        seconds_elapsed   = max(0.0, now_ts - (st.close_ts - 900.0)) if st.close_ts else None

        st.last_entry_features = {
            **feats,
            "ev": ev,
            "side": side,
            "entry_price": entry_px,
            # New context fields for DB logging
            "spread": round(best_ask - best_bid, 6),
            "btc_target": st.btc_target,
            "seconds_elapsed": seconds_elapsed,
            "seconds_remaining": seconds_remaining,
            "tick_count": len(st.price_history),
        }

        logger.info(
            "[EV] ENTRY: %s  side=%s  ask=%.4f  p_model=%.4f  ev=%.5f",
            ticker, side, entry_px, p_model, ev,
        )

        return Signal(
            ticker=ticker,
            market_id=market_id,
            signal_type=SignalType.ENTRY,
            price=entry_px,
            metadata={
                "engine":   "ev_grid",
                "side":     side,
                "p_model":  p_model,
                "ev":       ev,
                "best_ask": best_ask,
                "best_bid": best_bid,
            },
        )

    def _check_exit(
        self,
        st: EVMarketState,
        ticker: str,
        market_id: int,
        price: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
    ) -> Optional[Signal]:
        if best_bid is None or best_ask is None:
            return None

        # Don't exit within the first MIN_HOLD_TICKS ticks — normal bid/ask spread
        # movement can drop the buy-EV below the exit threshold on the very next tick.
        if st.ticks_in_position < MIN_HOLD_TICKS:
            return None

        cfg  = self._config
        side = st.position_side

        # ── Auto take-profit: exit when EV flips ─────────────────────
        volume         = st.volume_history[-1] if st.volume_history else None
        p_model, _     = self._compute_p_model(st, price, best_bid, best_ask, volume)
        fee            = cfg.ev_fee_rate

        if side == "YES":
            current_ev = self._ev_yes(p_model, best_ask, fee)
            exit_price = best_bid
        else:
            no_ask     = round(1.0 - best_bid, 6)
            current_ev = self._ev_no(p_model, no_ask, fee)
            exit_price = round(1.0 - best_ask, 6)

        if current_ev < cfg.ev_min_exit:
            logger.info(
                "[EV] AUTO-EXIT (EV flip): %s  ev=%.5f < min_exit=%.5f",
                ticker, current_ev, cfg.ev_min_exit,
            )
            return Signal(
                ticker=ticker,
                market_id=market_id,
                signal_type=SignalType.EXIT,
                price=exit_price,
                metadata={
                    "engine":  "ev_grid",
                    "side":    side,
                    "reason":  "ev_flip",
                    "ev":      current_ev,
                    "p_model": p_model,
                },
            )

        return None
