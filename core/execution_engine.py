"""
execution_engine.py — Places and manages orders on Kalshi.

Features:
  - Limit orders at best ask (entry) or best bid (exit)
  - Exponential backoff retry
  - Paper trade mode (simulates fills without API calls)
  - Full logging of every attempt
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from pykalshi import KalshiClient
from pykalshi._sync.portfolio import Action, Side as KalshiSide

from core.config import Config
from core.models import OrderResult, Side

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


class ExecutionEngine:
    """
    Handles order placement for entry and exit signals.
    In paper trade mode, fills are simulated at the provided price.
    """

    def __init__(self, client: KalshiClient, config: Config) -> None:
        self._client = client
        self._config = config

    def buy(
            self,
            ticker: str,
            price: float,
            best_ask: Optional[float],
            qty: Optional[int] = None,
            kalshi_side: str = "YES",   # "YES" or "NO"
    ) -> OrderResult:
        """
        Place a limit BUY order.
        kalshi_side="NO" buys NO contracts (bets price goes down).
        """
        limit_price = best_ask if best_ask is not None else round(price + 0.01, 2)
        limit_price = max(0.01, min(0.99, round(limit_price, 2)))
        contracts   = qty if qty is not None else self._config.position_size

        if kalshi_side == "NO":
            # Buying NO at 96-98¢. YES-equivalent = 1 - NO_price ≈ 2-4¢.
            # Kalshi receives yes_price = 1 - limit_price.
            logger.info(
                "[%s] BUY NO: qty=%d  NO_limit=%.4f  YES_equiv=%.4f  (paper=%s)",
                ticker, contracts, limit_price, round(1.0 - limit_price, 4),
                self._config.paper_trade,
            )
        else:
            logger.info(
                "[%s] BUY YES: qty=%d limit=%.4f (paper=%s)",
                ticker, contracts, limit_price, self._config.paper_trade,
            )

        if self._config.paper_trade:
            return OrderResult(
                success=True,
                order_id=f"PAPER-{ticker}-{int(__import__('time').time())}",
                filled_price=limit_price,
                filled_qty=contracts,
            )

        return self._place_order(ticker, Side.BUY, limit_price, contracts, kalshi_side)

    def sell(
            self,
            ticker: str,
            price: float,
            best_bid: Optional[float],
            quantity: int,
            kalshi_side: str = "YES",
    ) -> OrderResult:
        """
        Place a limit SELL to exit a position.
        `price` is the caller-computed limit for the correct side:
          YES position: price = YES bid (or discounted YES bid)
          NO  position: price = NO bid = 1 - YES ask (computed by risk_manager)
        """
        # Use the caller-provided side-aware price as-is.
        # risk_manager already computed the right price for YES vs NO.
        # Round to 2 decimal places — Kalshi only accepts whole-cent prices.
        limit_price = max(0.01, min(0.99, round(price, 2)))

        if kalshi_side == "NO":
            logger.info(
                "[%s] SELL NO: qty=%d  NO_limit=%.4f  YES_equiv=%.4f  (paper=%s)",
                ticker, quantity, limit_price, round(1.0 - limit_price, 4),
                self._config.paper_trade,
            )
        else:
            logger.info(
                "[%s] SELL YES: qty=%d limit=%.4f (paper=%s)",
                ticker, quantity, limit_price, self._config.paper_trade,
            )

        if self._config.paper_trade:
            return OrderResult(
                success=True,
                order_id=f"PAPER-SELL-{ticker}-{int(time.time())}",
                filled_price=limit_price,
                filled_qty=quantity,
            )

        return self._place_order(ticker, Side.SELL, limit_price, quantity, kalshi_side)

    def _place_order(
            self,
            ticker: str,
            side: Side,
            limit_price: float,
            qty: int,
            kalshi_side: str = "YES",
    ) -> OrderResult:
        # Kalshi API ALWAYS takes yes_price_dollars, regardless of side.
        # limit_price is the price of the side we're trading:
        #   YES side: limit_price IS the YES price → pass directly
        #   NO  side: limit_price is the NO price → convert to YES price = 1 - no_price
        if kalshi_side == "NO":
            yes_price = round(1.0 - limit_price, 4)
        else:
            yes_price = limit_price

        # ── Sanity guard — catch reversed-side bugs before hitting the API ──
        # BUY YES:  yes_price should be HIGH (we're paying ~96-98¢ for YES)
        # BUY NO:   yes_price should be LOW  (we're paying ~2-4¢ YES-equivalent for NO at 96-98¢)
        # SELL YES: yes_price should be HIGH (we're receiving ~96-98¢)
        # SELL NO:  yes_price should be LOW  (we're receiving ~96¢+ for our NO contracts)
        if side == Side.BUY:
            if kalshi_side == "YES" and yes_price < 0.50:
                logger.error(
                    "[%s] ORDER REJECTED: BUY YES but yes_price=%.4f is suspiciously LOW "
                    "(expected 0.96-0.98). Possible side confusion. Aborting.",
                    ticker, yes_price,
                )
                return OrderResult(success=False, order_id=None,
                                   filled_price=None, filled_qty=0,
                                   error="sanity_check: BUY YES price too low")
            if kalshi_side == "NO" and yes_price > 0.50:
                logger.error(
                    "[%s] ORDER REJECTED: BUY NO but yes_price=%.4f is suspiciously HIGH "
                    "(expected 0.02-0.04 for a 96-98¢ NO contract). Possible side confusion. Aborting.",
                    ticker, yes_price,
                )
                return OrderResult(success=False, order_id=None,
                                   filled_price=None, filled_qty=0,
                                   error="sanity_check: BUY NO yes_price too high")

        yes_price  = round(yes_price, 2)   # Kalshi only accepts whole-cent prices
        price_str  = f"{yes_price:.2f}"
        count_str  = str(qty)
        action     = Action.BUY if side == Side.BUY else Action.SELL
        ks         = KalshiSide.YES if kalshi_side == "YES" else KalshiSide.NO
        delay      = BASE_DELAY

        for attempt in range(MAX_RETRIES):
            try:
                resp = self._client.portfolio.place_order(
                    ticker=ticker,
                    action=action,
                    side=ks,
                    count_fp=count_str,
                    yes_price_dollars=price_str,
                )

                order_id   = getattr(resp, "order_id", None) or getattr(resp, "id", None)
                filled_qty = getattr(resp, "count_filled", None)
                filled_p   = getattr(resp, "yes_price_dollars", None)

                filled_qty   = int(filled_qty)   if filled_qty   is not None else qty
                yes_filled   = float(filled_p)   if filled_p     is not None else yes_price
                # Convert back: for NO orders filled_price should be the NO price
                filled_price = round(1.0 - yes_filled, 4) if kalshi_side == "NO" else yes_filled

                logger.info(
                    "[%s] Order placed: id=%s filled_qty=%d filled_price=%.4f",
                    ticker, order_id, filled_qty, filled_price,
                )

                return OrderResult(
                    success=True,
                    order_id=str(order_id),
                    filled_price=filled_price,
                    filled_qty=filled_qty,
                )

            except Exception as exc:
                exc_str = str(exc).lower()

                # Don't retry on errors that won't resolve in seconds
                non_retryable = (
                        "insufficient_balance" in exc_str
                        or "insufficient balance" in exc_str
                        or "market_closed"       in exc_str
                        or "market closed"       in exc_str
                )

                if non_retryable:
                    logger.error(
                        "[%s] Order failed (non-retryable): %s",
                        ticker, exc,
                    )
                    return OrderResult(
                        success=False,
                        order_id=None,
                        filled_price=None,
                        filled_qty=0,
                        error=str(exc),
                    )

                logger.warning(
                    "[%s] Order attempt %d/%d failed: %s — retrying in %.1fs",
                    ticker, attempt + 1, MAX_RETRIES, exc, delay,
                            )
                if attempt == MAX_RETRIES - 1:
                    return OrderResult(
                        success=False,
                        order_id=None,
                        filled_price=None,
                        filled_qty=0,
                        error=str(exc),
                    )
                time.sleep(delay)
                delay *= 2

        return OrderResult(success=False, order_id=None, filled_price=None, filled_qty=0)