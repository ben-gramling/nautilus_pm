"""
Kalshi spread-capture (market-making) strategy.

Places a limit buy at (fair_value - spread_cents) and a limit sell at
(fair_value + spread_cents) simultaneously.  Fair value is estimated as
an EMA of recent trade prices.  Orders are re-quoted only when fair value
shifts by more than `requote_threshold` cents, keeping churn low.

Inventory is capped at +/- max_position contracts so the strategy does
not accumulate a runaway directional bet.
"""

from __future__ import annotations

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import LimitOrder
from nautilus_trader.trading.strategy import Strategy


class KalshiSpreadCaptureConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``KalshiSpreadCapture``.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument to trade.
    spread_cents : float, default 2.0
        Half-spread in price units (same units as the instrument price).
        Bid placed at fair_value - spread_cents, ask at fair_value + spread_cents.
    max_position : int, default 5
        Maximum net long or short inventory in contracts.
    trade_size : float, default 1.0
        Contracts per limit order.
    ema_alpha : float, default 0.1
        Smoothing factor for the fair-value EMA (0 < alpha <= 1).
    min_ticks : int, default 5
        Number of ticks to consume before starting to quote.
    requote_threshold : float, default 0.5
        Minimum fair-value shift (in price units) required to cancel and
        re-place orders.  Keeps order churn low.
    """

    instrument_id: InstrumentId
    spread_cents: float = 2.0
    max_position: int = 5
    trade_size: float = 1.0
    ema_alpha: float = 0.1
    min_ticks: int = 5
    requote_threshold: float = 0.5


class KalshiSpreadCapture(Strategy):
    """
    A simple spread-capture / market-making strategy for Kalshi binary markets.

    Uses only trade tick data — no order book required.  Fair value is
    estimated from a running EMA of trade prices.
    """

    def __init__(self, config: KalshiSpreadCaptureConfig) -> None:
        super().__init__(config)
        self._instrument: Instrument | None = None
        self._fair_value: float | None = None
        self._last_quoted_fv: float | None = None
        self._tick_count: int = 0
        self._buy_order: LimitOrder | None = None
        self._sell_order: LimitOrder | None = None

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument not found: {self.config.instrument_id}")
            self.stop()
            return
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)

        # --- Update fair-value EMA ---
        if self._fair_value is None:
            self._fair_value = price
        else:
            self._fair_value = (
                self.config.ema_alpha * price
                + (1.0 - self.config.ema_alpha) * self._fair_value
            )
        self._tick_count += 1

        # Wait for warm-up
        if self._tick_count < self.config.min_ticks:
            return

        fv = self._fair_value

        # Only requote if fair value has moved enough
        if (
            self._last_quoted_fv is not None
            and abs(fv - self._last_quoted_fv) < self.config.requote_threshold
        ):
            return

        self._requote(fv)

    def _requote(self, fv: float) -> None:
        """Cancel existing orders and place fresh quotes around fair value."""
        # Cancel any live orders
        for order in (self._buy_order, self._sell_order):
            if order is not None and (order.is_open or order.is_emulated):
                self.cancel_order(order)
        self._buy_order = None
        self._sell_order = None

        inst = self._instrument
        spread = self.config.spread_cents
        size = inst.make_qty(self.config.trade_size)

        # --- Net position (positive = long) ---
        net = self._net_position()

        # Place buy if not at max long
        if net < self.config.max_position:
            bid_price = inst.make_price(fv - spread)
            self._buy_order = self.order_factory.limit(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=size,
                price=bid_price,
                time_in_force=TimeInForce.GTC,
                post_only=False,
            )
            self.submit_order(self._buy_order)

        # Place sell if not at max short
        if net > -self.config.max_position:
            ask_price = inst.make_price(fv + spread)
            self._sell_order = self.order_factory.limit(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.SELL,
                quantity=size,
                price=ask_price,
                time_in_force=TimeInForce.GTC,
                post_only=False,
            )
            self.submit_order(self._sell_order)

        self._last_quoted_fv = fv

    def _net_position(self) -> float:
        """Return net position in contracts (positive = long)."""
        positions = self.cache.positions_open(instrument_id=self.config.instrument_id)
        if not positions:
            return 0.0
        return sum(float(p.quantity) * (1 if p.is_long else -1) for p in positions)

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_trade_ticks(self.config.instrument_id)
