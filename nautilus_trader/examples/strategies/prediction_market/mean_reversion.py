# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software distributed under the
#  License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Protocol

from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig


class _MeanReversionConfig(Protocol):
    instrument_id: InstrumentId
    trade_size: Decimal
    entry_threshold: float
    take_profit: float
    stop_loss: float


class BarMeanReversionConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal(1)
    window: int = 20
    entry_threshold: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0


class TradeTickMeanReversionConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    vwap_window: int = 20
    entry_threshold: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0


class _MeanReversionBase(Strategy):
    """
    Single-instrument mean-reversion base with one open position max.
    """

    _window_field = "window"

    def __init__(self, config: _MeanReversionConfig) -> None:
        super().__init__(config)
        self._prices: deque[float] = deque(maxlen=self._window())
        self._entry_price: float | None = None
        self._pending: bool = False
        self._instrument = None

    def _window(self) -> int:
        return int(getattr(self.config, self._window_field))

    def _subscribe(self) -> None:
        raise NotImplementedError

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found - stopping.")
            self.stop()
            return
        self._subscribe()

    def _on_price(self, price: float) -> None:
        self._prices.append(price)
        if len(self._prices) < self._window() or self._pending:
            return

        rolling_avg = sum(self._prices) / len(self._prices)
        if self.portfolio.is_flat(self.config.instrument_id):
            if price <= rolling_avg - self.config.entry_threshold:
                self._buy()
            return

        if self._entry_price is None:
            return

        take_profit_hit = price >= self._entry_price + self.config.take_profit
        stop_loss_hit = price <= self._entry_price - self.config.stop_loss
        if take_profit_hit or stop_loss_hit:
            self.close_all_positions(self.config.instrument_id)
            self._pending = True

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.order_side == OrderSide.BUY:
            self._entry_price = float(event.last_px)
        else:
            self._entry_price = None
        self._pending = False

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)

    def on_reset(self) -> None:
        self._prices.clear()
        self._entry_price = None
        self._pending = False
        self._instrument = None

    def _buy(self) -> None:
        assert self._instrument is not None
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self._instrument.make_qty(float(self.config.trade_size)),
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)
        self._pending = True


class BarMeanReversionStrategy(_MeanReversionBase):
    def _subscribe(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self._on_price(float(bar.close))


class TradeTickMeanReversionStrategy(_MeanReversionBase):
    _window_field = "vwap_window"

    def _subscribe(self) -> None:
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_trade_tick(self, tick: TradeTick) -> None:
        self._on_price(float(tick.price))
