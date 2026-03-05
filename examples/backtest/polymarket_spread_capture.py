"""
Mean-reversion spread capture on Polymarket trade ticks.

Discovers active markets from the Polymarket Gamma API, fetches trade ticks,
and runs a VWAP-based mean-reversion backtest across multiple markets.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from decimal import Decimal

import pandas as pd

from nautilus_trader.adapters.polymarket.spread_capture import discover_spread_capture_slugs
from nautilus_trader.adapters.polymarket.spread_capture import load_spread_capture_market
from nautilus_trader.adapters.polymarket.spread_capture import print_spread_capture_summary
from nautilus_trader.adapters.polymarket.spread_capture import run_spread_capture_backtest
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig


# ── Strategy metadata (shown in the menu) ────────────────────────────────────
NAME = "polymarket_spread_capture"
DESCRIPTION = "Mean-reversion spread capture across Polymarket markets"

# ── Configure here ────────────────────────────────────────────────────────────
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "14"))
MAX_MARKETS = int(os.getenv("MAX_MARKETS", "15"))
# Prefer high-density markets so charts have richer point counts.
MIN_TRADES = int(os.getenv("MIN_TRADES", "200"))
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "120"))
MIN_PRICE_RANGE = float(os.getenv("MIN_PRICE_RANGE", "0.15"))
VWAP_WINDOW = 20
ENTRY_THRESHOLD = 0.005    # 0.5% deviation from rolling avg to enter
TAKE_PROFIT = 0.008        # 0.8% recovery to exit with profit
STOP_LOSS = 0.020          # 2.0% adverse move to cut loss
# Only trade markets whose YES price is in this range — avoids near-resolved markets
PRICE_MIN = 0.25
PRICE_MAX = 0.75
TRADE_SIZE = Decimal(20)
INITIAL_CASH = 1_000.0
# ─────────────────────────────────────────────────────────────────────────────


class SpreadCaptureConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(20)
    vwap_window: int = 20
    entry_threshold: float = ENTRY_THRESHOLD
    take_profit: float = TAKE_PROFIT
    stop_loss: float = STOP_LOSS


class SpreadCapture(Strategy):
    """
    Mean-reversion spread capture strategy.

    Buys when price dips below a rolling average and exits on recovery
    or stop-loss.  Holds at most one position at a time.
    """

    def __init__(self, config: SpreadCaptureConfig) -> None:
        super().__init__(config)
        self._prices: deque[float] = deque(maxlen=config.vwap_window)
        self._entry_price: float | None = None
        self._pending: bool = False
        self._instrument = None

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(
                f"Instrument {self.config.instrument_id} not found — stopping."
            )
            self.stop()
            return
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        self._prices.append(price)

        if len(self._prices) < self.config.vwap_window or self._pending:
            return

        rolling_avg = sum(self._prices) / len(self._prices)

        if self.portfolio.is_flat(self.config.instrument_id):
            if price <= rolling_avg - self.config.entry_threshold:
                self._buy()
        else:
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

async def run() -> None:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=LOOKBACK_DAYS))
    end = pd.Timestamp(now)

    print(f"Discovering top {CANDIDATE_LIMIT} active Polymarket markets by volume...")
    slugs = await discover_spread_capture_slugs(
        candidate_limit=CANDIDATE_LIMIT,
        lookback_days=LOOKBACK_DAYS,
        price_min=PRICE_MIN,
        price_max=PRICE_MAX,
    )
    print(f"Found {len(slugs)} markets → fetching trades in parallel...\n")

    loaded = await asyncio.gather(
        *[
            load_spread_capture_market(
                slug=slug,
                start=start,
                end=end,
                min_trades=MIN_TRADES,
                min_price_range=MIN_PRICE_RANGE,
            )
            for slug in slugs
        ]
    )

    results: list[dict] = []
    for slug, market_data in zip(slugs, loaded, strict=False):
        if len(results) >= MAX_MARKETS:
            break
        if market_data is None:
            continue
        loader, trades = market_data
        print(f"  {slug}: {len(trades)} trades → running backtest...")
        result = run_spread_capture_backtest(
            slug=slug,
            loader=loader,
            trades=trades,
            strategy=SpreadCapture(
                SpreadCaptureConfig(
                    instrument_id=loader.instrument.id,
                    trade_size=TRADE_SIZE,
                    vwap_window=VWAP_WINDOW,
                    entry_threshold=ENTRY_THRESHOLD,
                    take_profit=TAKE_PROFIT,
                    stop_loss=STOP_LOSS,
                )
            ),
            strategy_name=f"{NAME}:{slug}",
            output_prefix=NAME,
            initial_cash=INITIAL_CASH,
            probability_window=VWAP_WINDOW,
        )
        results.append(result)

    print_spread_capture_summary(results)
    print(f"\nLegacy charts saved to output/{NAME}_<slug>_legacy.html")


if __name__ == "__main__":
    asyncio.run(run())
