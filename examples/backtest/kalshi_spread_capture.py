"""
Bar-based mean-reversion (spread capture) on Kalshi minute bars.

Discovers active Kalshi markets using the same method as the Polymarket
spread capture: ranks by 24-hour volume, filters to markets with genuine
uncertainty (price in [PRICE_MIN, PRICE_MAX]) and enough time before
resolution, then runs a BarMeanReversion strategy on the top MAX_MARKETS
that have at least MIN_BARS of bar history.
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

from nautilus_trader.adapters.kalshi.spread_capture import discover_spread_capture_markets
from nautilus_trader.adapters.kalshi.spread_capture import load_spread_capture_market
from nautilus_trader.adapters.kalshi.spread_capture import print_spread_capture_summary
from nautilus_trader.adapters.kalshi.spread_capture import run_spread_capture_backtest
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig


# ── Strategy metadata (shown in the menu) ────────────────────────────────────
NAME = "kalshi_spread_capture"
DESCRIPTION = "Mean-reversion spread capture across Kalshi markets"

# ── Configure here ────────────────────────────────────────────────────────────
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))
# Prefer high-density markets so charts have rich point counts like MKHA.
MIN_BARS = int(os.getenv("MIN_BARS", "1000"))
MAX_MARKETS = int(os.getenv("MAX_MARKETS", "15"))
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "400"))
MIN_PRICE_RANGE = float(os.getenv("MIN_PRICE_RANGE", "0.15"))
# Stop scanning after this many candidates even if MAX_MARKETS is not reached.
MAX_SCANNED_MARKETS = int(os.getenv("MAX_SCANNED_MARKETS", "120"))
# Only trade markets whose YES price is in this range — avoids fully-resolved markets
PRICE_MIN = 0.05
PRICE_MAX = 0.95
# Markets must be at least this many days from resolution to avoid end-of-life drift
MIN_DAYS_TO_RESOLUTION = 2

WINDOW = 20  # rolling average window
ENTRY_THRESHOLD = 0.01  # enter when close is 1¢ below rolling average (0-1 scale)
TAKE_PROFIT = 0.01  # exit when price recovers 1¢ above fill price
STOP_LOSS = 0.03  # stop out 3¢ below fill price
TRADE_SIZE = Decimal(1)
INITIAL_CASH = 1_000.0
MAX_RETRIES = 4  # retry 429s up to this many times
RETRY_BASE_DELAY = 2.0  # seconds; doubles on each retry
# ─────────────────────────────────────────────────────────────────────────────


class BarMeanReversionConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal(1)
    window: int = 20
    entry_threshold: float = 1.0
    take_profit: float = 1.0
    stop_loss: float = 3.0


class BarMeanReversion(Strategy):
    """
    Mean-reversion spread capture on bar close prices.

    Buys when close dips below a rolling average by `entry_threshold`,
    exits when price recovers `take_profit` above fill, or stops out
    `stop_loss` below fill.  Holds at most one position at a time.
    """

    def __init__(self, config: BarMeanReversionConfig) -> None:
        super().__init__(config)
        self._prices: deque[float] = deque(maxlen=config.window)
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
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        price = float(bar.close)
        self._prices.append(price)

        if len(self._prices) < self.config.window or self._pending:
            return

        avg = sum(self._prices) / len(self._prices)

        if self.portfolio.is_flat(self.config.instrument_id):
            if price <= avg - self.config.entry_threshold:
                self._buy()
        else:
            assert self._entry_price is not None
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

    # Single shared client — conservative rate to avoid 429s.
    http_client = nautilus_pyo3.HttpClient(
        default_quota=nautilus_pyo3.Quota.rate_per_second(10),
    )

    print(f"Discovering top {MAX_MARKETS} active Kalshi markets by 24h volume...")
    candidates = await discover_spread_capture_markets(
        candidate_limit=CANDIDATE_LIMIT,
        http_client=http_client,
        price_min=PRICE_MIN,
        price_max=PRICE_MAX,
        min_days_to_resolution=MIN_DAYS_TO_RESOLUTION,
    )
    print(f"Found {len(candidates)} markets → scanning for {MIN_BARS}+ bars...")

    # Brief pause to let the rate-limit window reset after discovery.
    await asyncio.sleep(2)

    results: list[dict] = []
    scanned = 0
    for market in candidates:
        if len(results) >= MAX_MARKETS:
            break
        if scanned >= MAX_SCANNED_MARKETS:
            print(
                f"Reached scan cap ({MAX_SCANNED_MARKETS}) with "
                f"{len(results)} qualifying markets."
            )
            break
        scanned += 1
        market_data = await load_spread_capture_market(
            market=market,
            start=start,
            end=end,
            http_client=http_client,
            min_bars=MIN_BARS,
            min_price_range=MIN_PRICE_RANGE,
            max_retries=MAX_RETRIES,
            retry_base_delay=RETRY_BASE_DELAY,
        )
        if market_data is None:
            continue
        loader, bars = market_data
        ticker = market["ticker"]
        print(f"  {ticker}: {len(bars)} bars → running backtest...")
        bar_type = bars[0].bar_type
        result = run_spread_capture_backtest(
            ticker=ticker,
            loader=loader,
            bars=bars,
            strategy=BarMeanReversion(
                config=BarMeanReversionConfig(
                    instrument_id=loader.instrument.id,
                    bar_type=bar_type,
                    trade_size=TRADE_SIZE,
                    window=WINDOW,
                    entry_threshold=ENTRY_THRESHOLD,
                    take_profit=TAKE_PROFIT,
                    stop_loss=STOP_LOSS,
                )
            ),
            strategy_name=f"{NAME}:{ticker}",
            output_prefix=NAME,
            initial_cash=INITIAL_CASH,
            probability_window=WINDOW,
        )
        results.append(result)

    print_spread_capture_summary(results)
    print(f"\nLegacy charts saved to output/{NAME}_<ticker>_legacy.html")


if __name__ == "__main__":
    asyncio.run(run())
