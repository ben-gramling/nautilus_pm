"""
Compare spread-capture vs hold-20c on one Kalshi market.

Generates two legacy charts on the same bar set:
1) Mean-reversion spread capture
2) Buy at <= 20c and hold until dataset end
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from decimal import Decimal

import pandas as pd

from examples.backtest.kalshi_spread_capture import ENTRY_THRESHOLD
from examples.backtest.kalshi_spread_capture import STOP_LOSS
from examples.backtest.kalshi_spread_capture import TAKE_PROFIT
from examples.backtest.kalshi_spread_capture import TRADE_SIZE
from examples.backtest.kalshi_spread_capture import WINDOW
from examples.backtest.kalshi_spread_capture import BarMeanReversion
from examples.backtest.kalshi_spread_capture import BarMeanReversionConfig
from examples.backtest.kalshi_spread_capture import _build_brier_inputs_from_bars
from examples.backtest.kalshi_spread_capture import _build_market_prices_from_bars
from examples.backtest.kalshi_spread_capture import _extract_pnl
from nautilus_trader.adapters.kalshi.fee_model import KalshiProportionalFeeModel
from nautilus_trader.adapters.kalshi.loaders import KalshiDataLoader
from nautilus_trader.analysis.legacy_plot_adapter import create_legacy_backtest_chart
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.risk.config import RiskEngineConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig


TICKER = os.getenv("KALSHI_MARKET_TICKER", "KXNEXTIRANLEADER-45JAN01-MKHA")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "1000"))
HOLD_ENTRY_MAX = float(os.getenv("HOLD_ENTRY_MAX", "0.20"))
HOLD_SIZE = Decimal(os.getenv("HOLD_SIZE", "1"))


class Hold20Config(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal(1)
    entry_max: float = 0.20


class Hold20(Strategy):
    def __init__(self, config: Hold20Config) -> None:
        super().__init__(config)
        self._instrument = None
        self._pending = False
        self._entered = False

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found")
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if self._pending or self._entered:
            return
        if float(bar.close) <= self.config.entry_max:
            self._buy()

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False
        if event.order_side == OrderSide.BUY:
            self._entered = True

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self._entered and not self.portfolio.is_flat(self.config.instrument_id):
            self.close_all_positions(self.config.instrument_id)

    def on_reset(self) -> None:
        self._instrument = None
        self._pending = False
        self._entered = False

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


def _build_hold_brier_inputs_from_bars(
    bars: list[Bar],
    entry_max: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    empty = pd.Series(dtype=float)
    if not bars:
        return empty, empty, empty

    rows: list[tuple[pd.Timestamp, float]] = []
    for bar in bars:
        ts_ns = getattr(bar, "ts_event", None) or getattr(bar, "ts_init", None)
        if ts_ns is None:
            continue
        try:
            ts = pd.to_datetime(int(ts_ns), unit="ns", utc=True)
            px = float(bar.close)
        except (TypeError, ValueError):
            continue
        rows.append((ts, px))

    if not rows:
        return empty, empty, empty

    frame = pd.DataFrame(rows, columns=["ts", "market_probability"])
    frame = (
        frame.dropna()
        .sort_values("ts")
        .drop_duplicates(subset=["ts"], keep="last")
        .set_index("ts")
    )
    if frame.empty:
        return empty, empty, empty

    frame["market_probability"] = frame["market_probability"].clip(0.0, 1.0)
    signal = frame["market_probability"] <= entry_max
    user = frame["market_probability"].copy()
    if signal.any():
        first_idx = int(signal.to_numpy().argmax())
        user.iloc[first_idx:] = 1.0
    frame["user_probability"] = user
    frame["outcome"] = float(frame["market_probability"].iloc[-1] >= 0.5)

    return (
        frame["user_probability"].copy(),
        frame["market_probability"].copy(),
        frame["outcome"].copy(),
    )


async def _load_market_data(ticker: str) -> tuple[KalshiDataLoader, list[Bar]]:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=LOOKBACK_DAYS))
    end = pd.Timestamp(now)

    client = nautilus_pyo3.HttpClient(
        default_quota=nautilus_pyo3.Quota.rate_per_second(10),
    )
    loader = await KalshiDataLoader.from_market_ticker(ticker, http_client=client)

    bars: list[Bar] = []
    chunk = pd.Timedelta(minutes=5000)
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + chunk, end)
        bars.extend(await loader.load_bars(start=chunk_start, end=chunk_end, interval="Minutes1"))
        chunk_start = chunk_end

    if not bars:
        raise RuntimeError(f"No bars loaded for {ticker}")

    return loader, bars


def _new_engine() -> BacktestEngine:
    return BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="WARNING"),
            risk_engine=RiskEngineConfig(bypass=True),
        ),
    )


def _run_spread(loader: KalshiDataLoader, bars: list[Bar]) -> dict[str, float | int | str]:
    instrument = loader.instrument
    bar_type = bars[0].bar_type
    engine = _new_engine()

    engine.add_venue(
        venue=Venue("KALSHI"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(INITIAL_CASH, USD)],
        fee_model=KalshiProportionalFeeModel(),
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(
        BarMeanReversion(
            BarMeanReversionConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
                trade_size=TRADE_SIZE,
                window=WINDOW,
                entry_threshold=ENTRY_THRESHOLD,
                take_profit=TAKE_PROFIT,
                stop_loss=STOP_LOSS,
            ),
        ),
    )
    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    pnl = _extract_pnl(positions)
    u, m, o = _build_brier_inputs_from_bars(bars=bars, window=WINDOW)

    output = f"output/kalshi_compare_{TICKER}_spread_legacy.html"
    create_legacy_backtest_chart(
        engine=engine,
        output_path=output,
        strategy_name=f"kalshi_compare:{TICKER}:spread",
        platform="kalshi",
        initial_cash=INITIAL_CASH,
        market_prices={str(instrument.id): _build_market_prices_from_bars(bars)},
        user_probabilities=u,
        market_probabilities=m,
        outcomes=o,
        open_browser=False,
    )
    engine.reset()
    engine.dispose()

    return {"strategy": "spread", "fills": len(fills), "pnl": pnl, "output": output}


def _run_hold(loader: KalshiDataLoader, bars: list[Bar]) -> dict[str, float | int | str]:
    instrument = loader.instrument
    bar_type = bars[0].bar_type
    engine = _new_engine()

    engine.add_venue(
        venue=Venue("KALSHI"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(INITIAL_CASH, USD)],
        fee_model=KalshiProportionalFeeModel(),
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(
        Hold20(
            Hold20Config(
                instrument_id=instrument.id,
                bar_type=bar_type,
                trade_size=HOLD_SIZE,
                entry_max=HOLD_ENTRY_MAX,
            ),
        ),
    )
    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    pnl = _extract_pnl(positions)
    u, m, o = _build_hold_brier_inputs_from_bars(bars=bars, entry_max=HOLD_ENTRY_MAX)

    output = f"output/kalshi_compare_{TICKER}_hold20_legacy.html"
    create_legacy_backtest_chart(
        engine=engine,
        output_path=output,
        strategy_name=f"kalshi_compare:{TICKER}:hold20",
        platform="kalshi",
        initial_cash=INITIAL_CASH,
        market_prices={str(instrument.id): _build_market_prices_from_bars(bars)},
        user_probabilities=u,
        market_probabilities=m,
        outcomes=o,
        open_browser=False,
    )
    engine.reset()
    engine.dispose()

    return {"strategy": "hold20", "fills": len(fills), "pnl": pnl, "output": output}


async def run() -> None:
    loader, bars = await _load_market_data(TICKER)
    print(f"Loaded {len(bars)} bars for {TICKER}")

    spread = _run_spread(loader, bars)
    hold = _run_hold(loader, bars)

    print("\nStrategy comparison")
    print("-" * 72)
    print(f"{'Strategy':<10} {'Fills':>8} {'PnL (USD)':>12}   Output")
    print("-" * 72)
    for row in (spread, hold):
        print(
            f"{row['strategy']!s:<10} {int(row['fills']):>8} "
            f"{float(row['pnl']):>+12.4f}   {row['output']}",
        )
    print("-" * 72)


if __name__ == "__main__":
    asyncio.run(run())
