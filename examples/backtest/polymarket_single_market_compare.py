"""
Compare spread-capture vs hold-20c on one Polymarket market.

Generates two legacy charts on the same trade-tick set:
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

from examples.backtest.polymarket_spread_capture import ENTRY_THRESHOLD
from examples.backtest.polymarket_spread_capture import STOP_LOSS
from examples.backtest.polymarket_spread_capture import TAKE_PROFIT
from examples.backtest.polymarket_spread_capture import TRADE_SIZE
from examples.backtest.polymarket_spread_capture import VWAP_WINDOW
from examples.backtest.polymarket_spread_capture import SpreadCapture
from examples.backtest.polymarket_spread_capture import SpreadCaptureConfig
from examples.backtest.polymarket_spread_capture import _build_brier_inputs_from_trades
from examples.backtest.polymarket_spread_capture import _build_market_prices_from_trades
from examples.backtest.polymarket_spread_capture import _extract_pnl
from nautilus_trader.adapters.polymarket import POLYMARKET_VENUE
from nautilus_trader.adapters.polymarket import PolymarketDataLoader
from nautilus_trader.adapters.polymarket.fee_model import PolymarketFeeModel
from nautilus_trader.analysis.legacy_plot_adapter import create_legacy_backtest_chart
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDC_POS
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Money
from nautilus_trader.risk.config import RiskEngineConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig


SLUG = os.getenv(
    "POLYMARKET_SLUG",
    "will-the-oklahoma-city-thunder-win-the-2026-nba-finals",
)
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "1000"))
HOLD_ENTRY_MAX = float(os.getenv("HOLD_ENTRY_MAX", "0.20"))
HOLD_SIZE = Decimal(os.getenv("HOLD_SIZE", "20"))


class Hold20Config(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(20)
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
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_trade_tick(self, tick: TradeTick) -> None:
        if self._pending or self._entered:
            return
        if float(tick.price) <= self.config.entry_max:
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


def _build_hold_brier_inputs_from_trades(
    trades: list[TradeTick],
    entry_max: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    empty = pd.Series(dtype=float)
    if not trades:
        return empty, empty, empty

    rows: list[tuple[pd.Timestamp, float]] = []
    for tick in trades:
        ts_ns = getattr(tick, "ts_event", None) or getattr(tick, "ts_init", None)
        if ts_ns is None:
            continue
        try:
            ts = pd.to_datetime(int(ts_ns), unit="ns", utc=True)
            px = float(tick.price)
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


async def _load_market_data(slug: str) -> tuple[PolymarketDataLoader, list[TradeTick]]:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=LOOKBACK_DAYS))
    end = pd.Timestamp(now)

    loader = await PolymarketDataLoader.from_market_slug(slug)
    trades = await loader.load_trades(start=start, end=end)
    if not trades:
        raise RuntimeError(f"No trades loaded for {slug}")

    return loader, trades


def _new_engine() -> BacktestEngine:
    return BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="WARNING"),
            risk_engine=RiskEngineConfig(bypass=True),
        ),
    )


def _run_spread(
    loader: PolymarketDataLoader,
    trades: list[TradeTick],
) -> dict[str, float | int | str]:
    instrument = loader.instrument
    engine = _new_engine()

    engine.add_venue(
        venue=POLYMARKET_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USDC_POS,
        starting_balances=[Money(INITIAL_CASH, USDC_POS)],
        fee_model=PolymarketFeeModel(),
    )
    engine.add_instrument(instrument)
    engine.add_data(trades)
    engine.add_strategy(
        SpreadCapture(
            SpreadCaptureConfig(
                instrument_id=instrument.id,
                trade_size=TRADE_SIZE,
                vwap_window=VWAP_WINDOW,
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
    u, m, o = _build_brier_inputs_from_trades(trades=trades, window=VWAP_WINDOW)

    output = f"output/polymarket_compare_{SLUG}_spread_legacy.html"
    create_legacy_backtest_chart(
        engine=engine,
        output_path=output,
        strategy_name=f"polymarket_compare:{SLUG}:spread",
        platform="polymarket",
        initial_cash=INITIAL_CASH,
        market_prices={str(instrument.id): _build_market_prices_from_trades(trades)},
        user_probabilities=u,
        market_probabilities=m,
        outcomes=o,
        open_browser=False,
    )
    engine.reset()
    engine.dispose()

    return {"strategy": "spread", "fills": len(fills), "pnl": pnl, "output": output}


def _run_hold(
    loader: PolymarketDataLoader,
    trades: list[TradeTick],
) -> dict[str, float | int | str]:
    instrument = loader.instrument
    engine = _new_engine()

    engine.add_venue(
        venue=POLYMARKET_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USDC_POS,
        starting_balances=[Money(INITIAL_CASH, USDC_POS)],
        fee_model=PolymarketFeeModel(),
    )
    engine.add_instrument(instrument)
    engine.add_data(trades)
    engine.add_strategy(
        Hold20(
            Hold20Config(
                instrument_id=instrument.id,
                trade_size=HOLD_SIZE,
                entry_max=HOLD_ENTRY_MAX,
            ),
        ),
    )
    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    pnl = _extract_pnl(positions)
    u, m, o = _build_hold_brier_inputs_from_trades(trades=trades, entry_max=HOLD_ENTRY_MAX)

    output = f"output/polymarket_compare_{SLUG}_hold20_legacy.html"
    create_legacy_backtest_chart(
        engine=engine,
        output_path=output,
        strategy_name=f"polymarket_compare:{SLUG}:hold20",
        platform="polymarket",
        initial_cash=INITIAL_CASH,
        market_prices={str(instrument.id): _build_market_prices_from_trades(trades)},
        user_probabilities=u,
        market_probabilities=m,
        outcomes=o,
        open_browser=False,
    )
    engine.reset()
    engine.dispose()

    return {"strategy": "hold20", "fills": len(fills), "pnl": pnl, "output": output}


async def run() -> None:
    loader, trades = await _load_market_data(SLUG)
    print(f"Loaded {len(trades)} trades for {SLUG}")

    spread = _run_spread(loader, trades)
    hold = _run_hold(loader, trades)

    print("\nStrategy comparison")
    print("-" * 72)
    print(f"{'Strategy':<10} {'Fills':>8} {'PnL (USDC)':>12}   Output")
    print("-" * 72)
    for row in (spread, hold):
        print(
            f"{row['strategy']!s:<10} {int(row['fills']):>8} "
            f"{float(row['pnl']):>+12.4f}   {row['output']}",
        )
    print("-" * 72)


if __name__ == "__main__":
    asyncio.run(run())
