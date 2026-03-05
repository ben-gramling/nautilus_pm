"""
Deep-value Polymarket strategy: buy low-priced outcomes and hold.

Runs on a single configured Polymarket market slug and renders a legacy chart
for side-by-side strategy comparison.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from nautilus_trader.adapters.polymarket import POLYMARKET_VENUE
from nautilus_trader.adapters.polymarket import PolymarketDataLoader
from nautilus_trader.adapters.polymarket.fee_model import PolymarketFeeModel
from nautilus_trader.adapters.prediction_market.backtest_utils import build_market_prices
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_price_points
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_realized_pnl
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


NAME = "polymarket_deep_value_resolution_hold"
DESCRIPTION = "Buy below a configurable threshold and hold (single market)"

MARKET_SLUG = os.getenv(
    "MARKET_SLUG",
    "will-gavin-newsom-win-the-2028-democratic-presidential-nomination-568",
)
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "30"))
MIN_TRADES = int(os.getenv("MIN_TRADES", "200"))
CHART_RESAMPLE_RULE = os.getenv("CHART_RESAMPLE_RULE")

ENTRY_PRICE_MAX = float(os.getenv("ENTRY_PRICE_MAX", "0.247"))

TRADE_SIZE = Decimal(os.getenv("TRADE_SIZE", "100"))
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "100"))


class DeepValueHoldConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(20)
    entry_price_max: float = ENTRY_PRICE_MAX


class DeepValueHold(Strategy):
    """
    Buy YES when market price falls below threshold, then hold until near-resolution.
    """

    def __init__(self, config: DeepValueHoldConfig) -> None:
        super().__init__(config)
        self._instrument = None
        self._pending = False
        self._entered = False
        self._completed = False

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found; stopping")
            self.stop()
            return
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_trade_tick(self, tick: TradeTick) -> None:
        if self._pending or self._completed:
            return

        price = float(tick.price)

        if (
            self.portfolio.is_flat(self.config.instrument_id)
            and not self._entered
            and price <= self.config.entry_price_max
        ):
            self._buy()

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False
        if event.order_side == OrderSide.BUY:
            self._entered = True
        else:
            self._completed = True

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        # Force final exit for clean realized PnL and visible exit marker on chart.
        if self._entered and not self.portfolio.is_flat(self.config.instrument_id):
            self.close_all_positions(self.config.instrument_id)

    def on_reset(self) -> None:
        self._instrument = None
        self._pending = False
        self._entered = False
        self._completed = False

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


def _build_probability_frame(
    trades: list[TradeTick],
    entry_price_max: float,
) -> pd.DataFrame:
    rows: list[tuple[pd.Timestamp, float]] = []
    for tick in trades:
        ts = pd.to_datetime(int(tick.ts_event), unit="ns", utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append((ts, float(tick.price)))

    frame = pd.DataFrame(rows, columns=["ts", "market_probability"]) if rows else pd.DataFrame()
    if frame.empty:
        return frame

    frame = frame.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").set_index("ts")
    frame["market_probability"] = frame["market_probability"].clip(0.0, 1.0)

    signal_mask = frame["market_probability"] <= entry_price_max
    if signal_mask.any():
        first_signal_ts = frame.index[signal_mask.argmax()]
        user_probability = frame["market_probability"].copy()
        user_probability.loc[first_signal_ts:] = 1.0
    else:
        user_probability = frame["market_probability"].copy()

    frame["user_probability"] = user_probability
    frame["outcome"] = 1.0
    frame = frame.dropna(subset=["user_probability", "market_probability", "outcome"])

    return frame


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _run_backtest(
    slug: str,
    outcome: str,
    loader: PolymarketDataLoader,
    trades: list[TradeTick],
) -> dict[str, Any]:
    instrument = loader.instrument

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="WARNING"),
            risk_engine=RiskEngineConfig(bypass=True),
        ),
    )
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
        DeepValueHold(
            DeepValueHoldConfig(
                instrument_id=instrument.id,
                trade_size=TRADE_SIZE,
                entry_price_max=ENTRY_PRICE_MAX,
            ),
        ),
    )

    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    pnl = extract_realized_pnl(positions)

    prob_frame = _build_probability_frame(trades=trades, entry_price_max=ENTRY_PRICE_MAX)
    price_points = extract_price_points(trades, price_attr="price")

    safe_outcome = _slugify(outcome) or "outcome"
    output_path = f"output/{NAME}_{slug}_{safe_outcome}_legacy.html"
    os.makedirs("output", exist_ok=True)
    create_legacy_backtest_chart(
        engine=engine,
        output_path=output_path,
        strategy_name=f"{NAME}:{slug}:{outcome}",
        platform="polymarket",
        initial_cash=INITIAL_CASH,
        market_prices={
            str(instrument.id): build_market_prices(
                price_points,
                resample_rule=CHART_RESAMPLE_RULE or None,
            )
        },
        user_probabilities=prob_frame.get("user_probability"),
        market_probabilities=prob_frame.get("market_probability"),
        outcomes=prob_frame.get("outcome"),
        open_browser=False,
        progress=False,
    )

    engine.reset()
    engine.dispose()

    prices = [float(t.price) for t in trades]
    return {
        "slug": slug,
        "outcome": outcome,
        "output_path": output_path,
        "trades": len(trades),
        "fills": len(fills),
        "pnl": pnl,
        "entry_min": min(prices),
        "max": max(prices),
        "last": prices[-1],
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No qualifying resolved low-priced outcome markets were found.")
        return

    labels = [f"{r['slug']}:{r['outcome']}" for r in results]
    col_w = max(len(label) for label in labels) + 2
    header = (
        f"{'Market':<{col_w}} {'Trades':>7} {'Fills':>6} "
        f"{'Min Px':>8} {'Max Px':>8} {'Last Px':>8} {'PnL (USDC)':>12}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}\n{header}\n{sep}")
    for row, label in zip(results, labels, strict=False):
        print(
            f"{label:<{col_w}} {row['trades']:>7} {row['fills']:>6} "
            f"{row['entry_min']:>8.3f} {row['max']:>8.3f} {row['last']:>8.3f} "
            f"{row['pnl']:>+12.4f}"
        )

    total_pnl = sum(float(r["pnl"]) for r in results)
    total_fills = sum(int(r["fills"]) for r in results)
    print(sep)
    print(
        f"{'TOTAL':<{col_w}} {'':>7} {total_fills:>6} {'':>8} {'':>8} {'':>8} "
        f"{total_pnl:>+12.4f}"
    )
    print(sep)


async def run() -> None:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=LOOKBACK_DAYS))
    end = pd.Timestamp(now)

    print(
        f"Loading Polymarket market {MARKET_SLUG} "
        f"(lookback={LOOKBACK_DAYS}d)..."
    )
    try:
        loader = await PolymarketDataLoader.from_market_slug(MARKET_SLUG)
        trades = await loader.load_trades(start, end)
    except Exception as exc:
        print(f"Unable to load {MARKET_SLUG}: {exc}")
        return

    if len(trades) < MIN_TRADES:
        print(f"Skip {MARKET_SLUG}: {len(trades)} trades < {MIN_TRADES} required")
        return

    outcome = str(getattr(loader.instrument, "outcome", "yes"))
    print(f"  running backtest for {MARKET_SLUG}:{outcome}...")
    result = _run_backtest(
        slug=MARKET_SLUG,
        outcome=outcome,
        loader=loader,
        trades=trades,
    )

    _print_summary([result])
    print(f"\nLegacy chart saved to {result['output_path']}")


if __name__ == "__main__":
    asyncio.run(run())
