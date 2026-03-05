"""
Shared runner for single-market Polymarket trade-tick backtests.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pandas as pd

from nautilus_trader.adapters.polymarket import POLYMARKET_VENUE
from nautilus_trader.adapters.polymarket import PolymarketDataLoader
from nautilus_trader.adapters.polymarket.fee_model import PolymarketFeeModel
from nautilus_trader.adapters.prediction_market.research import print_backtest_summary
from nautilus_trader.adapters.prediction_market.research import run_market_backtest
from nautilus_trader.model.currencies import USDC_POS
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


type StrategyFactory = Callable[[InstrumentId], Strategy]


async def run_single_market_trade_backtest(
    *,
    name: str,
    market_slug: str,
    lookback_days: int,
    strategy_factory: StrategyFactory,
    probability_window: int,
    min_trades: int = 0,
    min_price_range: float = 0.0,
    initial_cash: float = 100.0,
    chart_resample_rule: str | None = None,
) -> None:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=lookback_days))
    end = pd.Timestamp(now)

    print(
        f"Loading Polymarket market {market_slug} "
        f"(lookback={lookback_days}d)..."
    )

    try:
        loader = await PolymarketDataLoader.from_market_slug(market_slug)
        trades = await loader.load_trades(start, end)
    except Exception as exc:
        print(f"Unable to load Polymarket market {market_slug}: {exc}")
        return

    if len(trades) < min_trades:
        print(f"Skip {market_slug}: {len(trades)} trades < {min_trades} required")
        return

    prices = [float(tick.price) for tick in trades]
    if prices:
        price_range = max(prices) - min(prices)
        if price_range < min_price_range:
            print(
                f"Skip {market_slug}: price range {price_range:.3f} "
                f"< {min_price_range:.3f}"
            )
            return

    if not trades:
        print(f"No trades returned for {market_slug}")
        return

    result = run_market_backtest(
        market_id=market_slug,
        instrument=loader.instrument,
        data=trades,
        strategy=strategy_factory(loader.instrument.id),
        strategy_name=f"{name}:{market_slug}",
        output_prefix=name,
        platform="polymarket",
        venue=POLYMARKET_VENUE,
        base_currency=USDC_POS,
        fee_model=PolymarketFeeModel(),
        initial_cash=initial_cash,
        probability_window=probability_window,
        price_attr="price",
        count_key="trades",
        chart_resample_rule=chart_resample_rule,
        market_key="slug",
    )

    print_backtest_summary(
        results=[result],
        market_key="slug",
        count_key="trades",
        count_label="Trades",
        pnl_label="PnL (USDC)",
    )
    print(f"\nLegacy chart saved to output/{name}_{market_slug}_legacy.html")
