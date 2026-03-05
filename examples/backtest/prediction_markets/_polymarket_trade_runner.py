"""
Shared runner for multi-market Polymarket trade-tick backtests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pandas as pd

from nautilus_trader.adapters.polymarket import POLYMARKET_VENUE
from nautilus_trader.adapters.polymarket.fee_model import PolymarketFeeModel
from nautilus_trader.adapters.polymarket.research import discover_markets
from nautilus_trader.adapters.polymarket.research import load_market_trades
from nautilus_trader.adapters.prediction_market.research import print_backtest_summary
from nautilus_trader.adapters.prediction_market.research import run_market_backtest
from nautilus_trader.model.currencies import USDC_POS
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


type StrategyFactory = Callable[[InstrumentId], Strategy]


async def run_multi_market_trade_backtest(
    *,
    name: str,
    lookback_days: int,
    strategy_factory: StrategyFactory,
    probability_window: int,
    candidate_limit: int = 120,
    max_markets: int = 10,
    min_trades: int = 200,
    min_price_range: float = 0.15,
    min_volume_24h: float = 0.0,
    yes_price_min: float | None = None,
    yes_price_max: float | None = None,
    min_days_to_expiry: int | None = None,
    initial_cash: float = 1_000.0,
) -> None:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=lookback_days))
    end = pd.Timestamp(now)

    min_expiry_days = lookback_days if min_days_to_expiry is None else min_days_to_expiry
    print(f"Discovering top {candidate_limit} active Polymarket markets by volume...")
    markets = await discover_markets(
        candidate_limit=candidate_limit,
        min_volume_24h=min_volume_24h,
        yes_price_min=yes_price_min,
        yes_price_max=yes_price_max,
        min_days_to_expiry=min_expiry_days,
    )
    slugs = [str(market.get("slug", "")) for market in markets if market.get("slug")]
    if not slugs:
        print("No candidate Polymarket slugs discovered.")
        return

    print(f"Found {len(slugs)} markets → fetching trades in parallel...\n")
    loaded = await asyncio.gather(
        *[
            load_market_trades(
                slug=slug,
                start=start,
                end=end,
                min_trades=min_trades,
                min_price_range=min_price_range,
            )
            for slug in slugs
        ]
    )

    results: list[dict] = []
    for slug, market_data in zip(slugs, loaded, strict=False):
        if len(results) >= max_markets:
            break
        if market_data is None:
            continue

        loader, trades = market_data
        print(f"  {slug}: {len(trades)} trades → running backtest...")
        result = run_market_backtest(
            market_id=slug,
            instrument=loader.instrument,
            data=trades,
            strategy=strategy_factory(loader.instrument.id),
            strategy_name=f"{name}:{slug}",
            output_prefix=name,
            platform="polymarket",
            venue=POLYMARKET_VENUE,
            base_currency=USDC_POS,
            fee_model=PolymarketFeeModel(),
            initial_cash=initial_cash,
            probability_window=probability_window,
            price_attr="price",
            count_key="trades",
            market_key="slug",
        )
        results.append(result)

    print_backtest_summary(
        results=results,
        market_key="slug",
        count_key="trades",
        count_label="Trades",
        pnl_label="PnL (USDC)",
    )
    print(f"\nLegacy charts saved to output/{name}_<slug>_legacy.html")
