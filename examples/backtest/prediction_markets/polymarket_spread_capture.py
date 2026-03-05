"""
Mean-reversion spread capture on Polymarket trade ticks.

Discovers active markets from the Polymarket Gamma API, fetches trade ticks,
and runs a VWAP-based mean-reversion backtest across multiple markets.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from decimal import Decimal

import pandas as pd

from nautilus_trader.adapters.polymarket import POLYMARKET_VENUE
from nautilus_trader.adapters.polymarket.fee_model import PolymarketFeeModel
from nautilus_trader.adapters.polymarket.research import discover_markets
from nautilus_trader.adapters.polymarket.research import load_market_trades
from nautilus_trader.adapters.prediction_market.research import print_backtest_summary
from nautilus_trader.adapters.prediction_market.research import run_market_backtest
from nautilus_trader.examples.strategies.prediction_market.mean_reversion import (
    TradeTickMeanReversionConfig as SpreadCaptureConfig,
)
from nautilus_trader.examples.strategies.prediction_market.mean_reversion import (
    TradeTickMeanReversionStrategy as SpreadCapture,
)
from nautilus_trader.model.currencies import USDC_POS


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


async def run() -> None:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=LOOKBACK_DAYS))
    end = pd.Timestamp(now)

    print(f"Discovering top {CANDIDATE_LIMIT} active Polymarket markets by volume...")
    markets = await discover_markets(
        candidate_limit=CANDIDATE_LIMIT,
        min_volume_24h=0.0,
        yes_price_min=PRICE_MIN,
        yes_price_max=PRICE_MAX,
        min_days_to_expiry=LOOKBACK_DAYS,
    )
    slugs = [str(market.get("slug", "")) for market in markets if market.get("slug")]
    print(f"Found {len(slugs)} markets → fetching trades in parallel...\n")

    loaded = await asyncio.gather(
        *[
            load_market_trades(
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
        result = run_market_backtest(
            market_id=slug,
            instrument=loader.instrument,
            data=trades,
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
            platform="polymarket",
            venue=POLYMARKET_VENUE,
            base_currency=USDC_POS,
            fee_model=PolymarketFeeModel(),
            initial_cash=INITIAL_CASH,
            probability_window=VWAP_WINDOW,
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
    print(f"\nLegacy charts saved to output/{NAME}_<slug>_legacy.html")


if __name__ == "__main__":
    asyncio.run(run())
