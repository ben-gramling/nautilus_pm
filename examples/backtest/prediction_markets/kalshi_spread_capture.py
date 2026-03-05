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
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from decimal import Decimal

import pandas as pd

from nautilus_trader.adapters.kalshi.fee_model import KalshiProportionalFeeModel
from nautilus_trader.adapters.kalshi.research import discover_markets
from nautilus_trader.adapters.kalshi.research import load_market_bars
from nautilus_trader.adapters.prediction_market.research import print_backtest_summary
from nautilus_trader.adapters.prediction_market.research import run_market_backtest
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.examples.strategies.prediction_market.mean_reversion import (
    BarMeanReversionConfig,
)
from nautilus_trader.examples.strategies.prediction_market.mean_reversion import (
    BarMeanReversionStrategy as BarMeanReversion,
)
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import Venue


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

async def run() -> None:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=LOOKBACK_DAYS))
    end = pd.Timestamp(now)

    # Single shared client — conservative rate to avoid 429s.
    http_client = nautilus_pyo3.HttpClient(
        default_quota=nautilus_pyo3.Quota.rate_per_second(10),
    )

    print(f"Discovering top {MAX_MARKETS} active Kalshi markets by 24h volume...")
    candidates = await discover_markets(
        http_client=http_client,
        candidate_limit=CANDIDATE_LIMIT,
        min_volume_24h=0.0,
        yes_price_min=PRICE_MIN,
        yes_price_max=PRICE_MAX,
        min_days_to_expiry=MIN_DAYS_TO_RESOLUTION,
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
        market_data = await load_market_bars(
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
        result = run_market_backtest(
            market_id=ticker,
            instrument=loader.instrument,
            data=bars,
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
            platform="kalshi",
            venue=Venue("KALSHI"),
            base_currency=USD,
            fee_model=KalshiProportionalFeeModel(),
            initial_cash=INITIAL_CASH,
            probability_window=WINDOW,
            price_attr="close",
            count_key="bars",
            market_key="ticker",
        )
        results.append(result)

    print_backtest_summary(
        results=results,
        market_key="ticker",
        count_key="bars",
        count_label="Bars",
        pnl_label="PnL (USD)",
    )
    print(f"\nLegacy charts saved to output/{NAME}_<ticker>_legacy.html")


if __name__ == "__main__":
    asyncio.run(run())
