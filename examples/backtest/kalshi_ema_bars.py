#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Example script demonstrating a two-phase EMA-cross backtest using Kalshi market data.

Phase 1 uses KalshiDataLoader to fetch hourly bars from the Kalshi API and write them
into a local ParquetDataCatalog for efficient replay.

Phase 2 runs an EMA-cross strategy over the catalogued bars using BacktestNode.

Before running, set the constants at the top of the file (MARKET_TICKER, BAR_INTERVAL,
CATALOG_PATH, START, END, FAST_EMA, SLOW_EMA, TRADE_SIZE) to match the market and date
range you want to backtest.

"""

import asyncio
from decimal import Decimal

import pandas as pd

from nautilus_trader.adapters.kalshi.loaders import KalshiDataLoader
from nautilus_trader.backtest.config import BacktestDataConfig  # noqa: F401
from nautilus_trader.backtest.config import BacktestEngineConfig  # noqa: F401
from nautilus_trader.backtest.config import BacktestRunConfig  # noqa: F401
from nautilus_trader.backtest.config import BacktestVenueConfig  # noqa: F401
from nautilus_trader.backtest.node import BacktestNode  # noqa: F401
from nautilus_trader.config import ImportableStrategyConfig  # noqa: F401
from nautilus_trader.config import LoggingConfig  # noqa: F401
from nautilus_trader.model.identifiers import TraderId  # noqa: F401
from nautilus_trader.model.identifiers import Venue  # noqa: F401
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


# ---------------------------------------------------------------------------
# Configure these constants for your backtest
# ---------------------------------------------------------------------------
MARKET_TICKER = "KXBTCD24"        # Kalshi market ticker
BAR_INTERVAL = "Hours1"           # Minutes1 | Hours1 | Days1
CATALOG_PATH = "./kalshi_catalog"  # Local directory for parquet catalog
START = "2024-01-01"               # ISO 8601 UTC date string
END = "2024-12-31"                 # ISO 8601 UTC date string
FAST_EMA = 10
SLOW_EMA = 20
TRADE_SIZE = Decimal("1")          # Number of contracts per trade  # noqa: FURB157


async def fetch_and_catalog() -> None:
    """Phase 1 - fetch bars from Kalshi API and write to local catalog."""
    print(f"Fetching {BAR_INTERVAL} bars for {MARKET_TICKER} from {START} to {END}...")
    loader = await KalshiDataLoader.from_market_ticker(MARKET_TICKER)

    bars = await loader.load_bars(
        start=pd.Timestamp(START, tz="UTC"),
        end=pd.Timestamp(END, tz="UTC"),
        interval=BAR_INTERVAL,
    )

    catalog = ParquetDataCatalog(CATALOG_PATH)
    catalog.write_data([loader.instrument])
    catalog.write_data(bars)

    print(f"Wrote instrument {loader.instrument.id} and {len(bars)} bars to {CATALOG_PATH}")


def run_backtest() -> None:
    """Phase 2 - run EMA-cross backtest against the catalog data."""
    pass  # TODO  # noqa: PIE790


if __name__ == "__main__":
    asyncio.run(fetch_and_catalog())
    run_backtest()
