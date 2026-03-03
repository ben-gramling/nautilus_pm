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
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# sys.path: wheel takes priority for compiled extensions, but we extend
# nautilus_trader.adapters.__path__ so the local kalshi adapter is findable.
# ---------------------------------------------------------------------------
_NAUTILUS_SRC = Path(__file__).resolve().parents[2]  # nautilus_pm/
_VENV_SITE = _NAUTILUS_SRC / ".venv/lib/python3.13/site-packages"
if str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))
if str(_NAUTILUS_SRC) in sys.path:
    sys.path.remove(str(_NAUTILUS_SRC))

import nautilus_trader.adapters as _nt_adapters  # noqa: E402


_LOCAL_ADAPTERS = _NAUTILUS_SRC / "nautilus_trader" / "adapters"
if str(_LOCAL_ADAPTERS) not in _nt_adapters.__path__:
    _nt_adapters.__path__.append(str(_LOCAL_ADAPTERS))

from nautilus_trader.adapters.kalshi.loaders import KalshiDataLoader  # noqa: E402
from nautilus_trader.analysis.config import TearsheetConfig  # noqa: E402
from nautilus_trader.analysis.tearsheet import create_tearsheet  # noqa: E402
from nautilus_trader.backtest.config import BacktestDataConfig  # noqa: E402
from nautilus_trader.backtest.config import BacktestEngineConfig  # noqa: E402
from nautilus_trader.backtest.config import BacktestRunConfig  # noqa: E402
from nautilus_trader.backtest.config import BacktestVenueConfig  # noqa: E402
from nautilus_trader.backtest.node import BacktestNode  # noqa: E402
from nautilus_trader.config import ImportableStrategyConfig  # noqa: E402
from nautilus_trader.config import LoggingConfig  # noqa: E402
from nautilus_trader.model.data import Bar  # noqa: E402
from nautilus_trader.model.identifiers import TraderId  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog  # noqa: E402


# ---------------------------------------------------------------------------
# Configure these constants for your backtest
# ---------------------------------------------------------------------------
MARKET_TICKER = "KXFEDCHAIRNOM-29-KW"        # Kalshi market ticker
BAR_INTERVAL = "Hours1"           # Minutes1 | Hours1 | Days1
CATALOG_PATH = "./kalshi_catalog"  # Local directory for parquet catalog
START = "2026-01-01"               # ISO 8601 UTC date string
END = "2026-03-01"                 # ISO 8601 UTC exclusive end date
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
    instrument_id = f"{MARKET_TICKER}.KALSHI"
    # NOTE: bar_type must match the interval written by fetch_and_catalog().
    # If BAR_INTERVAL changes, update "1-HOUR-LAST" here accordingly.
    bar_type = f"{instrument_id}-1-HOUR-LAST-EXTERNAL"

    venue_config = BacktestVenueConfig(
        name="KALSHI",
        oms_type="NETTING",
        account_type="CASH",
        base_currency="USD",
        starting_balances=["10000 USD"],
    )

    data_config = BacktestDataConfig(
        catalog_path=CATALOG_PATH,
        data_cls=Bar,
        instrument_id=instrument_id,
        bar_spec="1-HOUR-LAST",
        start_time=START,
        end_time=END,
    )

    strategy_config = ImportableStrategyConfig(
        strategy_path="nautilus_trader.examples.strategies.ema_cross_long_only:EMACrossLongOnly",
        config_path="nautilus_trader.examples.strategies.ema_cross_long_only:EMACrossLongOnlyConfig",
        config={
            "instrument_id": instrument_id,
            "bar_type": bar_type,
            "fast_ema_period": FAST_EMA,
            "slow_ema_period": SLOW_EMA,
            "trade_size": str(TRADE_SIZE),
        },
    )

    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO"),
        strategies=[strategy_config],
    )

    run_config = BacktestRunConfig(
        venues=[venue_config],
        data=[data_config],
        engine=engine_config,
        dispose_on_completion=False,
    )

    node = BacktestNode(configs=[run_config])
    node.run()

    engine = node.get_engine(run_config.id)
    kalshi_venue = Venue("KALSHI")

    with pd.option_context(
        "display.max_rows",
        100,
        "display.max_columns",
        None,
        "display.width",
        300,
    ):
        print(engine.trader.generate_account_report(kalshi_venue))
        print(engine.trader.generate_order_fills_report())
        print(engine.trader.generate_positions_report())

    tearsheet_path = "./kalshi_ema_tearsheet.html"
    create_tearsheet(engine, tearsheet_path, config=TearsheetConfig(theme="nautilus_dark"))
    print(f"Tearsheet saved to {tearsheet_path}")

    node.dispose()


if __name__ == "__main__":
    asyncio.run(fetch_and_catalog())
    run_backtest()
