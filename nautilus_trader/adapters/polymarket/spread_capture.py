# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software distributed under the
#  License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

import pandas as pd

from nautilus_trader.adapters.polymarket.common.constants import POLYMARKET_VENUE
from nautilus_trader.adapters.polymarket.common.gamma_markets import list_markets
from nautilus_trader.adapters.polymarket.common.market_selection import end_date_utc
from nautilus_trader.adapters.polymarket.common.market_selection import volume_24h
from nautilus_trader.adapters.polymarket.common.market_selection import yes_price
from nautilus_trader.adapters.polymarket.fee_model import PolymarketFeeModel
from nautilus_trader.adapters.polymarket.loaders import PolymarketDataLoader
from nautilus_trader.adapters.prediction_market.backtest_utils import build_brier_inputs
from nautilus_trader.adapters.prediction_market.backtest_utils import build_market_prices
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_price_points
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_realized_pnl
from nautilus_trader.analysis.legacy_plot_adapter import create_legacy_backtest_chart
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.currencies import USDC_POS
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Money
from nautilus_trader.risk.config import RiskEngineConfig
from nautilus_trader.trading.strategy import Strategy


async def discover_spread_capture_slugs(
    *,
    candidate_limit: int,
    lookback_days: int,
    price_min: float,
    price_max: float,
    quota_rate_per_second: int = 20,
) -> list[str]:
    """
    Discover active Polymarket slugs suited to spread-capture strategies.
    """
    client = nautilus_pyo3.HttpClient(
        default_quota=nautilus_pyo3.Quota.rate_per_second(quota_rate_per_second),
    )
    markets = await list_markets(
        http_client=client,
        filters={"is_active": True, "limit": 200},
        max_results=200,
    )
    if not markets:
        return []

    now = datetime.now(UTC)
    min_end = now + timedelta(days=lookback_days)

    markets.sort(key=volume_24h, reverse=True)
    slugs: list[str] = []
    skipped = {"price": 0, "end_date": 0, "no_volume": 0}
    for market in markets:
        slug = str(market.get("slug", ""))
        if not slug:
            continue
        if volume_24h(market) <= 0:
            skipped["no_volume"] += 1
            continue
        price = yes_price(market)
        if price is None or not (price_min <= price <= price_max):
            skipped["price"] += 1
            continue
        end = end_date_utc(market)
        if end is not None and end < min_end:
            skipped["end_date"] += 1
            continue
        slugs.append(slug)
        if len(slugs) >= candidate_limit:
            break

    print(
        f"  Selected {len(slugs)} markets "
        f"(skipped: {skipped['price']} by price, "
        f"{skipped['end_date']} resolving soon, "
        f"{skipped['no_volume']} inactive)"
    )
    return slugs


async def load_spread_capture_market(
    *,
    slug: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_trades: int,
    min_price_range: float,
) -> tuple[PolymarketDataLoader, list[TradeTick]] | None:
    """
    Load trade history for one Polymarket slug and validate minimum data quality.
    """
    try:
        loader = await PolymarketDataLoader.from_market_slug(slug)
        trades = await loader.load_trades(start, end)
        if len(trades) < min_trades:
            print(f"  skip {slug}: fewer than {min_trades} trades")
            return None

        prices = [float(tick.price) for tick in trades]
        if prices:
            price_range = max(prices) - min(prices)
            if price_range < min_price_range:
                print(f"  skip {slug}: price range {price_range:.3f} < {min_price_range:.3f}")
                return None

        return loader, trades
    except Exception as exc:
        print(f"  skip {slug}: {exc}")
        return None


def run_spread_capture_backtest(
    *,
    slug: str,
    loader: PolymarketDataLoader,
    trades: list[TradeTick],
    strategy: Strategy,
    strategy_name: str,
    output_prefix: str,
    initial_cash: float,
    probability_window: int,
    open_browser: bool = False,
) -> dict[str, Any]:
    """
    Run one Polymarket spread-capture backtest and generate a legacy chart.
    """
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
        starting_balances=[Money(initial_cash, USDC_POS)],
        fee_model=PolymarketFeeModel(),
    )
    engine.add_instrument(instrument)
    engine.add_data(trades)
    engine.add_strategy(strategy)
    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    pnl = extract_realized_pnl(positions)
    price_points = extract_price_points(trades, price_attr="price")
    user_probabilities, market_probabilities, outcomes = build_brier_inputs(
        points=price_points,
        window=probability_window,
    )

    chart_path = f"output/{output_prefix}_{slug}_legacy.html"
    os.makedirs("output", exist_ok=True)
    create_legacy_backtest_chart(
        engine=engine,
        output_path=chart_path,
        strategy_name=strategy_name,
        platform="polymarket",
        initial_cash=initial_cash,
        market_prices={str(instrument.id): build_market_prices(price_points)},
        user_probabilities=user_probabilities,
        market_probabilities=market_probabilities,
        outcomes=outcomes,
        open_browser=open_browser,
    )

    engine.reset()
    engine.dispose()
    return {
        "slug": slug,
        "trades": len(trades),
        "fills": len(fills),
        "pnl": pnl,
    }


def print_spread_capture_summary(results: list[dict[str, Any]]) -> None:
    """
    Print a market-level summary table for Polymarket spread-capture runs.
    """
    if not results:
        print("No markets had sufficient data.")
        return

    col_w = max(len(str(result["slug"])) for result in results) + 2
    header = f"{'Market':<{col_w}} {'Trades':>8} {'Fills':>6} {'PnL (USDC)':>12}"
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for result in results:
        print(
            f"{result['slug']:<{col_w}} {result['trades']:>8} "
            f"{result['fills']:>6} {result['pnl']:>+12.4f}"
        )
    total_pnl = sum(float(result["pnl"]) for result in results)
    total_fills = sum(int(result["fills"]) for result in results)
    print(sep)
    print(f"{'TOTAL':<{col_w}} {'':>8} {total_fills:>6} {total_pnl:>+12.4f}")
    print(sep)
