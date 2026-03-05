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

import asyncio
import os
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

import msgspec
import pandas as pd

from nautilus_trader.adapters.kalshi.fee_model import KalshiProportionalFeeModel
from nautilus_trader.adapters.kalshi.loaders import KalshiDataLoader
from nautilus_trader.adapters.kalshi.market_selection import end_date_utc
from nautilus_trader.adapters.kalshi.market_selection import volume_24h
from nautilus_trader.adapters.kalshi.market_selection import yes_price
from nautilus_trader.adapters.kalshi.providers import KALSHI_REST_BASE
from nautilus_trader.adapters.kalshi.providers import market_dict_to_instrument
from nautilus_trader.adapters.prediction_market.backtest_utils import build_brier_inputs
from nautilus_trader.adapters.prediction_market.backtest_utils import build_market_prices
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_price_points
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_realized_pnl
from nautilus_trader.analysis.legacy_plot_adapter import create_legacy_backtest_chart
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.risk.config import RiskEngineConfig
from nautilus_trader.trading.strategy import Strategy


def _extend_event_markets(
    all_markets: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    for event in events:
        series_ticker = event.get("series_ticker", "")
        nested = event.get("markets") or []
        for market in nested:
            if market.get("ticker", "").startswith("KXMVE"):
                continue
            market["series_ticker"] = series_ticker
            all_markets.append(market)


def _filter_discovered_markets(
    markets: list[dict[str, Any]],
    *,
    candidate_limit: int,
    price_min: float,
    price_max: float,
    min_days_to_resolution: int,
) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    min_end = now + timedelta(days=min_days_to_resolution)

    skipped = {"price": 0, "end_date": 0, "no_volume": 0}
    filtered: list[dict[str, Any]] = []
    for market in markets:
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
        filtered.append(market)

    filtered.sort(key=volume_24h, reverse=True)
    print(
        f"  Selected {min(len(filtered), candidate_limit)} markets "
        f"(skipped: {skipped['price']} by price, "
        f"{skipped['end_date']} resolving soon, "
        f"{skipped['no_volume']} inactive)"
    )
    return filtered[:candidate_limit]


async def discover_spread_capture_markets(
    *,
    candidate_limit: int,
    http_client: nautilus_pyo3.HttpClient,
    price_min: float,
    price_max: float,
    min_days_to_resolution: int,
) -> list[dict[str, Any]]:
    """
    Discover active Kalshi markets suited to spread-capture strategies.
    """
    all_markets: list[dict[str, Any]] = []
    cursor: str | None = None

    while len(all_markets) < candidate_limit:
        params: dict[str, str] = {
            "status": "open",
            "limit": "200",
            "with_nested_markets": "true",
        }
        if cursor:
            params["cursor"] = cursor

        resp = await http_client.get(
            url=f"{KALSHI_REST_BASE}/events",
            params=params,
        )
        if resp.status != 200:
            break

        data = msgspec.json.decode(resp.body)
        events = data.get("events", [])
        if not events:
            break

        _extend_event_markets(all_markets, events)

        cursor = data.get("cursor")
        if not cursor:
            break

    return _filter_discovered_markets(
        all_markets,
        candidate_limit=candidate_limit,
        price_min=price_min,
        price_max=price_max,
        min_days_to_resolution=min_days_to_resolution,
    )


async def load_spread_capture_market(
    *,
    market: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    http_client: nautilus_pyo3.HttpClient,
    min_bars: int,
    min_price_range: float,
    max_retries: int,
    retry_base_delay: float,
) -> tuple[KalshiDataLoader, list[Bar]] | None:
    """
    Load bar history for one Kalshi market and validate minimum data quality.
    """
    ticker = str(market.get("ticker", "UNKNOWN"))
    try:
        instrument = market_dict_to_instrument(dict(market))
        series_ticker = str(market["series_ticker"])
        loader = KalshiDataLoader(
            instrument=instrument,
            series_ticker=series_ticker,
            http_client=http_client,
        )

        chunk_delta = pd.Timedelta(minutes=5_000)
        chunk_start = start
        bars: list[Bar] = []
        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_delta, end)
            for attempt in range(max_retries + 1):
                try:
                    bars.extend(
                        await loader.load_bars(
                            start=chunk_start,
                            end=chunk_end,
                            interval="Minutes1",
                        )
                    )
                    break
                except RuntimeError as rt_err:
                    if "429" not in str(rt_err) or attempt >= max_retries:
                        raise
                    delay = retry_base_delay * (2**attempt)
                    print(f"    rate-limited on {ticker}, retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)
            chunk_start = chunk_end

        if len(bars) < min_bars:
            print(f"  skip {ticker}: fewer than {min_bars} bars")
            return None

        closes = [float(bar.close) for bar in bars]
        if closes:
            price_range = max(closes) - min(closes)
            if price_range < min_price_range:
                print(f"  skip {ticker}: price range {price_range:.3f} < {min_price_range:.3f}")
                return None

        return loader, bars
    except Exception as exc:
        print(f"  skip {ticker}: {exc}")
        return None


def run_spread_capture_backtest(
    *,
    ticker: str,
    loader: KalshiDataLoader,
    bars: list[Bar],
    strategy: Strategy,
    strategy_name: str,
    output_prefix: str,
    initial_cash: float,
    probability_window: int,
    open_browser: bool = False,
) -> dict[str, Any]:
    """
    Run one Kalshi spread-capture backtest and generate a legacy chart.
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
        venue=Venue("KALSHI"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(initial_cash, USD)],
        fee_model=KalshiProportionalFeeModel(),
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(strategy)
    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    pnl = extract_realized_pnl(positions)
    price_points = extract_price_points(bars, price_attr="close")
    user_probabilities, market_probabilities, outcomes = build_brier_inputs(
        points=price_points,
        window=probability_window,
    )

    chart_path = f"output/{output_prefix}_{ticker}_legacy.html"
    os.makedirs("output", exist_ok=True)
    create_legacy_backtest_chart(
        engine=engine,
        output_path=chart_path,
        strategy_name=strategy_name,
        platform="kalshi",
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
        "ticker": ticker,
        "bars": len(bars),
        "fills": len(fills),
        "pnl": pnl,
    }


def print_spread_capture_summary(results: list[dict[str, Any]]) -> None:
    """
    Print a market-level summary table for Kalshi spread-capture runs.
    """
    if not results:
        print("No markets had sufficient data.")
        return

    col_w = max(len(str(result["ticker"])) for result in results) + 2
    header = f"{'Market':<{col_w}} {'Bars':>6} {'Fills':>6} {'PnL (USD)':>12}"
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for result in results:
        print(
            f"{result['ticker']:<{col_w}} {result['bars']:>6} "
            f"{result['fills']:>6} {result['pnl']:>+12.4f}"
        )
    total_pnl = sum(float(result["pnl"]) for result in results)
    total_fills = sum(int(result["fills"]) for result in results)
    print(sep)
    print(f"{'TOTAL':<{col_w}} {'':>6} {total_fills:>6} {total_pnl:>+12.4f}")
    print(sep)
