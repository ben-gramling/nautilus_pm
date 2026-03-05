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

from collections.abc import Callable
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

import pandas as pd

from nautilus_trader.adapters.polymarket.common.gamma_markets import list_markets
from nautilus_trader.adapters.polymarket.common.market_selection import end_date_utc
from nautilus_trader.adapters.polymarket.common.market_selection import volume_24h
from nautilus_trader.adapters.polymarket.common.market_selection import yes_price
from nautilus_trader.adapters.polymarket.loaders import PolymarketDataLoader
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.data import TradeTick


MarketPredicate = Callable[[Mapping[str, Any]], bool]
MarketSortKey = Callable[[Mapping[str, Any]], float]


def _passes_filters(
    market: Mapping[str, Any],
    *,
    min_volume_24h: float,
    yes_price_min: float | None,
    yes_price_max: float | None,
    min_expiry_dt: datetime | None,
    predicate: MarketPredicate | None,
) -> bool:
    if volume_24h(market) < min_volume_24h:
        return False

    if yes_price_min is not None or yes_price_max is not None:
        market_yes_price = yes_price(market)
        if market_yes_price is None:
            return False
        if yes_price_min is not None and market_yes_price < yes_price_min:
            return False
        if yes_price_max is not None and market_yes_price > yes_price_max:
            return False

    if min_expiry_dt is not None:
        expiry = end_date_utc(market)
        if expiry is not None and expiry < min_expiry_dt:
            return False

    return predicate is None or predicate(market)


async def discover_markets(
    *,
    candidate_limit: int,
    http_client: nautilus_pyo3.HttpClient | None = None,
    api_filters: dict[str, Any] | None = None,
    max_results: int = 200,
    quota_rate_per_second: int = 20,
    min_volume_24h: float = 0.0,
    yes_price_min: float | None = None,
    yes_price_max: float | None = None,
    min_days_to_expiry: int | None = None,
    predicate: MarketPredicate | None = None,
    sort_key: MarketSortKey = volume_24h,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """
    Discover Polymarket markets from Gamma with optional filtering.
    """
    client = http_client
    if client is None:
        client = nautilus_pyo3.HttpClient(
            default_quota=nautilus_pyo3.Quota.rate_per_second(quota_rate_per_second),
        )

    filters = {"is_active": True, "limit": 200}
    if api_filters is not None:
        filters.update(api_filters)

    markets = await list_markets(
        http_client=client,
        filters=filters,
        max_results=max_results,
    )
    if not markets:
        return []

    min_expiry_dt = None
    if min_days_to_expiry is not None:
        min_expiry_dt = datetime.now(UTC) + timedelta(days=min_days_to_expiry)

    filtered = [
        market
        for market in markets
        if _passes_filters(
            market,
            min_volume_24h=min_volume_24h,
            yes_price_min=yes_price_min,
            yes_price_max=yes_price_max,
            min_expiry_dt=min_expiry_dt,
            predicate=predicate,
        )
    ]
    filtered.sort(key=sort_key, reverse=descending)
    return filtered[:candidate_limit]


async def load_market_trades(
    *,
    slug: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_trades: int = 0,
    min_price_range: float = 0.0,
) -> tuple[PolymarketDataLoader, list[TradeTick]] | None:
    """
    Load and validate trade history for a Polymarket market slug.
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
