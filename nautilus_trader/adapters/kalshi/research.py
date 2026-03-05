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
from collections.abc import Callable
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

import msgspec
import pandas as pd

from nautilus_trader.adapters.kalshi.loaders import KalshiDataLoader
from nautilus_trader.adapters.kalshi.market_selection import end_date_utc
from nautilus_trader.adapters.kalshi.market_selection import volume_24h
from nautilus_trader.adapters.kalshi.market_selection import yes_price
from nautilus_trader.adapters.kalshi.providers import KALSHI_REST_BASE
from nautilus_trader.adapters.kalshi.providers import market_dict_to_instrument
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.data import Bar


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


def _extend_with_event_markets(
    all_markets: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    exclude_ticker_prefixes: tuple[str, ...],
) -> None:
    for event in events:
        series_ticker = event.get("series_ticker", "")
        nested = event.get("markets") or []
        for market in nested:
            ticker = market.get("ticker", "")
            if ticker.startswith(exclude_ticker_prefixes):
                continue
            market["series_ticker"] = series_ticker
            all_markets.append(market)


async def discover_markets(
    *,
    http_client: nautilus_pyo3.HttpClient,
    candidate_limit: int,
    status: str = "open",
    page_limit: int = 200,
    include_nested_markets: bool = True,
    exclude_ticker_prefixes: tuple[str, ...] = ("KXMVE",),
    min_volume_24h: float = 0.0,
    yes_price_min: float | None = None,
    yes_price_max: float | None = None,
    min_days_to_expiry: int | None = None,
    predicate: MarketPredicate | None = None,
    sort_key: MarketSortKey = volume_24h,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """
    Discover Kalshi markets from the events endpoint with optional filtering.
    """
    all_markets: list[dict[str, Any]] = []
    cursor: str | None = None

    while len(all_markets) < candidate_limit:
        params: dict[str, str] = {
            "status": status,
            "limit": str(page_limit),
            "with_nested_markets": "true" if include_nested_markets else "false",
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

        _extend_with_event_markets(
            all_markets,
            events,
            exclude_ticker_prefixes=exclude_ticker_prefixes,
        )

        cursor = data.get("cursor")
        if not cursor:
            break

    min_expiry_dt = None
    if min_days_to_expiry is not None:
        min_expiry_dt = datetime.now(UTC) + timedelta(days=min_days_to_expiry)

    filtered = [
        market
        for market in all_markets
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


async def load_market_bars(
    *,
    market: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    http_client: nautilus_pyo3.HttpClient,
    interval: str = "Minutes1",
    chunk_minutes: int = 5_000,
    min_bars: int = 0,
    min_price_range: float = 0.0,
    max_retries: int = 4,
    retry_base_delay: float = 2.0,
) -> tuple[KalshiDataLoader, list[Bar]] | None:
    """
    Load and validate bar history for a Kalshi market.
    """
    ticker = str(market.get("ticker", "UNKNOWN"))
    try:
        instrument = market_dict_to_instrument(dict(market))
        series_ticker = str(market.get("series_ticker", ""))
        loader = KalshiDataLoader(
            instrument=instrument,
            series_ticker=series_ticker,
            http_client=http_client,
        )

        bars: list[Bar] = []
        chunk_delta = pd.Timedelta(minutes=chunk_minutes)
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_delta, end)
            for attempt in range(max_retries + 1):
                try:
                    bars.extend(
                        await loader.load_bars(
                            start=chunk_start,
                            end=chunk_end,
                            interval=interval,
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
