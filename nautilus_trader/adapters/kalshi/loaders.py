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
"""
Provides a data loader for historical Kalshi prediction market data.
"""

from __future__ import annotations

from typing import Any

import msgspec

from nautilus_trader.adapters.kalshi.providers import KALSHI_REST_BASE
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.core.datetime import secs_to_nanos
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.instruments import BinaryOption


KALSHI_HTTP_RATE_LIMIT_RPS = 20  # Basic tier


def _market_dict_to_instrument(market: dict[str, Any]) -> BinaryOption:
    """Convert a Kalshi market dict to a NautilusTrader BinaryOption."""
    import decimal
    from datetime import datetime

    from nautilus_trader.core.datetime import dt_to_unix_nanos
    from nautilus_trader.model.enums import AssetClass
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.identifiers import Symbol
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Currency
    from nautilus_trader.model.objects import Price
    from nautilus_trader.model.objects import Quantity

    ticker = market["ticker"]

    def parse_ts(s: str | None) -> int:
        if not s:
            return 0
        dt = datetime.fromisoformat(s)
        return dt_to_unix_nanos(dt)

    return BinaryOption(
        instrument_id=InstrumentId(Symbol(ticker), Venue("KALSHI")),
        raw_symbol=Symbol(ticker),
        asset_class=AssetClass.ALTERNATIVE,
        currency=Currency.from_str("USD"),
        activation_ns=parse_ts(market.get("open_time")),
        expiration_ns=parse_ts(
            market.get("close_time") or market.get("latest_expiration_time")
        ),
        price_precision=4,
        size_precision=2,
        price_increment=Price.from_str("0.0001"),
        size_increment=Quantity.from_str("0.01"),
        maker_fee=decimal.Decimal(0),
        taker_fee=decimal.Decimal(0),
        outcome="Yes",
        description=market.get("title"),
        ts_event=0,
        ts_init=0,
    )


class KalshiDataLoader:
    """
    Provides a data loader for historical Kalshi market data.

    This loader fetches data from the public Kalshi REST API:
    - ``GET /markets/{ticker}`` — instrument discovery
    - ``GET /historical/markets/{ticker}/trades`` — historical trades (cursor-paginated)
    - ``GET /historical/markets/{ticker}/candlesticks`` — OHLCV bars

    Historical endpoints are public and require no authentication.

    If no ``http_client`` is provided, the loader creates one with a default
    rate limit of 20 requests per second (Kalshi Basic tier).

    Parameters
    ----------
    instrument : BinaryOption
        The binary option instrument to load data for.
    http_client : nautilus_pyo3.HttpClient, optional
        HTTP client to use for requests. If not provided, a new client is created.
    """

    def __init__(
        self,
        instrument: BinaryOption,
        http_client: nautilus_pyo3.HttpClient | None = None,
    ) -> None:
        self._instrument = instrument
        self._http_client = http_client or self._create_http_client()

    @staticmethod
    def _create_http_client() -> nautilus_pyo3.HttpClient:
        return nautilus_pyo3.HttpClient(
            default_quota=nautilus_pyo3.Quota.rate_per_second(KALSHI_HTTP_RATE_LIMIT_RPS),
        )

    @property
    def instrument(self) -> BinaryOption:
        """Return the instrument for this loader."""
        return self._instrument

    @classmethod
    async def from_market_ticker(
        cls,
        ticker: str,
        http_client: nautilus_pyo3.HttpClient | None = None,
    ) -> KalshiDataLoader:
        """
        Create a loader by fetching market data for the given ticker.

        Parameters
        ----------
        ticker : str
            The Kalshi market ticker, e.g. ``"KXBTC-25MAR15-B100000"``.
        http_client : nautilus_pyo3.HttpClient, optional
            HTTP client to use. If not provided, a new client is created.

        Returns
        -------
        KalshiDataLoader

        Raises
        ------
        ValueError
            If the market ticker is not found.
        RuntimeError
            If the HTTP request fails.
        """
        client = http_client or cls._create_http_client()
        response = await client.get(url=f"{KALSHI_REST_BASE}/markets/{ticker}")

        if response.status == 404:
            raise ValueError(f"Market ticker '{ticker}' not found")
        if response.status != 200:
            raise RuntimeError(
                f"HTTP request failed with status {response.status}: "
                f"{response.body.decode('utf-8')}",
            )

        data = msgspec.json.decode(response.body)
        market = data["market"]
        instrument = _market_dict_to_instrument(market)

        return cls(instrument=instrument, http_client=client)

    async def fetch_trades(
        self,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Fetch historical trades from the Kalshi API.

        Automatically paginates using cursor-based pagination until all
        trades are retrieved.

        Parameters
        ----------
        min_ts : int, optional
            Minimum Unix timestamp in seconds (inclusive).
        max_ts : int, optional
            Maximum Unix timestamp in seconds (inclusive).
        limit : int, default 1000
            Number of trades per page (Kalshi maximum is 1000).

        Returns
        -------
        list[dict[str, Any]]
            Raw trade dicts as returned by the Kalshi API.
        """
        ticker = self._instrument.id.symbol.value
        all_trades: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {"limit": str(limit)}
            if min_ts is not None:
                params["min_ts"] = str(min_ts)
            if max_ts is not None:
                params["max_ts"] = str(max_ts)
            if cursor:
                params["cursor"] = cursor

            response = await self._http_client.get(
                url=f"{KALSHI_REST_BASE}/historical/markets/{ticker}/trades",
                params=params,
            )

            if response.status != 200:
                raise RuntimeError(
                    f"HTTP request failed with status {response.status}: "
                    f"{response.body.decode('utf-8')}",
                )

            data = msgspec.json.decode(response.body)
            page_trades = data.get("trades", [])
            all_trades.extend(page_trades)

            cursor = data.get("cursor") or None
            if not cursor or not page_trades:
                break

        return all_trades

    def parse_trades(
        self,
        trades_data: list[dict[str, Any]],
    ) -> list[TradeTick]:
        """
        Parse raw Kalshi trade dicts into TradeTick objects.

        Parameters
        ----------
        trades_data : list[dict[str, Any]]
            Raw trade dicts from the Kalshi historical trades API.

        Returns
        -------
        list[TradeTick]
        """
        ticker = self._instrument.id.symbol.value
        instrument_id = self._instrument.id
        make_price = self._instrument.make_price
        make_qty = self._instrument.make_qty
        trades: list[TradeTick] = []

        for trade in trades_data:
            ts_event = secs_to_nanos(trade["ts"])
            taker_side = trade.get("taker_side", "")
            if taker_side == "yes":
                aggressor_side = AggressorSide.BUYER
            elif taker_side == "no":
                aggressor_side = AggressorSide.SELLER
            else:
                aggressor_side = AggressorSide.NO_AGGRESSOR

            raw_id = f"{ticker}_{trade['ts']}_{trade['yes_price']}_{trade['count']}"
            trade_id = TradeId(raw_id[:36])

            trades.append(
                TradeTick(
                    instrument_id=instrument_id,
                    price=make_price(trade["yes_price"]),
                    size=make_qty(trade["count"]),
                    aggressor_side=aggressor_side,
                    trade_id=trade_id,
                    ts_event=ts_event,
                    ts_init=ts_event,
                )
            )

        return trades
