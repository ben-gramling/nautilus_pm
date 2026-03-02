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

import decimal
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import msgspec
import pytest

from nautilus_trader.adapters.kalshi.loaders import KalshiDataLoader
from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import BinaryOption
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


def make_instrument() -> BinaryOption:
    """Return a minimal BinaryOption for testing."""
    return BinaryOption(
        instrument_id=InstrumentId(Symbol("KXBTC-25MAR15-B100000"), Venue("KALSHI")),
        raw_symbol=Symbol("KXBTC-25MAR15-B100000"),
        asset_class=AssetClass.ALTERNATIVE,
        currency=Currency.from_str("USD"),
        activation_ns=0,
        expiration_ns=0,
        price_precision=4,
        size_precision=2,
        price_increment=Price.from_str("0.0001"),
        size_increment=Quantity.from_str("0.01"),
        maker_fee=decimal.Decimal(0),
        taker_fee=decimal.Decimal(0),
        outcome="Yes",
        description="Test market",
        ts_event=0,
        ts_init=0,
    )


def test_init_stores_instrument():
    instrument = make_instrument()
    http_client = MagicMock()
    loader = KalshiDataLoader(instrument=instrument, http_client=http_client)
    assert loader.instrument is instrument


def test_init_creates_default_http_client():
    instrument = make_instrument()
    loader = KalshiDataLoader(instrument=instrument)
    assert loader._http_client is not None


def make_market_dict(ticker: str = "KXBTC-25MAR15-B100000") -> dict:
    return {
        "ticker": ticker,
        "title": "BTC above 100k on March 15?",
        "open_time": "2025-01-01T00:00:00Z",
        "close_time": "2025-03-15T00:00:00Z",
        "latest_expiration_time": "2025-03-15T00:00:00Z",
    }


def make_mock_response(body: dict | list, status: int = 200):
    mock = MagicMock()
    mock.status = status
    mock.body = msgspec.json.encode(body)
    return mock


@pytest.mark.asyncio
async def test_from_market_ticker_returns_loader():
    ticker = "KXBTC-25MAR15-B100000"
    market = make_market_dict(ticker)
    mock_client = MagicMock()
    mock_client.get = AsyncMock(
        return_value=make_mock_response({"market": market})
    )

    loader = await KalshiDataLoader.from_market_ticker(ticker, http_client=mock_client)

    assert isinstance(loader, KalshiDataLoader)
    assert loader.instrument.id.symbol.value == ticker
    mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_from_market_ticker_raises_on_404():
    mock_client = MagicMock()
    mock_client.get = AsyncMock(
        return_value=make_mock_response({}, status=404)
    )

    with pytest.raises(ValueError, match="not found"):
        await KalshiDataLoader.from_market_ticker("NONEXISTENT", http_client=mock_client)
