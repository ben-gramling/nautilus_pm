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
from unittest.mock import MagicMock

import pytest

from nautilus_trader.adapters.kalshi.loaders import KalshiDataLoader
from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import BinaryOption
from nautilus_trader.model.objects import Currency, Price, Quantity


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
        maker_fee=decimal.Decimal("0"),
        taker_fee=decimal.Decimal("0"),
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
