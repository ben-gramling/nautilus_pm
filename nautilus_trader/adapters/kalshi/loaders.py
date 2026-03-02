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

from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.instruments import BinaryOption


KALSHI_HTTP_RATE_LIMIT_RPS = 20  # Basic tier


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
