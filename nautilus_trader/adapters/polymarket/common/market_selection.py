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

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import msgspec


def volume_24h(market: Mapping[str, Any]) -> float:
    """
    Extract 24-hour volume from a Polymarket Gamma market payload.
    """
    try:
        return float(market.get("volume24hr", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def yes_price(market: Mapping[str, Any]) -> float | None:
    """
    Extract YES probability from ``outcomePrices``.
    """
    raw = market.get("outcomePrices")
    if not raw:
        return None

    try:
        prices = msgspec.json.decode(raw) if isinstance(raw, str | bytes) else raw
        if not isinstance(prices, list) or not prices:
            return None
        return float(prices[0])
    except Exception:
        return None


def end_date_utc(market: Mapping[str, Any]) -> datetime | None:
    """
    Parse market end date from Gamma payload fields.
    """
    raw = market.get("endDate") or market.get("end_date_iso")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
