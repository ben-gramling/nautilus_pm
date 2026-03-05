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


def volume_24h(market: Mapping[str, Any]) -> float:
    """
    Extract 24-hour volume from a Kalshi market payload.
    """
    for key in ("volume_24h", "volume_fp", "volume"):
        raw = market.get(key)
        if raw is None:
            continue
        try:
            volume = float(raw)
        except (TypeError, ValueError):
            continue
        if volume > 0:
            return volume
    return 0.0


def yes_price(market: Mapping[str, Any]) -> float | None:
    """
    Extract and normalize the current YES price for a Kalshi market.
    """
    for key in ("last_price_dollars", "yes_bid_dollars", "yes_price_dollars", "yes_price"):
        raw = market.get(key)
        if raw is None:
            continue
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if price >= 1.0:
            price /= 100.0  # legacy integer-cents fields
        if 0.0 < price < 1.0:
            return price
    return None


def end_date_utc(market: Mapping[str, Any]) -> datetime | None:
    """
    Parse market expiry from a Kalshi market payload.
    """
    raw = market.get("close_time") or market.get("latest_expiration_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
