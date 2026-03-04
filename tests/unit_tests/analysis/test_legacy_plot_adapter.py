# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import pandas as pd
import pytest

from nautilus_trader.analysis.legacy_plot_adapter import prepare_cumulative_brier_advantage


def test_prepare_cumulative_brier_advantage_calculates_expected_values() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    user_probabilities = pd.Series([0.8, 0.4, 0.3], index=index)
    market_probabilities = pd.Series([0.6, 0.5, 0.7], index=index)
    outcomes = pd.Series([1.0, 0.0, 0.0], index=index)

    result = prepare_cumulative_brier_advantage(
        user_probabilities=user_probabilities,
        market_probabilities=market_probabilities,
        outcomes=outcomes,
    )

    assert not result.empty
    assert result["brier_advantage"].iloc[0] == pytest.approx(0.12)
    assert result["brier_advantage"].iloc[1] == pytest.approx(0.09)
    assert result["brier_advantage"].iloc[2] == pytest.approx(0.40)
    assert result["cumulative_brier_advantage"].iloc[-1] == pytest.approx(0.61)


def test_prepare_cumulative_brier_advantage_returns_empty_when_inputs_missing() -> None:
    result = prepare_cumulative_brier_advantage(
        user_probabilities=None,
        market_probabilities=pd.Series([0.5]),
        outcomes=pd.Series([1.0]),
    )

    assert result.empty
