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

import os
from collections.abc import Sequence
from typing import Any

from nautilus_trader.adapters.prediction_market.backtest_utils import build_brier_inputs
from nautilus_trader.adapters.prediction_market.backtest_utils import build_market_prices
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_price_points
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_realized_pnl
from nautilus_trader.analysis.legacy_plot_adapter import create_legacy_backtest_chart
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Money
from nautilus_trader.risk.config import RiskEngineConfig
from nautilus_trader.trading.strategy import Strategy


def run_market_backtest(
    *,
    market_id: str,
    instrument: Any,
    data: Sequence[object],
    strategy: Strategy,
    strategy_name: str,
    output_prefix: str,
    platform: str,
    venue: Venue,
    base_currency: Currency,
    fee_model: Any,
    initial_cash: float,
    probability_window: int,
    price_attr: str,
    count_key: str,
    market_key: str = "market",
    open_browser: bool = False,
) -> dict[str, Any]:
    """
    Run one prediction-market backtest and emit a legacy chart.
    """
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="WARNING"),
            risk_engine=RiskEngineConfig(bypass=True),
        ),
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=base_currency,
        starting_balances=[Money(initial_cash, base_currency)],
        fee_model=fee_model,
    )
    engine.add_instrument(instrument)
    engine.add_data(list(data))
    engine.add_strategy(strategy)
    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    pnl = extract_realized_pnl(positions)
    price_points = extract_price_points(data, price_attr=price_attr)
    user_probabilities, market_probabilities, outcomes = build_brier_inputs(
        points=price_points,
        window=probability_window,
    )

    chart_path = f"output/{output_prefix}_{market_id}_legacy.html"
    os.makedirs("output", exist_ok=True)
    create_legacy_backtest_chart(
        engine=engine,
        output_path=chart_path,
        strategy_name=strategy_name,
        platform=platform,
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
        market_key: market_id,
        count_key: len(data),
        "fills": len(fills),
        "pnl": pnl,
    }


def print_backtest_summary(
    *,
    results: list[dict[str, Any]],
    market_key: str,
    count_key: str,
    count_label: str,
    pnl_label: str,
    empty_message: str = "No markets had sufficient data.",
) -> None:
    """
    Print a normalized backtest summary table.
    """
    if not results:
        print(empty_message)
        return

    col_w = max(len(str(result[market_key])) for result in results) + 2
    header = f"{'Market':<{col_w}} {count_label:>8} {'Fills':>6} {pnl_label:>12}"
    sep = "─" * len(header)

    print(f"\n{sep}\n{header}\n{sep}")
    for result in results:
        print(
            f"{result[market_key]:<{col_w}} {result[count_key]:>8} "
            f"{result['fills']:>6} {result['pnl']:>+12.4f}"
        )

    total_pnl = sum(float(result["pnl"]) for result in results)
    total_fills = sum(int(result["fills"]) for result in results)
    print(sep)
    print(f"{'TOTAL':<{col_w}} {'':>8} {total_fills:>6} {total_pnl:>+12.4f}")
    print(sep)

