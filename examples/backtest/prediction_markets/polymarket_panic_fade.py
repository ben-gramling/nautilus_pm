"""
Panic-fade strategy across Polymarket markets.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal

from _polymarket_trade_runner import run_multi_market_trade_backtest

from nautilus_trader.examples.strategies.prediction_market import TradeTickPanicFadeConfig
from nautilus_trader.examples.strategies.prediction_market import TradeTickPanicFadeStrategy


NAME = "polymarket_panic_fade"
DESCRIPTION = "Panic selloff fade strategy across Polymarket markets"

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "30"))
MAX_MARKETS = int(os.getenv("MAX_MARKETS", "10"))
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "140"))
MIN_TRADES = int(os.getenv("MIN_TRADES", "300"))
MIN_PRICE_RANGE = float(os.getenv("MIN_PRICE_RANGE", "0.12"))
PRICE_MIN = float(os.getenv("PRICE_MIN", "0.15"))
PRICE_MAX = float(os.getenv("PRICE_MAX", "0.80"))

DROP_WINDOW = int(os.getenv("DROP_WINDOW", "120"))
MIN_DROP = float(os.getenv("MIN_DROP", "0.060"))
PANIC_PRICE = float(os.getenv("PANIC_PRICE", "0.300"))
REBOUND_EXIT = float(os.getenv("REBOUND_EXIT", "0.420"))
MAX_HOLDING_PERIODS = int(os.getenv("MAX_HOLDING_PERIODS", "500"))
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.040"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.030"))

TRADE_SIZE = Decimal(os.getenv("TRADE_SIZE", "20"))
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "1000"))


async def run() -> None:
    await run_multi_market_trade_backtest(
        name=NAME,
        lookback_days=LOOKBACK_DAYS,
        candidate_limit=CANDIDATE_LIMIT,
        max_markets=MAX_MARKETS,
        min_trades=MIN_TRADES,
        min_price_range=MIN_PRICE_RANGE,
        yes_price_min=PRICE_MIN,
        yes_price_max=PRICE_MAX,
        min_days_to_expiry=LOOKBACK_DAYS,
        initial_cash=INITIAL_CASH,
        probability_window=DROP_WINDOW,
        strategy_factory=lambda instrument_id: TradeTickPanicFadeStrategy(
            config=TradeTickPanicFadeConfig(
                instrument_id=instrument_id,
                trade_size=TRADE_SIZE,
                drop_window=DROP_WINDOW,
                min_drop=MIN_DROP,
                panic_price=PANIC_PRICE,
                rebound_exit=REBOUND_EXIT,
                max_holding_periods=MAX_HOLDING_PERIODS,
                take_profit=TAKE_PROFIT,
                stop_loss=STOP_LOSS,
            ),
        ),
    )


if __name__ == "__main__":
    asyncio.run(run())
