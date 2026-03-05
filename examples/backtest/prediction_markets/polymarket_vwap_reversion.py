"""
VWAP-reversion strategy across Polymarket markets.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal

from _polymarket_trade_runner import run_multi_market_trade_backtest

from nautilus_trader.examples.strategies.prediction_market import TradeTickVWAPReversionConfig
from nautilus_trader.examples.strategies.prediction_market import TradeTickVWAPReversionStrategy


NAME = "polymarket_vwap_reversion"
DESCRIPTION = "VWAP dislocation mean-reversion across Polymarket markets"

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "30"))
MAX_MARKETS = int(os.getenv("MAX_MARKETS", "10"))
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "140"))
MIN_TRADES = int(os.getenv("MIN_TRADES", "300"))
MIN_PRICE_RANGE = float(os.getenv("MIN_PRICE_RANGE", "0.12"))
PRICE_MIN = float(os.getenv("PRICE_MIN", "0.20"))
PRICE_MAX = float(os.getenv("PRICE_MAX", "0.80"))

VWAP_WINDOW = int(os.getenv("VWAP_WINDOW", "120"))
ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.006"))
EXIT_THRESHOLD = float(os.getenv("EXIT_THRESHOLD", "0.0015"))
MIN_TICK_SIZE = float(os.getenv("MIN_TICK_SIZE", "1.0"))
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.015"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.020"))

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
        probability_window=VWAP_WINDOW,
        strategy_factory=lambda instrument_id: TradeTickVWAPReversionStrategy(
            config=TradeTickVWAPReversionConfig(
                instrument_id=instrument_id,
                trade_size=TRADE_SIZE,
                vwap_window=VWAP_WINDOW,
                entry_threshold=ENTRY_THRESHOLD,
                exit_threshold=EXIT_THRESHOLD,
                min_tick_size=MIN_TICK_SIZE,
                take_profit=TAKE_PROFIT,
                stop_loss=STOP_LOSS,
            ),
        ),
    )


if __name__ == "__main__":
    asyncio.run(run())
