"""
Shared defaults for prediction-market backtest scripts.

Backtest entrypoints should import these constants instead of hardcoding
market IDs in each file.
"""

DEFAULT_KALSHI_MARKET_TICKER = "KXNEXTIRANLEADER-45JAN01-MKHA"
DEFAULT_POLYMARKET_MARKET_SLUG = (
    "will-gavin-newsom-win-the-2028-democratic-presidential-nomination-568"
)
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_INITIAL_CASH = 100.0
