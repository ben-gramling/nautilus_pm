#!/usr/bin/env python3
"""
Kalshi spread-capture backtest — single market, trade-tick data from REST API.

Strategy: place a limit buy at (fair_value - SPREAD_CENTS) and a limit sell
at (fair_value + SPREAD_CENTS) simultaneously.  Fair value is an EMA of
recent trade prices.  Captures the bid-ask spread on round trips while
keeping inventory capped at +/- MAX_POSITION contracts.

Pipeline
--------
1. Fetch trade ticks for one Kalshi market via the public REST API
2. Run KalshiSpreadCapture strategy via BacktestEngine
3. Print account / fills / positions reports
4. Save NautilusTrader HTML tearsheet

Run
---
    cd /Users/evankolberg/nautilus_pm
    python examples/backtest/kalshi_spread_capture.py

Tune the constants below to explore different markets and parameters.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# sys.path: wheel takes priority for compiled extensions, but we extend
# nautilus_trader.adapters.__path__ so the local kalshi adapter is findable.
# ---------------------------------------------------------------------------
_NAUTILUS_SRC = Path(__file__).resolve().parents[2]  # nautilus_pm/
_VENV_SITE = _NAUTILUS_SRC / ".venv/lib/python3.13/site-packages"
if str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))
if str(_NAUTILUS_SRC) in sys.path:
    sys.path.remove(str(_NAUTILUS_SRC))

import nautilus_trader.adapters as _nt_adapters  # noqa: E402


_LOCAL_ADAPTERS = _NAUTILUS_SRC / "nautilus_trader" / "adapters"
if str(_LOCAL_ADAPTERS) not in _nt_adapters.__path__:
    _nt_adapters.__path__.append(str(_LOCAL_ADAPTERS))

# Strategy lives next to this script
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from kalshi_spread_strategy import KalshiSpreadCapture  # noqa: E402
from kalshi_spread_strategy import KalshiSpreadCaptureConfig  # noqa: E402

from nautilus_trader.adapters.kalshi.loaders import KalshiDataLoader  # noqa: E402
from nautilus_trader.analysis.config import TearsheetConfig  # noqa: E402
from nautilus_trader.analysis.tearsheet import create_tearsheet  # noqa: E402
from nautilus_trader.backtest.config import BacktestEngineConfig  # noqa: E402
from nautilus_trader.backtest.engine import BacktestEngine  # noqa: E402
from nautilus_trader.config import LoggingConfig  # noqa: E402
from nautilus_trader.model.data import TradeTick  # noqa: E402
from nautilus_trader.model.enums import AccountType  # noqa: E402
from nautilus_trader.model.enums import AggressorSide  # noqa: E402
from nautilus_trader.model.enums import OmsType  # noqa: E402
from nautilus_trader.model.identifiers import TradeId  # noqa: E402
from nautilus_trader.model.identifiers import TraderId  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.objects import Currency  # noqa: E402
from nautilus_trader.model.objects import Money  # noqa: E402


# ---------------------------------------------------------------------------
# Configure these constants
# ---------------------------------------------------------------------------
MARKET_TICKER    = "KXFEDCHAIRNOM-29-KW"   # Kalshi market ticker
START            = "2026-01-01"             # ISO 8601 UTC
END              = "2026-03-01"             # ISO 8601 UTC (exclusive)
SPREAD_CENTS     = 2.0    # half-spread: bid at fv-2, ask at fv+2
MAX_POSITION     = 5      # max contracts long or short
TRADE_SIZE       = 1.0    # contracts per order
EMA_ALPHA        = 0.1    # fair-value EMA smoothing (0-1, lower = smoother)
MIN_TICKS        = 5      # warm-up ticks before quoting starts
REQUOTE_THRESHOLD = 0.5   # min fair-value shift to trigger a requote
STARTING_BALANCE = 10_000.0

KALSHI_VENUE = Venue("KALSHI")
USD_CURRENCY = Currency.from_str("USD")
TEARSHEET_PATH = "./kalshi_spread_tearsheet.html"


async def fetch_ticks() -> tuple:
    """
    Fetch bars from the Kalshi REST API and synthesize TradeTick objects.

    The historical trades endpoint is not yet live on the Kalshi API, so we
    use OHLCV candlesticks instead.  Each bar generates four ticks in OHLC
    order (open → high → low → close), giving the backtest engine realistic
    intrabar price swings for limit-order fills.
    """
    print(f"Fetching hourly bars for {MARKET_TICKER} from {START} to {END}...")
    loader = await KalshiDataLoader.from_market_ticker(MARKET_TICKER)
    bars = await loader.load_bars(
        start=pd.Timestamp(START, tz="UTC"),
        end=pd.Timestamp(END, tz="UTC"),
        interval="Hours1",
    )
    print(f"  Loaded {len(bars):,} bars → synthesizing OHLC ticks...")

    instrument = loader.instrument
    ticks: list[TradeTick] = []
    bar_ns = 3_600_000_000_000  # 1 hour in nanoseconds

    for i, bar in enumerate(bars):
        # Space the 4 ticks evenly inside the bar period
        base_ts = bar.ts_event - bar_ns
        offsets = [bar_ns // 4, bar_ns // 2, 3 * bar_ns // 4, bar_ns]
        ohlc = [float(bar.open), float(bar.high), float(bar.low), float(bar.close)]

        for j, (offset, price) in enumerate(zip(offsets, ohlc, strict=True)):
            ts = base_ts + offset
            ticks.append(TradeTick(
                instrument_id=instrument.id,
                price=instrument.make_price(price),
                size=instrument.make_qty(1.0),
                aggressor_side=AggressorSide.NO_AGGRESSOR,
                trade_id=TradeId(f"SYN-{i:05d}-{j}"),
                ts_event=ts,
                ts_init=ts,
            ))

    print(f"  Synthesized {len(ticks):,} trade ticks from {len(bars)} bars")
    return instrument, ticks


def run_backtest(instrument, ticks: list) -> BacktestEngine:
    """Run the spread-capture strategy over the provided trade ticks."""
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="ERROR"),
        )
    )

    engine.add_venue(
        venue=KALSHI_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD_CURRENCY,
        starting_balances=[Money(STARTING_BALANCE, USD_CURRENCY)],
    )

    engine.add_instrument(instrument)
    engine.add_data(ticks)

    engine.add_strategy(
        KalshiSpreadCapture(
            config=KalshiSpreadCaptureConfig(
                instrument_id=instrument.id,
                spread_cents=SPREAD_CENTS,
                max_position=MAX_POSITION,
                trade_size=TRADE_SIZE,
                ema_alpha=EMA_ALPHA,
                min_ticks=MIN_TICKS,
                requote_threshold=REQUOTE_THRESHOLD,
            )
        )
    )

    print("Running backtest...")
    engine.run()
    return engine


def report(engine: BacktestEngine) -> None:
    """Print reports and save tearsheet."""
    with pd.option_context("display.max_rows", 50, "display.max_columns", None, "display.width", 300):
        print("\n--- Account ---")
        print(engine.trader.generate_account_report(KALSHI_VENUE))
        print("\n--- Order Fills ---")
        print(engine.trader.generate_order_fills_report())
        print("\n--- Positions ---")
        print(engine.trader.generate_positions_report())

    create_tearsheet(engine, TEARSHEET_PATH, config=TearsheetConfig(theme="nautilus_dark"))
    print(f"\nTearsheet saved to {TEARSHEET_PATH}")


async def main() -> None:
    instrument, ticks = await fetch_ticks()
    if not ticks:
        print("No ticks found — check MARKET_TICKER and date range.")
        return
    engine = run_backtest(instrument, ticks)
    report(engine)
    engine.reset()
    engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
