#!/usr/bin/env python3
"""
Mean-reversion spread capture on synthesized Kalshi trade ticks.

Fetches hourly bars for each market in MARKET_TICKERS, synthesizes OHLC
trade ticks, and runs a rolling-average mean-reversion backtest across all
markets.  A tearsheet is saved to output/ for each market.

Ported from the Polymarket spread-capture strategy, adapted for Kalshi OHLCV
data (historical trades endpoint is not yet live on the Kalshi API).

Run
---
    cd /Users/evankolberg/nautilus_pm
    python examples/backtest/kalshi_spread_capture.py
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections import deque
from decimal import Decimal
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

from nautilus_trader.adapters.kalshi.fee_model import KalshiProportionalFeeModel  # noqa: E402
from nautilus_trader.adapters.kalshi.loaders import KalshiDataLoader  # noqa: E402
from nautilus_trader.adapters.kalshi.providers import KALSHI_REST_BASE  # noqa: E402
from nautilus_trader.adapters.kalshi.providers import _KalshiHttpClient  # noqa: E402
from nautilus_trader.analysis.config import TearsheetConfig  # noqa: E402
from nautilus_trader.analysis.tearsheet import create_tearsheet  # noqa: E402
from nautilus_trader.backtest.config import BacktestEngineConfig  # noqa: E402
from nautilus_trader.backtest.engine import BacktestEngine  # noqa: E402
from nautilus_trader.config import LoggingConfig  # noqa: E402
from nautilus_trader.core import nautilus_pyo3  # noqa: E402
from nautilus_trader.model.data import TradeTick  # noqa: E402
from nautilus_trader.model.enums import AccountType  # noqa: E402
from nautilus_trader.model.enums import AggressorSide  # noqa: E402
from nautilus_trader.model.enums import OmsType  # noqa: E402
from nautilus_trader.model.enums import OrderSide  # noqa: E402
from nautilus_trader.model.enums import TimeInForce  # noqa: E402
from nautilus_trader.model.identifiers import InstrumentId  # noqa: E402
from nautilus_trader.model.identifiers import TradeId  # noqa: E402
from nautilus_trader.model.identifiers import TraderId  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.objects import Currency  # noqa: E402
from nautilus_trader.model.objects import Money  # noqa: E402
from nautilus_trader.trading.strategy import Strategy  # noqa: E402
from nautilus_trader.trading.strategy import StrategyConfig  # noqa: E402


# ── Strategy metadata (shown in the menu) ────────────────────────────────────
NAME = "Kalshi Spread Capture"
DESCRIPTION = "Mean-reversion spread capture across Kalshi markets"

# ── Configure here ────────────────────────────────────────────────────────────
MAX_MARKETS     = 15            # top N markets by volume to backtest
MIN_TICKS       = 50            # skip markets with fewer synthesized ticks
START           = "2026-01-01"  # ISO 8601 UTC
END             = "2026-03-01"  # ISO 8601 UTC (exclusive)
VWAP_WINDOW     = 20
ENTRY_THRESHOLD = 0.001
TAKE_PROFIT     = 0.003
STOP_LOSS       = 0.015
TRADE_SIZE      = Decimal(1)
INITIAL_CASH    = 10_000.0
# ─────────────────────────────────────────────────────────────────────────────

KALSHI_VENUE = Venue("KALSHI")
USD_CURRENCY = Currency.from_str("USD")


# ---------------------------------------------------------------------------
# Strategy (same VWAP mean-reversion logic as polymarket_spread_capture.py)
# ---------------------------------------------------------------------------

class SpreadCaptureConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(1)
    vwap_window: int = 20
    entry_threshold: float = 0.005
    take_profit: float = 0.005
    stop_loss: float = 0.015


class SpreadCapture(Strategy):
    """
    Mean-reversion spread capture strategy.

    Buys when price dips below a rolling average and exits on recovery
    or stop-loss.  Holds at most one position at a time.
    """

    def __init__(self, config: SpreadCaptureConfig) -> None:
        super().__init__(config)
        self._prices: deque[float] = deque(maxlen=config.vwap_window)
        self._entry_price: float | None = None
        self._pending: bool = False
        self._instrument = None

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found -- stopping.")
            self.stop()
            return
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        self._prices.append(price)

        if len(self._prices) < self.config.vwap_window or self._pending:
            return

        rolling_avg = sum(self._prices) / len(self._prices)

        if self.portfolio.is_flat(self.config.instrument_id):
            if price <= rolling_avg - self.config.entry_threshold:
                self._buy()
        else:
            take_profit_hit = price >= self._entry_price + self.config.take_profit  # type: ignore[operator]
            stop_loss_hit = price <= self._entry_price - self.config.stop_loss  # type: ignore[operator]
            if take_profit_hit or stop_loss_hit:
                self.close_all_positions(self.config.instrument_id)
                self._pending = True

    def on_order_filled(self, event) -> None:
        if event.order_side == OrderSide.BUY:
            self._entry_price = float(event.last_px)
        else:
            self._entry_price = None
        self._pending = False

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)

    def _buy(self) -> None:
        assert self._instrument is not None
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self._instrument.make_qty(float(self.config.trade_size)),
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)
        self._pending = True


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------

async def _discover_tickers(max_markets: int) -> list[str]:
    """Query Kalshi REST API for top active markets by volume."""
    client = _KalshiHttpClient(base_url=KALSHI_REST_BASE)
    markets = await client.get_markets()

    def _vol(m: dict) -> float:
        with contextlib.suppress(TypeError, ValueError):
            return float(m.get("volume", 0) or 0)
        return 0.0

    markets.sort(key=_vol, reverse=True)
    return [m["ticker"] for m in markets if m.get("ticker")][:max_markets]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

async def _load_market(ticker: str, http_client: nautilus_pyo3.HttpClient) -> tuple | None:
    """Fetch bars for one Kalshi ticker and synthesize OHLC trade ticks."""
    try:
        loader = await KalshiDataLoader.from_market_ticker(ticker, http_client=http_client)
        bars = await loader.load_bars(
            start=pd.Timestamp(START, tz="UTC"),
            end=pd.Timestamp(END, tz="UTC"),
            interval="Hours1",
        )
    except Exception as exc:
        print(f"  skip {ticker}: {exc}")
        return None

    if not bars:
        print(f"  skip {ticker}: no bars returned")
        return None

    instrument = loader.instrument
    ticks: list[TradeTick] = []
    bar_ns = 3_600_000_000_000  # 1 hour in nanoseconds

    for i, bar in enumerate(bars):
        base_ts = bar.ts_event - bar_ns
        offsets = [bar_ns // 4, bar_ns // 2, 3 * bar_ns // 4, bar_ns]
        ohlc = [float(bar.open), float(bar.high), float(bar.low), float(bar.close)]
        for j, (offset, price) in enumerate(zip(offsets, ohlc, strict=True)):
            ticks.append(TradeTick(
                instrument_id=instrument.id,
                price=instrument.make_price(price),
                size=instrument.make_qty(1.0),
                aggressor_side=AggressorSide.NO_AGGRESSOR,
                trade_id=TradeId(f"SYN-{i:05d}-{j}"),
                ts_event=base_ts + offset,
                ts_init=base_ts + offset,
            ))

    if len(ticks) < MIN_TICKS:
        print(f"  skip {ticker}: only {len(ticks)} ticks (< {MIN_TICKS})")
        return None

    return instrument, ticks


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def _extract_pnl(pos_report: pd.DataFrame) -> float:
    """Parse total realized PnL from a positions report DataFrame."""
    total = 0.0
    for _, row in pos_report.iterrows():
        pnl_str = str(row.get("realized_pnl", "")).strip()
        if pnl_str and pnl_str.lower() != "nan":
            with contextlib.suppress(ValueError, IndexError):
                total += float(pnl_str.split()[0].replace("\u2212", "-"))
    return total


def _run_backtest(ticker: str, instrument, ticks: list) -> dict:
    """Run one market's backtest and return a results dict."""
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="INFO"),
        )
    )
    engine.add_venue(
        venue=KALSHI_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD_CURRENCY,
        starting_balances=[Money(INITIAL_CASH, USD_CURRENCY)],
        fee_model=KalshiProportionalFeeModel(),
    )
    engine.add_instrument(instrument)
    engine.add_data(ticks)
    engine.add_strategy(
        SpreadCapture(
            SpreadCaptureConfig(
                instrument_id=instrument.id,
                trade_size=TRADE_SIZE,
                vwap_window=VWAP_WINDOW,
                entry_threshold=ENTRY_THRESHOLD,
                take_profit=TAKE_PROFIT,
                stop_loss=STOP_LOSS,
            )
        )
    )
    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    pnl = _extract_pnl(positions)

    tearsheet_path = f"output/{NAME.replace(' ', '_')}_{ticker}_tearsheet.html"
    os.makedirs("output", exist_ok=True)
    create_tearsheet(engine, tearsheet_path, config=TearsheetConfig(theme="nautilus_dark"))

    engine.reset()
    engine.dispose()

    return {"ticker": ticker, "ticks": len(ticks), "fills": len(fills), "pnl": pnl}


def _print_summary(results: list[dict]) -> None:
    if not results:
        print("No markets had sufficient data.")
        return

    col_w = max(len(r["ticker"]) for r in results) + 2
    header = f"{'Market':<{col_w}} {'Ticks':>8} {'Fills':>6} {'PnL (USD)':>12}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(f"{r['ticker']:<{col_w}} {r['ticks']:>8} {r['fills']:>6} {r['pnl']:>+12.4f}")
    total_pnl = sum(r["pnl"] for r in results)
    total_fills = sum(r["fills"] for r in results)
    print(sep)
    print(f"{'TOTAL':<{col_w}} {'':>8} {total_fills:>6} {total_pnl:>+12.4f}")
    print(sep)


async def run() -> None:
    print(f"Discovering top {MAX_MARKETS} active Kalshi markets by volume...")
    tickers = await _discover_tickers(MAX_MARKETS)
    print(f"Found {len(tickers)} markets -> fetching bars in parallel...\n")

    http_client = nautilus_pyo3.HttpClient(
        default_quota=nautilus_pyo3.Quota.rate_per_second(10),
    )
    loaded = []
    for t in tickers:
        loaded.append(await _load_market(t, http_client))

    results: list[dict] = []
    for ticker, market_data in zip(tickers, loaded, strict=True):
        if market_data is None:
            continue
        instrument, ticks = market_data
        print(f"  {ticker}: {len(ticks)} ticks -> running backtest...")
        result = _run_backtest(ticker, instrument, ticks)
        results.append(result)

    _print_summary(results)
    if results:
        print(f"\nTearsheets saved to output/{NAME.replace(' ', '_')}_<ticker>_tearsheet.html")


if __name__ == "__main__":
    asyncio.run(run())
