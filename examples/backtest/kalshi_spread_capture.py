"""
Bar-based mean-reversion (spread capture) on Kalshi minute bars.

Discovers active Kalshi markets using the same method as the Polymarket
spread capture: ranks by 24-hour volume, filters to markets with genuine
uncertainty (price in [PRICE_MIN, PRICE_MAX]) and enough time before
resolution, then runs a BarMeanReversion strategy on the top MAX_MARKETS
that have at least MIN_BARS of bar history.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from decimal import Decimal

import msgspec
import pandas as pd

from nautilus_trader.adapters.kalshi.fee_model import (
    KalshiProportionalFeeModel,
)
from nautilus_trader.adapters.kalshi.loaders import (
    KalshiDataLoader,
)
from nautilus_trader.adapters.kalshi.providers import (
    KALSHI_REST_BASE,
)
from nautilus_trader.adapters.kalshi.providers import (
    market_dict_to_instrument,
)
from nautilus_trader.analysis.config import TearsheetConfig
from nautilus_trader.analysis.tearsheet import create_tearsheet
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.risk.config import RiskEngineConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig


# ── Strategy metadata (shown in the menu) ────────────────────────────────────
NAME = "kalshi_spread_capture"
DESCRIPTION = "Mean-reversion spread capture across Kalshi markets"

# ── Configure here ────────────────────────────────────────────────────────────
LOOKBACK_DAYS = 7  # days of bar history to fetch per market
MIN_BARS = 20  # skip markets with fewer non-empty minute bars
MAX_MARKETS = 15  # how many qualifying markets to backtest
CANDIDATE_LIMIT = 400  # how many open markets to fetch and rank by volume
# Only trade markets whose YES price is in this range — avoids fully-resolved markets
PRICE_MIN = 0.05
PRICE_MAX = 0.95
# Markets must be at least this many days from resolution to avoid end-of-life drift
MIN_DAYS_TO_RESOLUTION = 2

WINDOW = 20  # rolling average window
ENTRY_THRESHOLD = 0.01  # enter when close is 1¢ below rolling average (0–1 scale)
TAKE_PROFIT = 0.01  # exit when price recovers 1¢ above fill price
STOP_LOSS = 0.03  # stop out 3¢ below fill price
TRADE_SIZE = Decimal(1)
INITIAL_CASH = 10_000.0
MAX_RETRIES = 4  # retry 429s up to this many times
RETRY_BASE_DELAY = 2.0  # seconds; doubles on each retry
# ─────────────────────────────────────────────────────────────────────────────


class BarMeanReversionConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal(1)
    window: int = 20
    entry_threshold: float = 1.0
    take_profit: float = 1.0
    stop_loss: float = 3.0


class BarMeanReversion(Strategy):
    """
    Mean-reversion spread capture on bar close prices.

    Buys when close dips below a rolling average by `entry_threshold`,
    exits when price recovers `take_profit` above fill, or stops out
    `stop_loss` below fill.  Holds at most one position at a time.
    """

    def __init__(self, config: BarMeanReversionConfig) -> None:
        super().__init__(config)
        self._prices: deque[float] = deque(maxlen=config.window)
        self._entry_price: float | None = None
        self._pending: bool = False
        self._instrument = None

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(
                f"Instrument {self.config.instrument_id} not found — stopping."
            )
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        price = float(bar.close)
        self._prices.append(price)

        if len(self._prices) < self.config.window or self._pending:
            return

        avg = sum(self._prices) / len(self._prices)

        if self.portfolio.is_flat(self.config.instrument_id):
            if price <= avg - self.config.entry_threshold:
                self._buy()
        else:
            assert self._entry_price is not None
            take_profit_hit = price >= self._entry_price + self.config.take_profit
            stop_loss_hit = price <= self._entry_price - self.config.stop_loss
            if take_profit_hit or stop_loss_hit:
                self.close_all_positions(self.config.instrument_id)
                self._pending = True

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.order_side == OrderSide.BUY:
            self._entry_price = float(event.last_px)
        else:
            self._entry_price = None
        self._pending = False

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
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


def _vol24h(m: dict) -> float:
    """Extract volume from a Kalshi market dict as a float.

    Tries ``volume_24h``, ``volume_fp`` (REST markets endpoint), and
    ``volume`` (candlestick endpoint) in order.
    """
    for key in ("volume_24h", "volume_fp", "volume"):
        raw = m.get(key)
        if raw is not None:
            try:
                v = float(raw)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                continue
    return 0.0


def _yes_price_kalshi(m: dict) -> float | None:
    """Extract and normalize the current YES price from a Kalshi market dict.

    The REST ``/markets`` and ``/events`` endpoints expose the price under
    ``last_price_dollars`` (decimal 0–1 string).  Older fields like
    ``yes_bid_dollars``, ``yes_price_dollars``, and the legacy integer-cents
    ``yes_price`` are tried as fallbacks.
    """
    for key in ("last_price_dollars", "yes_bid_dollars", "yes_price_dollars", "yes_price"):
        raw = m.get(key)
        if raw is not None:
            try:
                p = float(raw)
                if p >= 1.0:
                    p /= 100.0  # legacy integer cents → dollars
                if 0.0 < p < 1.0:
                    return p
            except (TypeError, ValueError):
                continue
    return None


def _end_date_kalshi(m: dict) -> datetime | None:
    """Parse the market expiry from a Kalshi market dict."""
    raw = m.get("close_time") or m.get("latest_expiration_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


async def _discover_markets(
    candidate_limit: int,
    http_client: nautilus_pyo3.HttpClient,
) -> list[dict]:
    """
    Discover open Kalshi markets sorted by 24-hour volume descending.

    Mirrors the Polymarket discovery method: ranks by 24h activity, filters
    to markets with genuine uncertainty (price in [PRICE_MIN, PRICE_MAX]) and
    enough time left before resolution (>= MIN_DAYS_TO_RESOLUTION out).

    Uses the ``/events`` endpoint with ``with_nested_markets=true`` to obtain
    both the ``series_ticker`` (on the event) and full market dicts in one pass.
    Each returned market dict is augmented with a ``series_ticker`` key so that
    ``_load_market`` can build a loader without extra API calls.

    The ``/markets`` list endpoint is dominated by KXMVE parlay stubs that lack
    candlestick data, so we avoid it entirely.
    """
    all_markets: list[dict] = []
    cursor: str | None = None

    while len(all_markets) < candidate_limit:
        params: dict[str, str] = {
            "status": "open",
            "limit": "200",
            "with_nested_markets": "true",
        }
        if cursor:
            params["cursor"] = cursor

        resp = await http_client.get(
            url=f"{KALSHI_REST_BASE}/events",
            params=params,
        )
        if resp.status != 200:
            break

        data = msgspec.json.decode(resp.body)
        events = data.get("events", [])
        if not events:
            break

        for event in events:
            series_ticker = event.get("series_ticker", "")
            nested = event.get("markets") or []
            for mkt in nested:
                # Skip KXMVE parlay/multi-event markets — no candlestick data.
                if mkt.get("ticker", "").startswith("KXMVE"):
                    continue
                # Attach series_ticker so _load_market doesn't need an extra call.
                mkt["series_ticker"] = series_ticker
                all_markets.append(mkt)

        cursor = data.get("cursor")
        if not cursor:
            break

    now = datetime.now(UTC)
    min_end = now + timedelta(days=MIN_DAYS_TO_RESOLUTION)

    skipped = {"price": 0, "end_date": 0, "no_volume": 0}
    filtered: list[dict] = []
    for m in all_markets:
        if _vol24h(m) <= 0:
            skipped["no_volume"] += 1
            continue
        price = _yes_price_kalshi(m)
        if price is None or not (PRICE_MIN <= price <= PRICE_MAX):
            skipped["price"] += 1
            continue
        end = _end_date_kalshi(m)
        if end is not None and end < min_end:
            skipped["end_date"] += 1
            continue
        filtered.append(m)

    filtered.sort(key=_vol24h, reverse=True)
    print(
        f"  Selected {min(len(filtered), candidate_limit)} markets "
        f"(skipped: {skipped['price']} by price, "
        f"{skipped['end_date']} resolving soon, "
        f"{skipped['no_volume']} inactive)"
    )
    return filtered[:candidate_limit]


async def _load_market(
    market: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    http_client: nautilus_pyo3.HttpClient,
) -> tuple[KalshiDataLoader, list[Bar]] | None:
    """
    Fetch bars for one market.

    Expects ``market`` to already contain ``series_ticker`` (set by
    ``_discover_markets``).
    """
    ticker = market["ticker"]
    try:
        instrument = market_dict_to_instrument(market)
        series_ticker = market["series_ticker"]
        loader = KalshiDataLoader(
            instrument=instrument,
            series_ticker=series_ticker,
            http_client=http_client,
        )
        # Chunk requests to stay under the 5 000-candle API cap.
        chunk_delta = pd.Timedelta(minutes=5_000)
        chunk_start = start
        bars: list[Bar] = []
        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_delta, end)
            # Retry with exponential backoff on 429.
            for attempt in range(MAX_RETRIES + 1):
                try:
                    bars.extend(
                        await loader.load_bars(
                            start=chunk_start,
                            end=chunk_end,
                            interval="Minutes1",
                        )
                    )
                    break
                except RuntimeError as rt_err:
                    if "429" in str(rt_err) and attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2**attempt)
                        print(
                            f"    rate-limited on {ticker}, retrying in {delay:.0f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise
            chunk_start = chunk_end
        if len(bars) < MIN_BARS:
            print(f"  skip {ticker}: fewer than {MIN_BARS} bars")
            return None
        return loader, bars
    except Exception as exc:
        print(f"  skip {ticker}: {exc}")
        return None


def _extract_pnl(pos_report: pd.DataFrame) -> float:
    """Parse total realized PnL from a positions report DataFrame."""
    total = 0.0
    for _, row in pos_report.iterrows():
        pnl_str = str(row.get("realized_pnl", "")).strip()
        if pnl_str and pnl_str.lower() != "nan":
            try:  # noqa: SIM105
                # Money strings look like "-1.00 USD"; handle unicode minus.
                total += float(pnl_str.split()[0].replace("\u2212", "-"))
            except (ValueError, IndexError):
                pass
    return total


def _run_backtest(ticker: str, loader: KalshiDataLoader, bars: list[Bar]) -> dict:
    """Run one market's backtest and return a results dict."""
    instrument = loader.instrument
    bar_type = bars[0].bar_type

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="INFO"),
            risk_engine=RiskEngineConfig(bypass=True),
        )
    )
    engine.add_venue(
        venue=Venue("KALSHI"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(INITIAL_CASH, USD)],
        fee_model=KalshiProportionalFeeModel(),
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(
        BarMeanReversion(
            config=BarMeanReversionConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
                trade_size=TRADE_SIZE,
                window=WINDOW,
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

    tearsheet_path = f"output/{NAME}_{ticker}_tearsheet.html"
    os.makedirs("output", exist_ok=True)
    create_tearsheet(
        engine, tearsheet_path, config=TearsheetConfig(theme="nautilus_dark")
    )

    engine.reset()
    engine.dispose()

    return {
        "ticker": ticker,
        "bars": len(bars),
        "fills": len(fills),
        "pnl": pnl,
    }


def _print_summary(results: list[dict]) -> None:
    if not results:
        print("No markets had sufficient data.")
        return

    col_w = max(len(r["ticker"]) for r in results) + 2
    header = f"{'Market':<{col_w}} {'Bars':>6} {'Fills':>6} {'PnL (USD)':>12}"
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(
            f"{r['ticker']:<{col_w}} {r['bars']:>6} {r['fills']:>6} {r['pnl']:>+12.4f}"
        )
    total_pnl = sum(r["pnl"] for r in results)
    total_fills = sum(r["fills"] for r in results)
    print(sep)
    print(f"{'TOTAL':<{col_w}} {'':>6} {total_fills:>6} {total_pnl:>+12.4f}")
    print(sep)


async def run() -> None:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=LOOKBACK_DAYS))
    end = pd.Timestamp(now)

    # Single shared client — conservative rate to avoid 429s.
    http_client = nautilus_pyo3.HttpClient(
        default_quota=nautilus_pyo3.Quota.rate_per_second(10),
    )

    print(f"Discovering top {MAX_MARKETS} active Kalshi markets by 24h volume...")
    candidates = await _discover_markets(CANDIDATE_LIMIT, http_client)
    print(f"Found {len(candidates)} markets → scanning for {MIN_BARS}+ bars...")

    # Brief pause to let the rate-limit window reset after discovery.
    await asyncio.sleep(2)

    results: list[dict] = []
    for market in candidates:
        if len(results) >= MAX_MARKETS:
            break
        market_data = await _load_market(market, start, end, http_client)
        if market_data is None:
            continue
        loader, bars = market_data
        ticker = market["ticker"]
        print(f"  {ticker}: {len(bars)} bars → running backtest...")
        result = _run_backtest(ticker, loader, bars)
        results.append(result)

    _print_summary(results)
    print(f"\nTearsheets saved to output/{NAME}_<ticker>_tearsheet.html")
