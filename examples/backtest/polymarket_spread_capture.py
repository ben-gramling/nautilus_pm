"""
Mean-reversion spread capture on Polymarket trade ticks.

Discovers active markets from the Polymarket Gamma API, fetches trade ticks,
and runs a VWAP-based mean-reversion backtest across multiple markets.
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

from nautilus_trader.adapters.polymarket import POLYMARKET_VENUE
from nautilus_trader.adapters.polymarket import (
    PolymarketDataLoader,
)
from nautilus_trader.adapters.polymarket.common.gamma_markets import list_markets
from nautilus_trader.adapters.polymarket.fee_model import PolymarketFeeModel
from nautilus_trader.analysis.config import TearsheetConfig
from nautilus_trader.analysis.tearsheet import create_tearsheet
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.model.currencies import USDC_POS
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Money
from nautilus_trader.risk.config import RiskEngineConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig


# ── Strategy metadata (shown in the menu) ────────────────────────────────────
NAME = "polymarket_spread_capture"
DESCRIPTION = "Mean-reversion spread capture across Polymarket markets"

# ── Configure here ────────────────────────────────────────────────────────────
LOOKBACK_DAYS = 7  # days of 1-min price-history ticks to fetch (API max ~10 days)
MAX_MARKETS = 15
MIN_TRADES = 100           # need enough oscillations for mean-reversion
VWAP_WINDOW = 20
ENTRY_THRESHOLD = 0.005    # 0.5% deviation from rolling avg to enter
TAKE_PROFIT = 0.008        # 0.8% recovery to exit with profit
STOP_LOSS = 0.020          # 2.0% adverse move to cut loss
# Only trade markets whose YES price is in this range — avoids near-resolved markets
PRICE_MIN = 0.25
PRICE_MAX = 0.75
TRADE_SIZE = Decimal(20)
INITIAL_CASH = 10_000.0
# ─────────────────────────────────────────────────────────────────────────────


class SpreadCaptureConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(20)
    vwap_window: int = 20
    entry_threshold: float = ENTRY_THRESHOLD
    take_profit: float = TAKE_PROFIT
    stop_loss: float = STOP_LOSS


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
            self.log.error(
                f"Instrument {self.config.instrument_id} not found — stopping."
            )
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


async def _discover_slugs(max_markets: int) -> list[str]:
    """Query Polymarket Gamma API for markets suited to spread capture.

    Selection criteria (all must pass):
    - YES price in [PRICE_MIN, PRICE_MAX] — genuine uncertainty, not near-resolved
    - endDate at least LOOKBACK_DAYS out — won't trend monotonically to resolution
    - volumeNum24hr > 0 — actively trading today

    Sorted by 24h volume (not lifetime) so we get markets with current activity.
    """
    client = nautilus_pyo3.HttpClient(
        default_quota=nautilus_pyo3.Quota.rate_per_second(20),
    )
    markets = await list_markets(
        http_client=client,
        filters={"is_active": True, "limit": 200},
        max_results=200,
    )
    if not markets:
        return []

    now = datetime.now(UTC)
    min_end = now + timedelta(days=LOOKBACK_DAYS)

    def _vol24h(m: dict) -> float:
        try:
            return float(m.get("volume24hr", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _yes_price(m: dict) -> float | None:
        raw = m.get("outcomePrices")
        if not raw:
            return None
        try:
            prices = msgspec.json.decode(raw) if isinstance(raw, bytes | str) else raw
            return float(prices[0])
        except Exception:
            return None

    def _end_date(m: dict) -> datetime | None:
        raw = m.get("endDate") or m.get("end_date_iso")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return None

    markets.sort(key=_vol24h, reverse=True)

    slugs: list[str] = []
    skipped = {"price": 0, "end_date": 0, "no_volume": 0}
    for m in markets:
        slug = m.get("slug", "")
        if not slug:
            continue
        if _vol24h(m) <= 0:
            skipped["no_volume"] += 1
            continue
        price = _yes_price(m)
        if price is None or not (PRICE_MIN <= price <= PRICE_MAX):
            skipped["price"] += 1
            continue
        end = _end_date(m)
        if end is not None and end < min_end:
            skipped["end_date"] += 1
            continue
        slugs.append(slug)
        if len(slugs) >= max_markets:
            break

    print(
        f"  Selected {len(slugs)} markets "
        f"(skipped: {skipped['price']} by price, "
        f"{skipped['end_date']} resolving soon, "
        f"{skipped['no_volume']} inactive)"
    )
    return slugs


async def _load_market(slug: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[PolymarketDataLoader, list] | None:
    """Fetch price-history ticks for one market slug.  Returns None if data is insufficient."""
    try:
        loader = await PolymarketDataLoader.from_market_slug(slug)
        trades = await loader.load_trades(start, end)
        if len(trades) < MIN_TRADES:
            return None
        return loader, trades
    except Exception as exc:
        print(f"  skip {slug}: {exc}")
        return None


def _extract_pnl(pos_report: pd.DataFrame) -> float:
    """Parse total realized PnL from a positions report DataFrame."""
    total = 0.0
    for _, row in pos_report.iterrows():
        pnl_str = str(row.get("realized_pnl", "")).strip()
        if pnl_str and pnl_str.lower() != "nan":
            try:  # noqa: SIM105
                total += float(pnl_str.split()[0].replace("\u2212", "-"))
            except (ValueError, IndexError):
                pass
    return total


def _run_backtest(slug: str, loader: PolymarketDataLoader, trades: list) -> dict:
    """Run one market's backtest and return a results dict."""
    instrument = loader.instrument

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="INFO"),
            risk_engine=RiskEngineConfig(bypass=True),
        )
    )
    engine.add_venue(
        venue=POLYMARKET_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USDC_POS,
        starting_balances=[Money(INITIAL_CASH, USDC_POS)],
        fee_model=PolymarketFeeModel(),
    )
    engine.add_instrument(instrument)
    engine.add_data(trades)
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

    tearsheet_path = f"output/{NAME}_{slug}_tearsheet.html"
    os.makedirs("output", exist_ok=True)
    create_tearsheet(
        engine, tearsheet_path, config=TearsheetConfig(theme="nautilus_dark")
    )

    engine.reset()
    engine.dispose()

    return {"slug": slug, "trades": len(trades), "fills": len(fills), "pnl": pnl}


def _print_summary(results: list[dict]) -> None:
    if not results:
        print("No markets had sufficient data.")
        return

    col_w = max(len(r["slug"]) for r in results) + 2
    header = f"{'Market':<{col_w}} {'Trades':>8} {'Fills':>6} {'PnL (USDC)':>12}"
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(
            f"{r['slug']:<{col_w}} {r['trades']:>8} {r['fills']:>6} {r['pnl']:>+12.4f}"
        )
    total_pnl = sum(r["pnl"] for r in results)
    total_fills = sum(r["fills"] for r in results)
    print(sep)
    print(f"{'TOTAL':<{col_w}} {'':>8} {total_fills:>6} {total_pnl:>+12.4f}")
    print(sep)


async def run() -> None:
    now = datetime.now(UTC)
    start = pd.Timestamp(now - timedelta(days=LOOKBACK_DAYS))
    end = pd.Timestamp(now)

    print(f"Discovering top {MAX_MARKETS} active Polymarket markets by volume...")
    slugs = await _discover_slugs(MAX_MARKETS)
    print(f"Found {len(slugs)} markets → fetching trades in parallel...\n")

    loaded = await asyncio.gather(*[_load_market(s, start, end) for s in slugs])

    results: list[dict] = []
    for slug, market_data in zip(slugs, loaded, strict=False):
        if market_data is None:
            print(f"  skip {slug}: fewer than {MIN_TRADES} trades")
            continue
        loader, trades = market_data
        print(f"  {slug}: {len(trades)} trades → running backtest...")
        result = _run_backtest(slug, loader, trades)
        results.append(result)

    _print_summary(results)
    print(f"\nTearsheets saved to output/{NAME}_<slug>_tearsheet.html")
