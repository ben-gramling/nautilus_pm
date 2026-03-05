"""
Deep-value Polymarket strategy: buy low-priced outcomes and hold to resolution.

This script finds recent high-volume closed markets with CLOB trade history,
selects an outcome token that traded <= $0.20 and later resolved near 1.0,
runs a simple buy-and-hold strategy, and renders legacy Bokeh charts with
cumulative Brier advantage.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Any

import msgspec
import pandas as pd

from nautilus_trader.adapters.polymarket import POLYMARKET_VENUE
from nautilus_trader.adapters.polymarket import PolymarketDataLoader
from nautilus_trader.adapters.polymarket.common.gamma_markets import list_markets
from nautilus_trader.adapters.polymarket.common.gamma_markets import (
    normalize_gamma_market_to_clob_format,
)
from nautilus_trader.adapters.polymarket.common.parsing import parse_polymarket_instrument
from nautilus_trader.adapters.polymarket.fee_model import PolymarketFeeModel
from nautilus_trader.adapters.prediction_market.backtest_utils import build_market_prices
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_price_points
from nautilus_trader.adapters.prediction_market.backtest_utils import extract_realized_pnl
from nautilus_trader.analysis.legacy_plot_adapter import create_legacy_backtest_chart
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


NAME = "polymarket_deep_value_resolution_hold"
DESCRIPTION = "Buy low-priced outcome tokens <= 20c and hold to resolution"

MAX_MARKETS = int(os.getenv("MAX_MARKETS", "3"))
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "500"))
MIN_TRADES = int(os.getenv("MIN_TRADES", "200"))
MIN_VOLUME_NUM = float(os.getenv("MIN_VOLUME_NUM", "20000"))
START_DATE_MIN = os.getenv("START_DATE_MIN", "2024-01-01T00:00:00Z")

ENTRY_PRICE_MAX = float(os.getenv("ENTRY_PRICE_MAX", "0.20"))
RESOLUTION_PRICE_MIN = float(os.getenv("RESOLUTION_PRICE_MIN", "0.95"))
WIN_LAST_PRICE_MIN = float(os.getenv("WIN_LAST_PRICE_MIN", "0.90"))

TRADE_SIZE = Decimal(os.getenv("TRADE_SIZE", "20"))
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "1000"))


class DeepValueHoldConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(20)
    entry_price_max: float = ENTRY_PRICE_MAX


class DeepValueHold(Strategy):
    """
    Buy YES when market price falls below threshold, then hold until near-resolution.
    """

    def __init__(self, config: DeepValueHoldConfig) -> None:
        super().__init__(config)
        self._instrument = None
        self._pending = False
        self._entered = False
        self._completed = False

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self.config.instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found; stopping")
            self.stop()
            return
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_trade_tick(self, tick: TradeTick) -> None:
        if self._pending or self._completed:
            return

        price = float(tick.price)

        if (
            self.portfolio.is_flat(self.config.instrument_id)
            and not self._entered
            and price <= self.config.entry_price_max
        ):
            self._buy()

    def on_order_filled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False
        if event.order_side == OrderSide.BUY:
            self._entered = True
        else:
            self._completed = True

    def on_order_rejected(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_order_canceled(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pending = False

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        # Force final exit for clean realized PnL and visible exit marker on chart.
        if self._entered and not self.portfolio.is_flat(self.config.instrument_id):
            self.close_all_positions(self.config.instrument_id)

    def on_reset(self) -> None:
        self._instrument = None
        self._pending = False
        self._entered = False
        self._completed = False

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


def _parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str | bytes):
        try:
            decoded = msgspec.json.decode(value)
            return decoded if isinstance(decoded, list) else []
        except Exception:
            return []
    return []


def _extract_outcome_tokens(gamma_market: dict[str, Any]) -> list[tuple[str, str]]:
    outcomes = _parse_json_list(gamma_market.get("outcomes"))
    token_ids = _parse_json_list(gamma_market.get("clobTokenIds"))
    if not token_ids:
        return []

    pairs: list[tuple[str, str]] = []
    for idx, token_id in enumerate(token_ids):
        outcome = str(outcomes[idx]) if idx < len(outcomes) else f"OUTCOME_{idx + 1}"
        pairs.append((outcome, str(token_id)))
    return pairs


def _build_loader_from_gamma_market(
    gamma_market: dict[str, Any],
    token_id: str,
    outcome: str,
    http_client: nautilus_pyo3.HttpClient,
) -> PolymarketDataLoader | None:
    normalized = normalize_gamma_market_to_clob_format(gamma_market)
    condition_id = normalized.get("condition_id")
    if not condition_id:
        return None

    instrument = parse_polymarket_instrument(
        market_info=normalized,
        token_id=token_id,
        outcome=outcome,
    )

    return PolymarketDataLoader(
        instrument=instrument,
        token_id=token_id,
        condition_id=str(condition_id),
        http_client=http_client,
    )


def _build_probability_frame(
    trades: list[TradeTick],
    entry_price_max: float,
) -> pd.DataFrame:
    rows: list[tuple[pd.Timestamp, float]] = []
    for tick in trades:
        ts = pd.to_datetime(int(tick.ts_event), unit="ns", utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append((ts, float(tick.price)))

    frame = pd.DataFrame(rows, columns=["ts", "market_probability"]) if rows else pd.DataFrame()
    if frame.empty:
        return frame

    frame = frame.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").set_index("ts")
    frame["market_probability"] = frame["market_probability"].clip(0.0, 1.0)

    signal_mask = frame["market_probability"] <= entry_price_max
    if signal_mask.any():
        first_signal_ts = frame.index[signal_mask.argmax()]
        user_probability = frame["market_probability"].copy()
        user_probability.loc[first_signal_ts:] = 1.0
    else:
        user_probability = frame["market_probability"].copy()

    frame["user_probability"] = user_probability
    frame["outcome"] = 1.0
    frame = frame.dropna(subset=["user_probability", "market_probability", "outcome"])

    return frame


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _run_backtest(
    slug: str,
    outcome: str,
    loader: PolymarketDataLoader,
    trades: list[TradeTick],
) -> dict[str, Any]:
    instrument = loader.instrument

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="WARNING"),
            risk_engine=RiskEngineConfig(bypass=True),
        ),
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
        DeepValueHold(
            DeepValueHoldConfig(
                instrument_id=instrument.id,
                trade_size=TRADE_SIZE,
                entry_price_max=ENTRY_PRICE_MAX,
            ),
        ),
    )

    engine.run()

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    pnl = extract_realized_pnl(positions)

    prob_frame = _build_probability_frame(trades=trades, entry_price_max=ENTRY_PRICE_MAX)
    price_points = extract_price_points(trades, price_attr="price")

    safe_outcome = _slugify(outcome) or "outcome"
    output_path = f"output/{NAME}_{slug}_{safe_outcome}_legacy.html"
    os.makedirs("output", exist_ok=True)
    create_legacy_backtest_chart(
        engine=engine,
        output_path=output_path,
        strategy_name=f"{NAME}:{slug}:{outcome}",
        platform="polymarket",
        initial_cash=INITIAL_CASH,
        market_prices={str(instrument.id): build_market_prices(price_points)},
        user_probabilities=prob_frame.get("user_probability"),
        market_probabilities=prob_frame.get("market_probability"),
        outcomes=prob_frame.get("outcome"),
        open_browser=False,
        progress=False,
    )

    engine.reset()
    engine.dispose()

    prices = [float(t.price) for t in trades]
    return {
        "slug": slug,
        "outcome": outcome,
        "trades": len(trades),
        "fills": len(fills),
        "pnl": pnl,
        "entry_min": min(prices),
        "max": max(prices),
        "last": prices[-1],
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No qualifying resolved low-priced outcome markets were found.")
        return

    labels = [f"{r['slug']}:{r['outcome']}" for r in results]
    col_w = max(len(label) for label in labels) + 2
    header = (
        f"{'Market':<{col_w}} {'Trades':>7} {'Fills':>6} "
        f"{'Min Px':>8} {'Max Px':>8} {'Last Px':>8} {'PnL (USDC)':>12}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}\n{header}\n{sep}")
    for row, label in zip(results, labels, strict=False):
        print(
            f"{label:<{col_w}} {row['trades']:>7} {row['fills']:>6} "
            f"{row['entry_min']:>8.3f} {row['max']:>8.3f} {row['last']:>8.3f} "
            f"{row['pnl']:>+12.4f}"
        )

    total_pnl = sum(float(r["pnl"]) for r in results)
    total_fills = sum(int(r["fills"]) for r in results)
    print(sep)
    print(
        f"{'TOTAL':<{col_w}} {'':>7} {total_fills:>6} {'':>8} {'':>8} {'':>8} "
        f"{total_pnl:>+12.4f}"
    )
    print(sep)


def _get_price_stats(trades: list[TradeTick]) -> tuple[float, float, float]:
    prices = [float(t.price) for t in trades]
    return (min(prices), max(prices), prices[-1])


async def _evaluate_outcome_candidate(
    market: dict[str, Any],
    outcome: str,
    token_id: str,
    client: nautilus_pyo3.HttpClient,
) -> dict[str, Any] | None:
    loader = _build_loader_from_gamma_market(
        gamma_market=market,
        token_id=token_id,
        outcome=outcome,
        http_client=client,
    )
    if loader is None:
        return None

    try:
        trades = await loader.load_trades()
    except Exception:
        return None

    if len(trades) < MIN_TRADES:
        return None

    min_price, max_price, last_price = _get_price_stats(trades)
    if min_price > ENTRY_PRICE_MAX:
        return None
    if max_price < RESOLUTION_PRICE_MIN:
        return None
    if last_price < WIN_LAST_PRICE_MIN:
        return None

    return {
        "loader": loader,
        "trades": trades,
        "outcome": outcome,
        "min_price": min_price,
        "max_price": max_price,
        "last_price": last_price,
    }


def _market_filters() -> dict[str, Any]:
    filters: dict[str, Any] = {"closed": True, "archived": False, "limit": CANDIDATE_LIMIT}
    if START_DATE_MIN:
        filters["start_date_min"] = START_DATE_MIN
    return filters


async def _select_market_candidate(
    market: dict[str, Any],
    client: nautilus_pyo3.HttpClient,
    volume_fn,
) -> dict[str, Any] | None:
    slug = str(market.get("slug", ""))
    if not slug:
        return None

    volume_num = volume_fn(market.get("volumeNum"))
    if volume_num < MIN_VOLUME_NUM:
        return None

    outcome_tokens = _extract_outcome_tokens(market)
    if not outcome_tokens:
        return None

    best_match: dict[str, Any] | None = None
    for outcome, token_id in outcome_tokens:
        candidate = await _evaluate_outcome_candidate(
            market=market,
            outcome=outcome,
            token_id=token_id,
            client=client,
        )
        if candidate is None:
            continue
        if best_match is None or float(candidate["last_price"]) > float(best_match["last_price"]):
            best_match = candidate

    if best_match is None:
        return None

    return {
        "slug": slug,
        "loader": best_match["loader"],
        "trades": best_match["trades"],
        "outcome": best_match["outcome"],
        "volume_num": volume_num,
        "min_price": best_match["min_price"],
        "max_price": best_match["max_price"],
        "last_price": best_match["last_price"],
    }


async def _discover_candidates(
    client: nautilus_pyo3.HttpClient,
) -> tuple[list[dict[str, Any]], int]:
    def _volume(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    raw_markets = await list_markets(
        http_client=client,
        filters=_market_filters(),
        max_results=CANDIDATE_LIMIT,
    )
    ranked = sorted(
        raw_markets,
        key=lambda m: _volume(m.get("volumeNum")),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    checked = 0

    for market in ranked:
        if len(selected) >= MAX_MARKETS:
            break

        checked += 1
        candidate = await _select_market_candidate(market=market, client=client, volume_fn=_volume)
        if candidate is None:
            continue

        selected.append(candidate)

        print(
            f"  selected {candidate['slug']}:{candidate['outcome']} | "
            f"volume={candidate['volume_num']:.0f} trades={len(candidate['trades'])} "
            f"min={candidate['min_price']:.3f} max={candidate['max_price']:.3f} "
            f"last={candidate['last_price']:.3f}"
        )

    return selected, checked


async def run() -> None:
    client = nautilus_pyo3.HttpClient(
        default_quota=nautilus_pyo3.Quota.rate_per_second(10),
    )

    print(
        f"Discovering up to {MAX_MARKETS} high-volume closed markets "
        f"(entry <= {ENTRY_PRICE_MAX:.2f}, final >= {WIN_LAST_PRICE_MIN:.2f}, "
        f"since {START_DATE_MIN})..."
    )
    selected, checked = await _discover_candidates(client)

    if not selected:
        print(f"Checked {checked} markets; found no qualifying candidates.")
        return

    results: list[dict[str, Any]] = []
    for candidate in selected:
        slug = str(candidate["slug"])
        outcome = str(candidate["outcome"])
        print(f"  running backtest for {slug}:{outcome}...")
        result = _run_backtest(
            slug=slug,
            outcome=outcome,
            loader=candidate["loader"],
            trades=candidate["trades"],
        )
        results.append(result)

    _print_summary(results)
    print(f"\nLegacy charts saved to output/{NAME}_<slug>_<outcome>_legacy.html")


if __name__ == "__main__":
    asyncio.run(run())
