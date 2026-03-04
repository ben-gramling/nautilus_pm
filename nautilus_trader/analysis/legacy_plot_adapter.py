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
"""
Bridge Nautilus backtest results into the legacy prediction-market plotting framework.

This adapter maps Nautilus reports into the `BacktestResult` expected by
`prediction-market-backtesting` legacy charts and appends a cumulative Brier
advantage panel.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nautilus_trader.analysis.reporter import ReportProvider


DEFAULT_LEGACY_CLONE = "https://github.com/evan-kolberg/prediction-market-backtesting.git"
DEFAULT_LEGACY_WORKTREE = Path("/tmp/prediction-market-backtesting-legacy")  # noqa: S108


def _parse_float(value: Any, default: float = 0.0) -> float:
    """
    Parse a float from numbers and money-like strings.
    """
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return default

    text = text.replace("_", "").replace("\u2212", "-")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match is None:
        return default

    try:
        return float(match.group(0))
    except ValueError:
        return default


def _to_naive_utc(value: Any) -> datetime | None:
    """
    Convert a timestamp-like value to naive UTC datetime.
    """
    if value is None:
        return None

    # Handle raw nanosecond timestamps common in Nautilus reports.
    if isinstance(value, int | float) and abs(float(value)) > 1e12:
        ts = pd.to_datetime(int(value), unit="ns", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(ts):
        return None

    if isinstance(ts, pd.DatetimeIndex):
        if len(ts) == 0:
            return None
        ts = ts[0]

    assert isinstance(ts, pd.Timestamp)
    return ts.tz_convert("UTC").tz_localize(None).to_pydatetime()


def _first_value(row: pd.Series, *keys: str) -> Any:
    for key in keys:
        if key in row.index:
            value = row[key]
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                return value
    return None


def prepare_cumulative_brier_advantage(
    user_probabilities: pd.Series | None = None,
    market_probabilities: pd.Series | None = None,
    outcomes: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute cumulative Brier advantage through time.

    Advantage is `market_brier - strategy_brier` where Brier score is
    `(p - y)^2`.
    """
    if user_probabilities is None or market_probabilities is None or outcomes is None:
        return pd.DataFrame()

    frame = pd.concat(
        [
            user_probabilities.rename("user_probability"),
            market_probabilities.rename("market_probability"),
            outcomes.rename("outcome"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if frame.empty:
        return frame

    for col in ("user_probability", "market_probability", "outcome"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna()
    if frame.empty:
        return frame

    frame = frame.sort_index()
    frame["user_probability"] = frame["user_probability"].clip(0.0, 1.0)
    frame["market_probability"] = frame["market_probability"].clip(0.0, 1.0)
    frame["outcome"] = frame["outcome"].clip(0.0, 1.0)

    frame["user_brier"] = (frame["user_probability"] - frame["outcome"]) ** 2
    frame["market_brier"] = (frame["market_probability"] - frame["outcome"]) ** 2
    frame["brier_advantage"] = frame["market_brier"] - frame["user_brier"]
    frame["cumulative_brier_advantage"] = frame["brier_advantage"].cumsum()

    return frame


def _candidate_legacy_paths(explicit_path: str | Path | None) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]

    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())

    env_repo = os.environ.get("NAUTILUS_PM_LEGACY_PLOT_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser())

    candidates.extend(
        [
            repo_root.parent / "prediction-market-backtesting",
            Path("/Users/evankolberg/prediction-market-backtesting"),
            Path("/tmp/pmb_legacy_22772"),  # noqa: S108
            DEFAULT_LEGACY_WORKTREE,
        ],
    )

    # Deduplicate while preserving order.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except FileNotFoundError:
            resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)

    return deduped


def _path_has_legacy_plotting(repo_path: Path) -> bool:
    return (repo_path / "src" / "backtesting" / "plotting.py").exists()


def _git_has_legacy_branch(repo_path: Path) -> bool:
    if not (repo_path / ".git").exists():
        return False

    try:
        proc = subprocess.run(  # noqa: S603,S607
            ["git", "-C", str(repo_path), "rev-parse", "--verify", "legacy"],
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0
    except OSError:
        return False


def _prepare_legacy_worktree(repo_path: Path) -> Path | None:
    if not _git_has_legacy_branch(repo_path):
        return None

    if _path_has_legacy_plotting(DEFAULT_LEGACY_WORKTREE):
        return DEFAULT_LEGACY_WORKTREE

    if DEFAULT_LEGACY_WORKTREE.exists() and not (DEFAULT_LEGACY_WORKTREE / ".git").exists():
        shutil.rmtree(DEFAULT_LEGACY_WORKTREE, ignore_errors=True)

    try:
        subprocess.run(  # noqa: S603,S607
            [
                "git",
                "-C",
                str(repo_path),
                "worktree",
                "add",
                "--force",
                "--detach",
                str(DEFAULT_LEGACY_WORKTREE),
                "legacy",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        # Worktree might already exist but in a stale state.
        try:
            subprocess.run(  # noqa: S603,S607
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "worktree",
                    "remove",
                    "--force",
                    str(DEFAULT_LEGACY_WORKTREE),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            shutil.rmtree(DEFAULT_LEGACY_WORKTREE, ignore_errors=True)
            subprocess.run(  # noqa: S603,S607
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "worktree",
                    "add",
                    "--force",
                    "--detach",
                    str(DEFAULT_LEGACY_WORKTREE),
                    "legacy",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            return None

    return DEFAULT_LEGACY_WORKTREE if _path_has_legacy_plotting(DEFAULT_LEGACY_WORKTREE) else None


def _clone_legacy_repo(target_path: Path) -> Path | None:
    if _path_has_legacy_plotting(target_path):
        return target_path

    if target_path.exists():
        shutil.rmtree(target_path, ignore_errors=True)

    try:
        subprocess.run(  # noqa: S603,S607
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "legacy",
                DEFAULT_LEGACY_CLONE,
                str(target_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None

    return target_path if _path_has_legacy_plotting(target_path) else None


def resolve_legacy_plot_repo(explicit_path: str | Path | None = None) -> Path:
    """
    Resolve a local checkout containing `src/backtesting/plotting.py`.
    """
    for candidate in _candidate_legacy_paths(explicit_path):
        if _path_has_legacy_plotting(candidate):
            return candidate

        worktree = _prepare_legacy_worktree(candidate)
        if worktree is not None and _path_has_legacy_plotting(worktree):
            return worktree

    cloned = _clone_legacy_repo(DEFAULT_LEGACY_WORKTREE)
    if cloned is not None:
        return cloned

    raise FileNotFoundError(
        "Could not locate legacy prediction-market plotting repo. "
        "Set NAUTILUS_PM_LEGACY_PLOT_REPO to a checkout containing "
        "src/backtesting/plotting.py (legacy branch).",
    )


def _load_legacy_modules(repo_path: Path) -> tuple[Any, Any]:
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    importlib.invalidate_caches()
    models = importlib.import_module("src.backtesting.models")
    plotting = importlib.import_module("src.backtesting.plotting")
    return models, plotting


def _extract_account_report(engine: Any) -> pd.DataFrame:
    accounts = []

    if hasattr(engine, "cache"):
        try:
            accounts = list(engine.cache.accounts())
        except Exception:  # pragma: no cover - defensive fallback
            accounts = []

    if not accounts and hasattr(engine, "kernel") and hasattr(engine.kernel, "cache"):
        try:
            accounts = list(engine.kernel.cache.accounts())
        except Exception:  # pragma: no cover - defensive fallback
            accounts = []

    if not accounts:
        raise ValueError("No accounts were found on the backtest engine cache.")

    report = ReportProvider.generate_account_report(accounts[0])
    if report.empty:
        raise ValueError("Account report is empty; cannot build chart.")

    frame = report.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()

    if frame.empty:
        raise ValueError("Account report has no valid timestamps.")

    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    frame = frame.groupby(frame.index).last().sort_index()
    return frame


def _infer_market_side(models_module: Any, market_id: str) -> Any:
    token = market_id.upper()
    if token.endswith("NO") or "-NO" in token or ".NO." in token or "_NO" in token:
        return models_module.Side.NO
    return models_module.Side.YES


def _signed_quantity(action: str, side: str, qty: float) -> float:
    if action == "buy" and side == "yes":
        return qty
    if action == "sell" and side == "yes":
        return -qty
    if action == "buy" and side == "no":
        return -qty
    if action == "sell" and side == "no":
        return qty
    return 0.0


def _convert_fills(fills_report: pd.DataFrame, models_module: Any) -> list[Any]:
    if fills_report is None or fills_report.empty:
        return []

    frame = fills_report.copy()
    if frame.index.name and frame.index.name not in frame.columns:
        frame = frame.reset_index()

    converted: list[Any] = []
    for idx, (_, row) in enumerate(frame.iterrows(), start=1):
        market_id = str(
            _first_value(row, "market_id", "instrument_id", "ticker", "symbol") or "",
        ).strip()
        if not market_id:
            continue

        timestamp = _to_naive_utc(
            _first_value(row, "ts_event", "ts_init", "ts_last", "timestamp", "datetime"),
        )
        if timestamp is None:
            continue

        action_raw = str(_first_value(row, "order_side", "action", "side") or "").upper()
        action = (
            models_module.OrderAction.BUY
            if action_raw == "BUY"
            else models_module.OrderAction.SELL
        )

        side = _infer_market_side(models_module, market_id)
        price = _parse_float(_first_value(row, "last_px", "avg_px", "price"), default=0.0)
        quantity = _parse_float(
            _first_value(row, "last_qty", "filled_qty", "quantity", "qty"),
            default=0.0,
        )
        if quantity <= 0:
            continue

        commission = _parse_float(
            _first_value(row, "commission", "commissions", "fee", "fees"),
            default=0.0,
        )
        order_id = str(
            _first_value(row, "order_id", "client_order_id", "venue_order_id")
            or f"fill-{idx}",
        )

        converted.append(
            models_module.Fill(
                order_id=order_id,
                market_id=market_id,
                action=action,
                side=side,
                price=price,
                quantity=quantity,
                timestamp=timestamp,
                commission=commission,
            ),
        )

    converted.sort(key=lambda fill: fill.timestamp)
    return converted


def _position_count_by_snapshot(snapshot_times: list[datetime], fills: list[Any]) -> list[int]:
    if not snapshot_times:
        return []

    counts: list[int] = []
    position_qty: dict[str, float] = {}
    fill_idx = 0

    for snapshot_time in snapshot_times:
        while fill_idx < len(fills) and fills[fill_idx].timestamp <= snapshot_time:
            fill = fills[fill_idx]
            signed_qty = _signed_quantity(fill.action.value, fill.side.value, float(fill.quantity))
            if signed_qty != 0.0:
                market_qty = position_qty.get(fill.market_id, 0.0) + signed_qty
                if abs(market_qty) < 1e-12:
                    position_qty.pop(fill.market_id, None)
                else:
                    position_qty[fill.market_id] = market_qty
            fill_idx += 1

        counts.append(len(position_qty))

    return counts


def _build_portfolio_snapshots(
    models_module: Any,
    account_report: pd.DataFrame,
    fills: list[Any],
) -> list[Any]:
    snapshot_times = [ts.to_pydatetime() for ts in account_report.index]
    num_positions = _position_count_by_snapshot(snapshot_times, fills)

    snapshots: list[Any] = []
    for idx, timestamp in enumerate(snapshot_times):
        row = account_report.iloc[idx]
        total_equity = _parse_float(row.get("total", row.get("equity", 0.0)), default=0.0)
        cash = _parse_float(row.get("free", row.get("cash", total_equity)), default=total_equity)
        unrealized_pnl = total_equity - cash

        snapshots.append(
            models_module.PortfolioSnapshot(
                timestamp=timestamp,
                cash=cash,
                total_equity=total_equity,
                unrealized_pnl=unrealized_pnl,
                num_positions=num_positions[idx] if idx < len(num_positions) else 0,
            ),
        )

    return snapshots


def _build_market_pnls(positions_report: pd.DataFrame) -> dict[str, float]:
    if positions_report is None or positions_report.empty:
        return {}

    frame = positions_report.copy()
    if frame.index.name and frame.index.name not in frame.columns:
        frame = frame.reset_index()

    pnls: dict[str, float] = {}
    for _, row in frame.iterrows():
        market_id = str(
            _first_value(row, "market_id", "instrument_id", "ticker", "symbol") or "",
        ).strip()
        if not market_id:
            continue
        realized = _parse_float(_first_value(row, "realized_pnl", "pnl"), default=0.0)
        pnls[market_id] = pnls.get(market_id, 0.0) + realized

    return pnls


def _normalize_market_prices(
    market_prices: Mapping[str, Sequence[tuple[Any, float]]] | None,
) -> dict[str, list[tuple[datetime, float]]]:
    if not market_prices:
        return {}

    normalized: dict[str, list[tuple[datetime, float]]] = {}
    for market_id, points in market_prices.items():
        values: list[tuple[datetime, float]] = []
        for ts_like, price_like in points:
            timestamp = _to_naive_utc(ts_like)
            if timestamp is None:
                continue
            values.append((timestamp, float(price_like)))

        if not values:
            continue

        # Keep the latest value for duplicate timestamps.
        frame = pd.DataFrame(values, columns=["ts", "price"]).sort_values("ts")
        frame = frame.drop_duplicates(subset=["ts"], keep="last")
        normalized[str(market_id)] = [
            (row.ts.to_pydatetime(), float(row.price))
            for row in frame.itertuples(index=False)
        ]

    return normalized


def _market_prices_from_fills(fills: list[Any]) -> dict[str, list[tuple[datetime, float]]]:
    market_prices: dict[str, list[tuple[datetime, float]]] = {}
    for fill in fills:
        market_prices.setdefault(fill.market_id, []).append((fill.timestamp, float(fill.price)))
    return market_prices


def _build_metrics(snapshots: list[Any], initial_cash: float) -> dict[str, float]:
    if not snapshots:
        return {}

    equity = pd.Series([float(snapshot.total_equity) for snapshot in snapshots])
    running_max = equity.cummax().replace(0, pd.NA)
    drawdown = (running_max - equity) / running_max

    final_equity = float(equity.iloc[-1])
    total_return = (final_equity - initial_cash) / initial_cash if initial_cash else 0.0

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": float(drawdown.max(skipna=True) or 0.0),
    }


def _platform_enum(models_module: Any, platform: str) -> Any:
    platform_lower = platform.lower()
    if "poly" in platform_lower:
        return models_module.Platform.POLYMARKET
    return models_module.Platform.KALSHI


def _append_brier_panel(layout: Any, brier_frame: pd.DataFrame) -> Any:
    if brier_frame.empty:
        return layout

    try:
        from bokeh.layouts import column
        from bokeh.models import ColumnDataSource
        from bokeh.models import HoverTool
        from bokeh.models import NumeralTickFormatter
        from bokeh.models import Span
        from bokeh.plotting import figure
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("Bokeh is required for legacy chart rendering.") from exc

    frame = brier_frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    if frame.empty:
        return layout

    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    frame = frame.drop_duplicates(keep="last")

    source = ColumnDataSource(
        {
            "datetime": frame.index,
            "cumulative_brier_advantage": frame["cumulative_brier_advantage"].to_numpy(),
            "brier_advantage": frame["brier_advantage"].to_numpy(),
        },
    )

    fig = figure(
        title="Cumulative Brier Advantage",
        x_axis_type="datetime",
        height=220,
        tools="xpan,xwheel_zoom,box_zoom,undo,redo,reset,save",
        active_drag="xpan",
        active_scroll="xwheel_zoom",
        sizing_mode="stretch_width",
        toolbar_location="right",
    )

    fig.line(
        x="datetime",
        y="cumulative_brier_advantage",
        source=source,
        line_width=2.0,
        line_color="#2ca0f0",
        legend_label="Cum. Brier Advantage",
    )
    fig.add_layout(
        Span(
            location=0,
            dimension="width",
            line_color="#666666",
            line_dash="dashed",
            line_width=1,
        ),
    )

    fig.add_tools(
        HoverTool(
            mode="vline",
            formatters={"@datetime": "datetime"},
            tooltips=[
                ("Date", "@datetime{%F %T}"),
                ("Cum Advantage", "@cumulative_brier_advantage{0.0000}"),
                ("Point Advantage", "@brier_advantage{0.0000}"),
            ],
        ),
    )

    fig.xaxis.axis_label = "Date"
    fig.yaxis.axis_label = "Market Brier - Strategy Brier"
    fig.yaxis.formatter = NumeralTickFormatter(format="0.0000")
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    if hasattr(layout, "children"):
        layout.children.append(fig)
        return layout

    return column(layout, fig, sizing_mode="stretch_width")


def create_legacy_backtest_chart(
    engine: Any,
    output_path: str | Path,
    strategy_name: str,
    platform: str,
    initial_cash: float,
    market_prices: Mapping[str, Sequence[tuple[Any, float]]] | None = None,
    user_probabilities: pd.Series | None = None,
    market_probabilities: pd.Series | None = None,
    outcomes: pd.Series | None = None,
    legacy_repo_path: str | Path | None = None,
    open_browser: bool = False,
    max_markets: int = 30,
    progress: bool = False,
) -> str:
    """
    Render a legacy-style interactive chart from a Nautilus backtest engine.

    Returns
    -------
    str
        Absolute output HTML path.
    """
    repo_path = resolve_legacy_plot_repo(legacy_repo_path)
    models_module, plotting_module = _load_legacy_modules(repo_path)

    account_report = _extract_account_report(engine)
    fills_report = engine.trader.generate_order_fills_report()
    positions_report = engine.trader.generate_positions_report()

    fills = _convert_fills(fills_report, models_module)
    snapshots = _build_portfolio_snapshots(models_module, account_report, fills)
    if not snapshots:
        raise ValueError("No portfolio snapshots were built from the account report.")

    normalized_market_prices = _normalize_market_prices(market_prices)
    if not normalized_market_prices:
        normalized_market_prices = _market_prices_from_fills(fills)

    market_pnls = _build_market_pnls(positions_report)
    metrics = _build_metrics(snapshots, initial_cash)

    result = models_module.BacktestResult(
        equity_curve=snapshots,
        fills=fills,
        metrics=metrics,
        strategy_name=strategy_name,
        platform=_platform_enum(models_module, platform),
        start_time=snapshots[0].timestamp,
        end_time=snapshots[-1].timestamp,
        initial_cash=float(initial_cash),
        final_equity=float(snapshots[-1].total_equity),
        num_markets_traded=len({fill.market_id for fill in fills}),
        num_markets_resolved=0,
        market_prices=normalized_market_prices,
        market_pnls=market_pnls,
    )

    output_abs = Path(output_path).expanduser().resolve()
    output_abs.parent.mkdir(parents=True, exist_ok=True)

    layout = plotting_module.plot(
        result,
        filename=str(output_abs),
        max_markets=max_markets,
        open_browser=open_browser,
        progress=progress,
    )

    brier_frame = prepare_cumulative_brier_advantage(
        user_probabilities=user_probabilities,
        market_probabilities=market_probabilities,
        outcomes=outcomes,
    )

    if not brier_frame.empty:
        layout = _append_brier_panel(layout, brier_frame)
        try:
            from bokeh.io import output_file
            from bokeh.io import save
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError("Bokeh is required for legacy chart rendering.") from exc

        output_file(str(output_abs), title=f"{strategy_name} legacy chart")
        save(layout, filename=str(output_abs), title=f"{strategy_name} legacy chart")

    return str(output_abs)
