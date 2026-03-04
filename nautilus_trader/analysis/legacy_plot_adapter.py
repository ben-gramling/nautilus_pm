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

import numpy as np
import pandas as pd

from nautilus_trader.analysis.reporter import ReportProvider


DEFAULT_LEGACY_CLONE = "https://github.com/evan-kolberg/prediction-market-backtesting.git"
DEFAULT_LEGACY_WORKTREE = Path("/tmp/prediction-market-backtesting-legacy")  # noqa: S108
GIT_BIN = shutil.which("git") or "git"


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
        proc = subprocess.run(  # noqa: S603
            [GIT_BIN, "-C", str(repo_path), "rev-parse", "--verify", "legacy"],
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
        subprocess.run(  # noqa: S603
            [
                GIT_BIN,
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
            subprocess.run(  # noqa: S603
                [
                    GIT_BIN,
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
            subprocess.run(  # noqa: S603
                [
                    GIT_BIN,
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
        subprocess.run(  # noqa: S603
            [
                GIT_BIN,
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


def _iter_layout_nodes(node: Any):
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return

    for child in children:
        obj = child[0] if isinstance(child, tuple) else child
        if obj is not None:
            yield from _iter_layout_nodes(obj)


def _iter_figures(layout: Any):
    for node in _iter_layout_nodes(layout):
        if hasattr(node, "renderers") and hasattr(node, "title") and hasattr(node, "yaxis"):
            yield node


def _remove_data_banner(layout: Any) -> Any:
    if not hasattr(layout, "children") or not layout.children:
        return layout

    first = layout.children[0]
    text = getattr(first, "text", "")
    if isinstance(text, str) and "<b>Data:</b>" in text:
        layout.children = list(layout.children[1:])
    return layout


def _extract_equity_timeline(layout: Any) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []

    for fig in _iter_figures(layout):
        for renderer in getattr(fig, "renderers", []):
            source = getattr(renderer, "data_source", None)
            data = getattr(source, "data", None)
            if not isinstance(data, dict):
                continue
            if "datetime" not in data or "index" not in data:
                continue

            if "equity_dollar" in data:
                equity_values = data["equity_dollar"]
            elif "equity" in data:
                equity_values = data["equity"]
            elif "cash" in data and "pos_value" in data:
                equity_values = np.asarray(data["cash"], dtype=float) + np.asarray(
                    data["pos_value"],
                    dtype=float,
                )
            else:
                continue

            frame = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(data["datetime"], errors="coerce"),
                    "index": pd.to_numeric(pd.Series(data["index"]), errors="coerce"),
                    "equity": pd.to_numeric(pd.Series(equity_values), errors="coerce"),
                },
            ).dropna()

            if frame.empty:
                continue

            frame = frame.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
            candidates.append(frame)

    if not candidates:
        return pd.DataFrame(columns=["datetime", "index", "equity"])

    return max(candidates, key=len)


def _build_daily_performance(
    equity_timeline: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    if equity_timeline.empty:
        return pd.DataFrame()

    series = (
        equity_timeline.set_index("datetime")["equity"]
        .sort_index()
        .astype(float)
    )
    daily_close = series.resample("1D").last().dropna()
    if daily_close.empty:
        return pd.DataFrame()

    daily_pnl = daily_close.diff()
    if len(daily_close) > 0:
        daily_pnl.iloc[0] = float(daily_close.iloc[0]) - float(initial_cash)

    daily_returns = daily_close.pct_change()
    if len(daily_close) > 0 and initial_cash:
        daily_returns.iloc[0] = (float(daily_close.iloc[0]) - float(initial_cash)) / float(
            initial_cash,
        )
    daily_returns = daily_returns.fillna(0.0)

    lookup = equity_timeline[["datetime", "index"]].sort_values("datetime")
    aligned = pd.merge_asof(
        pd.DataFrame({"datetime": daily_close.index}).sort_values("datetime"),
        lookup,
        on="datetime",
        direction="backward",
    )
    aligned["index"] = aligned["index"].ffill().bfill().fillna(0.0)

    return pd.DataFrame(
        {
            "datetime": daily_close.index.to_pydatetime(),
            "x": aligned["index"].to_numpy(dtype=float),
            "pnl": daily_pnl.to_numpy(dtype=float),
            "ret": daily_returns.to_numpy(dtype=float),
        },
    )


def _rebuild_daily_pnl_panel(layout: Any, daily: pd.DataFrame) -> None:
    if daily.empty:
        return

    try:
        from bokeh.models import ColumnDataSource
        from bokeh.models import HoverTool
        from bokeh.models import NumeralTickFormatter
    except ImportError:
        return

    target = None
    for fig in _iter_figures(layout):
        labels = [str(axis.axis_label or "") for axis in getattr(fig, "yaxis", [])]
        if any("periodic" in label.lower() for label in labels):
            target = fig
            break

    if target is None:
        return

    x_vals = daily["x"].to_numpy(dtype=float)
    diffs = pd.Series(x_vals).sort_values().diff().dropna()
    width = max(1.0, float(diffs.median()) * 0.8) if not diffs.empty else 1.0

    source = ColumnDataSource(
        {
            "x": x_vals,
            "pnl": daily["pnl"].to_numpy(dtype=float),
            "pnl_pos": np.maximum(daily["pnl"].to_numpy(dtype=float), 0.0),
            "pnl_neg": np.minimum(daily["pnl"].to_numpy(dtype=float), 0.0),
            "datetime": pd.to_datetime(daily["datetime"]).to_numpy(dtype="datetime64[ns]"),
        },
    )

    if target.yaxis:
        target.yaxis[0].axis_label = "P&L (Daily)"
    target.renderers = [r for r in target.renderers if not hasattr(r, "data_source")]
    target.tools = [tool for tool in target.tools if tool.__class__.__name__ != "HoverTool"]

    pos = target.vbar(
        x="x",
        top="pnl_pos",
        source=source,
        width=width,
        color="#2ecc71",
        alpha=0.75,
        legend_label="Gain",
    )
    neg = target.vbar(
        x="x",
        top="pnl_neg",
        source=source,
        width=width,
        color="#e74c3c",
        alpha=0.75,
        legend_label="Loss",
    )

    target.add_tools(
        HoverTool(
            renderers=[pos, neg],
            formatters={"@datetime": "datetime"},
            tooltips=[
                ("Date", "@datetime{%F}"),
                ("P&L", "@pnl{$0,0.00}"),
            ],
            mode="vline",
        ),
    )
    target.yaxis.formatter = NumeralTickFormatter(format="$ 0,0")


def _replace_monthly_with_daily_returns(layout: Any, daily: pd.DataFrame) -> Any:
    if daily.empty or not hasattr(layout, "children"):
        return layout

    try:
        from bokeh.models import ColumnDataSource
        from bokeh.models import HoverTool
        from bokeh.models import NumeralTickFormatter
        from bokeh.models import Span
        from bokeh.plotting import figure
    except ImportError:
        return layout

    target_index: int | None = None
    for idx, child in enumerate(layout.children):
        if hasattr(child, "yaxis"):
            labels = [axis.axis_label for axis in getattr(child, "yaxis", [])]
            if any(label == "Monthly Returns" for label in labels):
                target_index = idx
                break

    if target_index is None:
        return layout

    source = ColumnDataSource(
        {
            "datetime": pd.to_datetime(daily["datetime"]).to_numpy(dtype="datetime64[ns]"),
            "ret": daily["ret"].to_numpy(dtype=float),
            "ret_pos": np.maximum(daily["ret"].to_numpy(dtype=float), 0.0),
            "ret_neg": np.minimum(daily["ret"].to_numpy(dtype=float), 0.0),
        },
    )

    fig = figure(
        title="Daily Returns (%)",
        x_axis_type="datetime",
        height=130,
        tools="xpan,xwheel_zoom,box_zoom,undo,redo,reset,save",
        active_drag="xpan",
        active_scroll="xwheel_zoom",
        sizing_mode="stretch_width",
        toolbar_location="right",
    )
    fig.add_layout(
        Span(
            location=0.0,
            dimension="width",
            line_color="#666666",
            line_dash="dashed",
            line_width=1,
        ),
    )

    day_ms = 24 * 60 * 60 * 1000
    pos = fig.vbar(
        x="datetime",
        top="ret_pos",
        source=source,
        width=day_ms * 0.8,
        color="#2ecc71",
        alpha=0.75,
        legend_label="Positive",
    )
    neg = fig.vbar(
        x="datetime",
        top="ret_neg",
        source=source,
        width=day_ms * 0.8,
        color="#e74c3c",
        alpha=0.75,
        legend_label="Negative",
    )

    fig.add_tools(
        HoverTool(
            renderers=[pos, neg],
            formatters={"@datetime": "datetime"},
            tooltips=[
                ("Date", "@datetime{%F}"),
                ("Return", "@ret{+0.00%}"),
            ],
            mode="vline",
        ),
    )

    fig.yaxis.axis_label = "Daily Return"
    fig.yaxis.formatter = NumeralTickFormatter(format="+0.0%")
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"

    children = list(layout.children)
    children[target_index] = fig
    layout.children = children
    return layout


def _remove_periodic_pnl_panel(layout: Any) -> Any:
    """
    Remove the periodic/daily P&L panel.

    Daily returns already capture day-over-day performance and this avoids
    duplicated information.
    """
    keywords = ("periodic", "p&l (daily)")

    for node in _iter_layout_nodes(layout):
        children = getattr(node, "children", None)
        if children is None:
            continue

        new_children: list[Any] = []
        changed = False
        for child in children:
            obj = child[0] if isinstance(child, tuple) else child
            labels = [str(axis.axis_label or "") for axis in getattr(obj, "yaxis", [])]
            lower_labels = " ".join(labels).lower()
            if any(keyword in lower_labels for keyword in keywords):
                changed = True
                continue
            new_children.append(child)

        if changed:
            node.children = new_children

    return layout


def _legend_item_label_text(item: Any) -> str:
    label = getattr(item, "label", None)
    if isinstance(label, dict):
        return str(label.get("value", ""))
    return str(label)


def _remove_yes_price_profitability_legend_items(fig: Any) -> set[Any]:
    renderers_to_drop: set[Any] = set()

    for legend in getattr(fig, "legend", []):
        kept_items = []
        for item in list(getattr(legend, "items", [])):
            lower = _legend_item_label_text(item).lower()
            if "profitable" in lower or "losing" in lower:
                for renderer in getattr(item, "renderers", []):
                    renderers_to_drop.add(renderer)
                continue
            kept_items.append(item)
        legend.items = kept_items

    return renderers_to_drop


def _remove_yes_price_profitability_connectors(layout: Any) -> None:
    """
    Remove profitable/losing connector overlays from the YES price panel.
    """
    yes_fig = None
    for fig in _iter_figures(layout):
        labels = [str(axis.axis_label or "") for axis in getattr(fig, "yaxis", [])]
        if any(label == "YES Price" for label in labels):
            yes_fig = fig
            break

    if yes_fig is None:
        return

    renderers_to_drop = _remove_yes_price_profitability_legend_items(yes_fig)

    # Drop any unlabeled multiline overlays as a safety net.
    for renderer in getattr(yes_fig, "renderers", []):
        glyph = getattr(renderer, "glyph", None)
        if glyph is not None and glyph.__class__.__name__ == "MultiLine":
            renderers_to_drop.add(renderer)

    if renderers_to_drop:
        yes_fig.renderers = [r for r in yes_fig.renderers if r not in renderers_to_drop]


def _focus_allocation_panel(layout: Any) -> None:
    try:
        from bokeh.models import Range1d
    except ImportError:
        return

    for fig in _iter_figures(layout):
        labels = [axis.axis_label for axis in getattr(fig, "yaxis", [])]
        if "Allocation" not in labels:
            continue

        glyph_renderers = [r for r in fig.renderers if hasattr(r, "data_source")]
        if not glyph_renderers:
            continue

        source = glyph_renderers[0].data_source
        data = getattr(source, "data", {})
        alloc_cols = [k for k in data if str(k).startswith("alloc_")]
        non_cash = [k for k in alloc_cols if "Cash" not in str(k)]
        if not non_cash:
            continue

        stacked = np.zeros(len(data[non_cash[0]]), dtype=float)
        for col in non_cash:
            stacked += np.nan_to_num(np.asarray(data[col], dtype=float))

        peak = float(np.nanmax(stacked)) if len(stacked) else 0.0
        upper = min(1.0, max(0.05, peak * 1.3))
        fig.y_range = Range1d(0.0, upper)
        fig.yaxis[0].axis_label = "Market Allocation (ex-cash)"

        # Hide the grey cash stack so market allocation is visible.
        if glyph_renderers:
            glyph_renderers[-1].visible = False
        break


def _apply_layout_overrides(layout: Any, initial_cash: float) -> Any:
    layout = _remove_data_banner(layout)
    _focus_allocation_panel(layout)
    _remove_yes_price_profitability_connectors(layout)

    equity_timeline = _extract_equity_timeline(layout)
    daily = _build_daily_performance(equity_timeline, initial_cash=initial_cash)
    layout = _remove_periodic_pnl_panel(layout)
    layout = _replace_monthly_with_daily_returns(layout, daily)

    return layout


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

    fills = _convert_fills(fills_report, models_module)
    snapshots = _build_portfolio_snapshots(models_module, account_report, fills)
    if not snapshots:
        raise ValueError("No portfolio snapshots were built from the account report.")

    normalized_market_prices = _normalize_market_prices(market_prices)
    if not normalized_market_prices:
        normalized_market_prices = _market_prices_from_fills(fills)

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
        # Leave this empty so the legacy P/L panel falls back to per-fill
        # markers instead of one terminal point per market.
        market_pnls={},
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
    layout = _apply_layout_overrides(layout, initial_cash=float(initial_cash))

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
