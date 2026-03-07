#!/usr/bin/env python3
"""Backtest runner — interactive strategy menu.

Discovers strategies in examples/backtest/ (and any directories listed in
the EXTRA_STRATEGIES_DIRS environment variable) that expose:

    NAME        str   — display name shown in the menu
    DESCRIPTION str   — one-line description shown in the menu
    run()       async — entry point called when the strategy is selected

Run via:
    .venv/bin/python main.py
    make backtest
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path: wheel takes priority for compiled extensions, but we extend
# nautilus_trader.adapters.__path__ so the local kalshi adapter is findable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_VENV_SITE = _REPO_ROOT / ".venv" / "lib" / "python3.13" / "site-packages"

if str(_VENV_SITE) in sys.path:
    sys.path.remove(str(_VENV_SITE))
sys.path.insert(0, str(_VENV_SITE))

if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))

import nautilus_trader.adapters as _nt_adapters  # noqa: E402

_LOCAL_ADAPTERS = _REPO_ROOT / "nautilus_trader" / "adapters"
if str(_LOCAL_ADAPTERS) not in _nt_adapters.__path__:
    _nt_adapters.__path__.insert(0, str(_LOCAL_ADAPTERS))
# ---------------------------------------------------------------------------

# Directories to scan. EXTRA_STRATEGIES_DIRS can be a colon-separated list of
# additional paths (e.g. strategies from a sibling repo).
STRATEGIES_DIRS: list[Path] = [_REPO_ROOT / "examples" / "backtest"]

for _extra in os.environ.get("EXTRA_STRATEGIES_DIRS", "").split(":"):
    _extra = _extra.strip()
    if _extra:
        STRATEGIES_DIRS.append(Path(_extra))

# ---------------------------------------------------------------------------

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
RESET = "\033[0m"


def _load_strategy(path: Path) -> dict | None:
    """Load a strategy module by file path and return its menu entry, or None."""
    # Ensure the file's directory is importable (for sibling imports like
    # `from kalshi_spread_strategy import ...`).
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        print(f"  {DIM}Warning: could not import {path.name}: {exc}{RESET}")
        return None
    if not hasattr(mod, "run"):
        return None
    return {
        "name": getattr(mod, "NAME", path.stem),
        "description": getattr(mod, "DESCRIPTION", ""),
        "run": mod.run,
    }


def discover() -> list[dict]:
    """Scan strategy directories for modules that expose NAME, DESCRIPTION, and run()."""
    found = []
    for strats_dir in STRATEGIES_DIRS:
        if not strats_dir.exists():
            continue
        for path in sorted(strats_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            strategy = _load_strategy(path)
            if strategy:
                found.append(strategy)
    return found


def show_menu(strategies: list[dict]) -> int:
    """Print numbered menu and return the chosen index (0-based), or -1 to exit."""
    print(f"\n{BOLD}Select a backtest:{RESET}\n")
    for i, s in enumerate(strategies, 1):
        desc = f" {DIM}— {s['description']}{RESET}" if s["description"] else ""
        print(f"  {CYAN}{i}{RESET}. {s['name']}{desc}")
    print(f"\n  {DIM}0. Exit{RESET}\n")
    try:
        raw = input("Enter number: ").strip()
    except (EOFError, KeyboardInterrupt):
        return -1
    try:
        choice = int(raw)
    except ValueError:
        print("Invalid input.")
        return -1
    if choice == 0:
        return -1
    if choice < 1 or choice > len(strategies):
        print("Invalid choice.")
        return -1
    return choice - 1


def main() -> None:
    strategies = discover()
    if not strategies:
        dirs = ", ".join(str(d) for d in STRATEGIES_DIRS)
        print(f"No strategies found in: {dirs}")
        sys.exit(1)
    idx = show_menu(strategies)
    if idx == -1:
        print("Exiting.")
        sys.exit(0)
    chosen = strategies[idx]
    print(f"\n{BOLD}Running: {chosen['name']}{RESET}\n")
    asyncio.run(chosen["run"]())


if __name__ == "__main__":
    main()
