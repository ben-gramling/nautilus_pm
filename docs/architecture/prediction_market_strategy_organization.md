# Prediction Market Strategy Organization

## Goal

Scale to dozens of prediction-market strategies without duplicating helper logic
or mixing venue adapters with strategy code.

## External Patterns Benchmarked

- QuantConnect LEAN Algorithm Framework:
  separates alpha, portfolio construction, risk management, and execution.
- Freqtrade:
  keeps strategy classes as reusable units while data/exchange plumbing lives
  in the framework/runtime.
- Backtrader/Zipline:
  event-driven strategy classes with engine/data handling outside strategy modules.

## Repository Structure

- Adapter layer:
  `nautilus_trader.adapters.{kalshi,polymarket}.research`
  contains market discovery + historical data loading.
- Strategy layer:
  `nautilus_trader.examples.strategies.prediction_market`
  contains reusable, venue-agnostic strategy classes/configs.
- Backtest orchestration:
  `examples/backtest/prediction_markets`
  contains scripts that select markets and run strategies.

## Implementation Rules

- No market discovery helpers in strategy modules.
- No venue-specific API logic in strategy modules.
- Shared order lifecycle helpers belong in `prediction_market/core.py`.
- New strategies should expose config + strategy classes and be exported in
  `prediction_market/__init__.py`.

## New Strategy Modules Added

- `ema_crossover.py`
- `breakout.py`
- `rsi_reversion.py`
- `vwap_reversion.py`
- `panic_fade.py`
