# Prediction Market Backtests

This folder contains runnable backtest scripts for Kalshi and Polymarket.

## Purpose

Each script should only do orchestration:

- choose instruments/markets using adapter research helpers,
- instantiate strategy configs from `nautilus_trader.examples.strategies.prediction_market`,
- run backtests and report outputs.

## Current Scripts

- `kalshi_ema_crossover.py`
- `kalshi_breakout.py`
- `kalshi_ema_bars.py`
- `kalshi_spread_capture.py`
- `polymarket_ema_crossover.py`
- `polymarket_rsi_reversion.py`
- `polymarket_vwap_reversion.py`
- `polymarket_panic_fade.py`
- `polymarket_simple_quoter.py`
- `polymarket_spread_capture.py`
- `polymarket_deep_value_resolution_hold.py`

## Conventions

- Keep venue-specific data access in adapter research modules.
- Keep strategy classes in the shared prediction-market strategy package.
- Avoid helper duplication across scripts; prefer shared utilities.
