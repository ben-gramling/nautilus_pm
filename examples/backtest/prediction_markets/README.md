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
- `kalshi_rsi_reversion.py`
- `kalshi_panic_fade.py`
- `kalshi_ema_bars.py`
- `kalshi_spread_capture.py`
- `polymarket_ema_crossover.py`
- `polymarket_rsi_reversion.py`
- `polymarket_vwap_reversion.py`
- `polymarket_panic_fade.py`
- `polymarket_simple_quoter.py`
- `polymarket_spread_capture.py`
- `polymarket_deep_value_resolution_hold.py`

## Single-Market Comparison Set

Use these scripts to compare strategies on shared markets:

- Kalshi default market: `KXNEXTIRANLEADER-45JAN01-MKHA`
- Polymarket default market:
  `will-gavin-newsom-win-the-2028-democratic-presidential-nomination-568`

Kalshi:

- `kalshi_spread_capture.py`
- `kalshi_ema_crossover.py`
- `kalshi_breakout.py`
- `kalshi_rsi_reversion.py`
- `kalshi_panic_fade.py`

Polymarket:

- `polymarket_spread_capture.py`
- `polymarket_ema_crossover.py`
- `polymarket_rsi_reversion.py`
- `polymarket_vwap_reversion.py`
- `polymarket_panic_fade.py`
- `polymarket_deep_value_resolution_hold.py`
- `polymarket_simple_quoter.py`

Default chart readability settings for this comparison set:

- Kalshi and Polymarket chart prices are full-density by default (no resample).
- Set `CHART_RESAMPLE_RULE` if you want smoother chart lines for debugging.
- Polymarket strategy defaults are calibrated for the Newsom nominee market range so
  charts produce non-flat equity/PnL/allocation by default.

## Conventions

- Keep venue-specific data access in adapter research modules.
- Keep strategy classes in the shared prediction-market strategy package.
- Avoid helper duplication across scripts; prefer shared utilities.
