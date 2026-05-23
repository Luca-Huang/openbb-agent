# Indicator Reference

This document describes the indicator fields consumed by the research workbench.

## Source Locations

| Field | Code |
|---|---|
| `ma20`, `ma50`, `ma200` | `src/research_workbench/data_sources/indicators.py` |
| `rsi14` | `src/research_workbench/data_sources/indicators.py` |
| `support_level_primary`, `support_level_secondary` | `src/research_workbench/data_sources/indicators.py` |
| `volume_ma20`, `volume_spike_ratio` | `src/research_workbench/data_sources/indicators.py` |
| `high_20d`, `atr14`, `drawdown_60d`, `dollar_volume20` | `src/research_workbench/signal_engine/radar.py` |

## Core Technical Fields

- `ma20`: 20-session simple moving average of close.
- `ma50`: 50-session simple moving average of close.
- `ma200`: 200-session simple moving average of close.
- `rsi14`: 14-session relative strength index.
- `support_level_primary`: 20-session rolling minimum close.
- `support_level_secondary`: `support_level_primary * 1.1`.
- `volume_spike_ratio`: current volume divided by 20-session average volume.
- `high_20d`: prior 20-session high, shifted by one bar.
- `atr14`: 14-session average true range.
- `drawdown_60d`: distance from the rolling 60-session close peak.

## Trigger Rules

- `pullback`: close is near MA50, above primary support, and volume confirms.
- `breakout`: close breaks the prior 20-session high with elevated volume.

## Risk And Exit Fields

- `risk_unit`: entry price minus stop price.
- `take_profit_1`: entry plus 1R.
- `take_profit_2`: entry plus 2R.
- `trailing_stop`: trailing reference from MA20 and ATR-based logic.

These fields are planning references only. They do not place orders.
