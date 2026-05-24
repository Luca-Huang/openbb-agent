# Data Charter

The active data contract is intentionally narrow:

- Market data comes from Longbridge CLI (`longbridge`) and AKShare via
  AKTools HTTP (`python -m aktools`). See `docs/source_matrix.md`.
- Watchlist and manual events are local user-maintained inputs.
- Generated CSVs under `outputs/research_data/` are derived artifacts.
- UI and email code must not fetch market data directly — they read CSVs.

## Generated Artifacts

| Artifact | Producer | Consumer | Cadence |
|---|---|---|---|
| `three_month_close_history.csv` | `scripts/refresh.py` | UI, validation, signal engine | daily |
| `three_month_summary.csv` | `scripts/refresh.py` | UI, signal engine | daily |
| `signals_snapshot.csv` | `scripts/refresh.py`, `app.py` | change detection, email | daily |
| `annual_financials.csv` | `scripts/refresh.py` | `analysis.summary` scoring | daily (data changes quarterly) |
| `quarterly_financials.csv` | `scripts/refresh.py` | ad-hoc cadence analysis | daily |
| `valuation_history.csv` | `scripts/refresh.py` | `score_hist_valuation`, UI | daily |
| `earnings_express.csv` | `scripts/refresh.py` | ad-hoc | daily |
| `northbound_holdings.csv` | `scripts/refresh.py` | sentiment signal (TBD) | daily |
| `capital_flow_history.csv` | `scripts/refresh.py` | sentiment signal (TBD) | one row per symbol per refresh — accumulates |
| `auto_events.csv` | `scripts/refresh.py` | event-risk score, signal engine | daily |

## Schema Discipline

All artifacts must match the corresponding `ColumnDef` list in
`research_store/schema.py`:

- `HISTORY_SCHEMA` ↔ `three_month_close_history.csv`
- `SUMMARY_SCHEMA` ↔ `three_month_summary.csv`
- `SIGNAL_SCHEMA` ↔ `signals_snapshot.csv`
- `FINANCIAL_SCHEMA` ↔ `annual_financials.csv` / `quarterly_financials.csv`
- `EARNINGS_EXPRESS_SCHEMA` ↔ `earnings_express.csv`
- `VALUATION_HISTORY_SCHEMA` ↔ `valuation_history.csv`
- `NORTHBOUND_HOLDING_SCHEMA` ↔ `northbound_holdings.csv`
- `CAPITAL_FLOW_SCHEMA` ↔ `capital_flow_history.csv`
- `EVENT_SCHEMA` ↔ `manual_events.csv` + `auto_events.csv`

## Quality Rules

- Prefer empty outputs over inferred values when source data is missing.
- Preserve source symbols in Longbridge `<CODE>.<MARKET>` format
  (e.g. `002624.SZ`, `700.HK`, `TSLA.US`).
- Store normalized numeric fields as numeric CSV columns, not formatted
  strings (no `万`/`亿` suffixes, no percent signs).
- Per-symbol fetchers that wrap market-wide tables must offer a batched
  `market_df` parameter; the pipeline always calls the batch version once
  and slices per symbol.
- Auto-sourced events shadow manual events on duplicate
  (symbol, event_date, event_type) — manual wins. Use auto events as the
  default, manual to override.
- NaN ≠ 0 in raw fetcher output. Score functions explicitly treat missing
  data as "no signal" (zero score), not as zero value, to avoid penalizing
  symbols that simply don't disclose a metric.

## Operational Notes

- AKTools needs to be running during `scripts/refresh.py` for the CN-deep
  enrichment stage to populate. The pipeline degrades gracefully when it
  isn't — Longbridge-only data still flows, but the five enrichment-based
  scores fall to zero.
- Domestic-host requests (eastmoney / sina) must bypass any HTTP proxy;
  AKShare CN is the only fetcher that touches those hosts and the
  responsibility rests in the AKTools server process.
- The push2 eastmoney endpoints are intermittently flaky from mainland
  networks (see git history); per-symbol fetchers use a 3-attempt
  exponential backoff and the pipeline tolerates per-symbol failures.
