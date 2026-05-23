# Data Charter

The active data contract is intentionally narrow:

- Market data comes from Longbridge CLI.
- Watchlist and manual events are local user-maintained inputs.
- Generated CSVs under `outputs/research_data/` are derived artifacts.
- UI and email code must not fetch market data directly.

## Generated Artifacts

| Artifact | Producer | Consumer |
|---|---|---|
| `three_month_close_history.csv` | `scripts/refresh.py` | UI, validation, signal engine |
| `three_month_summary.csv` | `scripts/refresh.py` | UI, signal engine |
| `signals_snapshot.csv` | `scripts/refresh.py`, `app.py` | change detection, email |

## Quality Rules

- Prefer empty outputs over inferred values when source data is missing.
- Preserve source symbols in Longbridge `<CODE>.<MARKET>` format.
- Store normalized numeric fields as numeric CSV columns, not formatted strings.
