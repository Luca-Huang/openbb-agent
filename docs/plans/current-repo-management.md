# Current Repo Management Plan

## Module Boundaries

- `data_sources/`: market-data acquisition and normalization.
- `analysis/`: summary scoring and pure transforms.
- `signal_engine/`: rule-based signal, risk, holdings, and backtest logic.
- `validation/`: replay and validation reporting.
- `outputs/`: generated artifact loading and saving.
- `pipelines/`: orchestration across layers.
- `app.py`: UI entry point only.

## Active Data Source

Longbridge CLI is the active source. Other sources should not be added directly to scripts or UI code.

## Generated Outputs

Generated CSVs live under `outputs/research_data/`.

## Review Checklist

- New market access code belongs in `data_sources/`.
- New scoring logic belongs in `analysis/` or `signal_engine/`.
- New file or publishing logic belongs in `outputs/`.
- Scripts should stay thin and delegate to package modules.
