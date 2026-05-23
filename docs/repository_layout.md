# Repository Layout

The workbench is split by responsibility so data acquisition, analysis, and output code can change independently.

## Runtime Layers

| Layer | Package | Responsibility |
|---|---|---|
| Data sources | `src/research_workbench/data_sources/` | Longbridge CLI adapter and source-normalized market data |
| Analysis | `src/research_workbench/analysis/`, `src/research_workbench/signal_engine/`, `src/research_workbench/validation/` | Summary scoring, signal generation, holdings analysis, validation, and backtests |
| Outputs | `src/research_workbench/outputs/` | Local file loading and saving for generated datasets |
| Pipelines | `src/research_workbench/pipelines/` | Orchestration that connects data sources, analysis, and outputs |
| UI | `app.py`, `src/research_workbench/ui/` | Streamlit presentation and UI-specific formatting |

## Entry Points

- `scripts/refresh.py`: command-line refresh entry point; delegates work to `pipelines.refresh`.
- `scripts/send_radar_email.py`: email output entry point; reads generated outputs and performs optional holdings analysis.
- `scripts/backtest_local_strategy.py`: ad hoc backtest entry point; uses the shared Longbridge and indicator pipeline.
- `app.py`: Streamlit UI entry point; reads output files and renders views.

## Boundary Rules

- Data-source modules must not import UI, output, or signal rendering code.
- Analysis modules should consume normalized DataFrames and plain dictionaries, not call CLI tools directly.
- Output modules should only load or save artifacts; they should not fetch market data or compute signals.
- UI code should call analysis/output APIs and avoid owning business logic.
