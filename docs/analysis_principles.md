# Analysis Principles

The project is a stock research workbench with a deterministic pipeline:

1. Fetch normalized market data from Longbridge CLI.
2. Add technical features and valuation fields.
3. Build rule-based signals, validation samples, and action guidance.
4. Persist generated outputs for the UI and email layer.

LLM output, when added, should remain an explanation layer. It should not be the source of prices, financial metrics, signal state, stops, or position sizing.

## Data Boundary

- The data-source layer is responsible for CLI/API access and schema normalization.
- The analysis layer consumes DataFrames and plain dictionaries only.
- The output layer persists artifacts and does not compute investment logic.
- The UI layer renders already-computed results and should avoid owning business rules.

## Risk Controls

- Every signal must be traceable to price, volume, support, trend, valuation, and event inputs.
- Missing data should produce an empty or degraded signal rather than a fabricated conclusion.
- Generated action guidance is not automatic order execution.
