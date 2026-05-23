# Investment Framework

The workbench supports a rules-first research process. It does not place orders.

## Inputs

- Longbridge OHLCV history.
- Longbridge static and valuation fields.
- User watchlist metadata.
- User-maintained manual events.
- Optional holdings history for position review.

## Analysis Steps

1. Normalize history and add technical indicators.
2. Build a lightweight valuation summary.
3. Detect pullback and breakout conditions.
4. Compute risk, exit references, and action guidance.
5. Validate historical trigger samples.

## Output Principles

- Each recommendation must expose the underlying price, trend, volume, valuation, and event context.
- Stops and take-profit levels are planning references, not order instructions.
- A missing data field should reduce confidence or skip a signal.
