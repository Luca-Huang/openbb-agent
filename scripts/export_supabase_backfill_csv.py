#!/usr/bin/env python3
"""Generate CSV files for manual Supabase import.

When network constraints prevent running the automated backfill script,
this helper reads the local `openbb_outputs` CSVs and emits two new files:

* `supabase_equity_metrics_backfill.csv`
* `supabase_equity_history_backfill.csv`

Each contains the exact columns expected by the Supabase tables, so you can
upload them through the Supabase UI (`Import data from CSV`) without running
thousands of API calls locally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "openbb_outputs"
SUMMARY_CSV = DATA_DIR / "three_month_summary.csv"
HISTORY_CSV = DATA_DIR / "three_month_close_history.csv"
SUMMARY_OUTPUT = DATA_DIR / "supabase_equity_metrics_backfill.csv"
HISTORY_OUTPUT = DATA_DIR / "supabase_equity_history_backfill.csv"

# Columns expected by Supabase `equity_metrics`
SUMMARY_COLUMNS = [
    "symbol",
    "name_en",
    "name_cn",
    "market",
    "theme_category",
    "investment_reason",
    "value_score",
    "value_score_tier",
    "entry_recommendation",
    "entry_reason",
    "tier_reason",
    "pe_percentile_5y",
    "ps_percentile_5y",
    "peg_ratio",
    "fcf_yield",
    "end_pe",
    "current_ps",
    "forward_pe",
    "pe_coverage_years",
    "ps_coverage_years",
    "refresh_interval_days",
    "next_refresh_date",
    "pct_change",
    "support_level_primary",
    "support_level_secondary",
    "pct_change_7d",
    "pct_change_30d",
]

# Columns expected by Supabase `equity_metrics_history`
HISTORY_COLUMNS = [
    "symbol",
    "date",
    "name_en",
    "name_cn",
    "market",
    "open",
    "high",
    "low",
    "close",
    "close_norm",
    "close_percentile",
    "support_level_primary",
    "support_level_secondary",
    "ttm_eps",
    "pe",
    "ps_ratio",
    "ma50",
    "ma200",
    "rsi14",
    "fib_38_2",
    "fib_50",
    "fib_61_8",
    "volume",
    "volume_ma20",
    "volume_spike_ratio",
    "obv",
    "vpt",
    "vwap",
    "ad_line",
]


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a dataframe containing only the requested columns, creating blanks if needed."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = pd.NA
    return df[columns]


def export_summary() -> None:
    summary_df = pd.read_csv(SUMMARY_CSV)
    summary_df = ensure_columns(summary_df, SUMMARY_COLUMNS)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    print(f"[Summary] Saved {len(summary_df)} rows -> {SUMMARY_OUTPUT}")


def export_history() -> None:
    history_df = pd.read_csv(HISTORY_CSV)
    history_df.rename(columns={"support_level": "support_level_primary"}, inplace=True)
    history_df = ensure_columns(history_df, HISTORY_COLUMNS)
    # Supabase expects `as_of_date` instead of `date`; rename here for clarity.
    history_df = history_df.rename(columns={"date": "as_of_date"})
    history_df.to_csv(HISTORY_OUTPUT, index=False)
    print(f"[History] Saved {len(history_df)} rows -> {HISTORY_OUTPUT}")


def main() -> None:
    export_summary()
    export_history()


if __name__ == "__main__":
    main()
