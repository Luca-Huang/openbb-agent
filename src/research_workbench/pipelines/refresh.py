from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from research_workbench.analysis.summary import build_summary_row
from research_workbench.data_sources.indicators import add_technical_indicators
from research_workbench.data_sources.longbridge import LOOKBACK_DAYS, get_provider

log = logging.getLogger("pipelines.refresh")


def fetch_all_history(watchlist: pd.DataFrame) -> pd.DataFrame:
    """Fetch and normalize history for all symbols in the watchlist."""
    if watchlist.empty:
        return pd.DataFrame()

    end_date = date.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    all_dfs: list[pd.DataFrame] = []

    for market, group in watchlist.groupby("market"):
        market = str(market).upper()
        try:
            provider = get_provider(market)
        except ValueError as exc:
            log.warning("Skipping market %s: %s", market, exc)
            continue

        for _, item in group.iterrows():
            symbol = str(item["symbol"])
            try:
                df = provider.fetch_history(symbol, start_date, end_date)
                if df.empty:
                    log.warning("  %s: no data returned", symbol)
                    continue
                df = add_technical_indicators(df)
                all_dfs.append(df)
                log.info("  %s: %d rows", symbol, len(df))
            except Exception as exc:  # noqa: BLE001
                log.error("  %s: fetch error - %s", symbol, exc)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def build_all_summaries(watchlist: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Build summary rows from normalized history plus Longbridge valuation fields."""
    if watchlist.empty or history.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, item in watchlist.iterrows():
        symbol = str(item["symbol"])
        market = str(item.get("market", "CN")).upper()
        if history[history["symbol"] == symbol].empty:
            log.warning("  %s: no history, skipping summary", symbol)
            continue

        try:
            fund_data = get_provider(market).fetch_fundamentals(symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s: fundamentals fetch error - %s", symbol, exc)
            fund_data = {}

        row = build_summary_row(item, history, fund_data)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
