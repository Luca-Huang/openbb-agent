"""Unified data refresh entry-point for the research workbench.

Usage:
    python scripts/refresh.py              # fetch + build signals for all watchlist symbols
    python scripts/refresh.py --only history   # only refresh history data
    python scripts/refresh.py --no-fetch   # skip network fetch, use cached CSV only
    python scripts/refresh.py --dry-run    # preview what would be refreshed

This script orchestrates the full data pipeline:
1. Load watchlist (supports CN / HK / US markets)
2. Fetch / refresh history data via providers (AKShare for CN, yfinance for HK/US)
3. Compute technical indicators
4. Fetch / refresh fundamentals and build summary
5. Run signal engine to produce structured signals
6. Persist signals snapshot for daily change detection
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from research_workbench.config import default_settings
from research_workbench.ingestion.watchlist import load_watchlist, watchlist_symbols
from research_workbench.ingestion.providers import (
    LOOKBACK_DAYS,
    add_technical_indicators,
    get_provider,
    score_fundamentals,
)
from research_workbench.research_store.files import (
    load_history,
    load_manual_events,
    load_summary,
)
from research_workbench.signal_engine.radar import DEFAULT_RADAR_CONFIG, build_current_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("refresh")

SETTINGS = default_settings(ROOT)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_all_history(watchlist: pd.DataFrame, force: bool = True) -> pd.DataFrame:
    """Fetch history for all symbols in watchlist, grouped by market."""
    if watchlist.empty:
        return pd.DataFrame()

    end_date = date.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    all_dfs: list[pd.DataFrame] = []
    for market, group in watchlist.groupby("market"):
        market = str(market).upper()
        try:
            provider = get_provider(market)
        except ValueError as e:
            log.warning("Skipping market %s: %s", market, e)
            continue

        for _, item in group.iterrows():
            symbol = str(item["symbol"])
            try:
                df = provider.fetch_history(symbol, start_date, end_date)
                if not df.empty:
                    df = add_technical_indicators(df)
                    all_dfs.append(df)
                    log.info("  ✓ %s: %d rows", symbol, len(df))
                else:
                    log.warning("  ✗ %s: no data returned", symbol)
            except Exception as e:
                log.error("  ✗ %s: fetch error — %s", symbol, e)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def fetch_all_summaries(watchlist: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Build summary rows (fundamentals + scores) for each symbol."""
    if watchlist.empty or history.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, item in watchlist.iterrows():
        symbol = str(item["symbol"])
        market = str(item.get("market", "CN")).upper()

        sym_hist = history[history["symbol"] == symbol]
        if sym_hist.empty:
            log.warning("  ✗ %s: no history, skipping summary", symbol)
            continue

        latest = sym_hist.sort_values("date").iloc[-1]

        try:
            provider = get_provider(market)
            fund_data = provider.fetch_fundamentals(symbol)
        except Exception as e:
            log.warning("  ✗ %s: fundamentals fetch error — %s", symbol, e)
            fund_data = {}

        scores = score_fundamentals(fund_data)

        latest_close = float(latest["close"])
        start_close = float(sym_hist.sort_values("date").iloc[0]["close"])
        pct_change = (latest_close - start_close) / start_close if start_close > 0 else 0

        value_score = sum(scores.values())

        rows.append({
            "symbol": symbol,
            "name": item.get("name", ""),
            "name_cn": item.get("name", ""),
            "name_en": "",
            "market": market,
            "theme_category": item.get("sector", ""),
            "end_close": latest_close,
            "pct_change": pct_change,
            "value_score": value_score,
            "value_score_tier": "合理区" if value_score > 30 else "观望",
            **fund_data,
            **scores,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh research workbench data")
    parser.add_argument(
        "--only",
        choices=["watchlist", "history", "summary", "signals"],
        help="Only refresh a specific layer",
    )
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip network fetch, use existing local CSV data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview actions without executing")
    args = parser.parse_args()

    log.info("=== Research Workbench Refresh — %s ===", date.today().isoformat())

    # 1. Load watchlist
    log.info("Loading watchlist from %s", SETTINGS.watchlist_path)
    watchlist = load_watchlist(SETTINGS.watchlist_path, market="CN")
    # Also try loading with other markets if entries exist
    for market in ("HK", "US"):
        wl_market = load_watchlist(SETTINGS.watchlist_path, market=market)
        if not wl_market.empty:
            watchlist = pd.concat([watchlist, wl_market], ignore_index=True)

    symbols = watchlist_symbols(watchlist)
    log.info("Watchlist: %d symbols — %s", len(symbols), ", ".join(symbols[:10]))

    if args.dry_run:
        log.info("[DRY-RUN] Would refresh: %s", args.only or "all layers")
        log.info("[DRY-RUN] Markets: %s", watchlist["market"].unique().tolist() if "market" in watchlist.columns else "N/A")
        return

    if args.only == "watchlist":
        return

    # 2. Fetch/load history
    if args.no_fetch:
        log.info("Loading cached history (--no-fetch)")
        history = load_history(SETTINGS)
        if not history.empty and symbols:
            history = history[history["symbol"].isin(symbols)].copy()
    else:
        log.info("Fetching history from providers...")
        history = fetch_all_history(watchlist)
        if not history.empty:
            SETTINGS.history_path.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(SETTINGS.history_path, index=False)
            log.info("Saved %d history rows to %s", len(history), SETTINGS.history_path.name)

    log.info("History: %d rows, %d symbols",
             len(history),
             history["symbol"].nunique() if "symbol" in history.columns else 0)

    if args.only == "history":
        return

    # 3. Build summary (fundamentals + scores)
    if args.no_fetch:
        log.info("Loading cached summary (--no-fetch)")
        summary = load_summary(SETTINGS)
        if not summary.empty and symbols:
            summary = summary[summary["symbol"].isin(symbols)].copy()
    else:
        log.info("Building summaries from providers...")
        summary = fetch_all_summaries(watchlist, history)
        if not summary.empty:
            SETTINGS.summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(SETTINGS.summary_path, index=False)
            log.info("Saved %d summary rows to %s", len(summary), SETTINGS.summary_path.name)

    log.info("Summary: %d rows", len(summary))

    if args.only == "summary":
        return

    # 4. Load events
    events = load_manual_events(SETTINGS)
    log.info("Events: %d rows", len(events))

    # 5. Build signals
    log.info("Building current signals...")
    signals = build_current_signals(summary, history, watchlist, events, DEFAULT_RADAR_CONFIG)
    if signals.empty:
        log.warning("Signal engine produced no output.")
    else:
        log.info("Signals generated: %d rows", len(signals))
        try:
            SETTINGS.signals_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            signals.to_csv(SETTINGS.signals_snapshot_path, index=False)
            log.info("Signals snapshot saved to %s", SETTINGS.signals_snapshot_path)
        except Exception as exc:
            log.error("Failed to save signals snapshot: %s", exc)

    # 6. Print summary
    if not signals.empty:
        triggered = int((signals.get("signal_state", pd.Series(dtype=str)) == "triggered").sum())
        near_zone = int((signals.get("signal_state", pd.Series(dtype=str)) == "near_zone").sum())
        log.info("=== Result: %d triggered, %d near-zone, %d total ===", triggered, near_zone, len(signals))
    else:
        log.info("=== No signals generated ===")

    log.info("Refresh complete.")


if __name__ == "__main__":
    main()
