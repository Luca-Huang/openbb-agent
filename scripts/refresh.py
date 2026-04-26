"""Unified data refresh entry-point for the A-share research workbench.

Usage:
    python scripts/refresh.py              # refresh all data
    python scripts/refresh.py --only history   # refresh history only
    python scripts/refresh.py --dry-run    # preview what would be refreshed

This script orchestrates the full data pipeline:
1. Load watchlist
2. Fetch / refresh history data
3. Fetch / refresh summary (fundamentals) data
4. Run signal engine to produce structured signals
5. Persist signals snapshot for daily change detection
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from research_workbench.config import default_settings
from research_workbench.ingestion.watchlist import load_watchlist, watchlist_symbols
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
EQUITY_CONFIG_PATH = ROOT / "equity_config.json"


def refresh_watchlist() -> pd.DataFrame:
    """Load the current watchlist."""
    log.info("Loading watchlist from %s", SETTINGS.watchlist_path)
    wl = load_watchlist(SETTINGS.watchlist_path, EQUITY_CONFIG_PATH, market="CN")
    symbols = watchlist_symbols(wl)
    log.info("Watchlist loaded: %d symbols — %s", len(symbols), ", ".join(symbols[:10]))
    return wl


def refresh_history(symbols: list[str]) -> pd.DataFrame:
    """Load (and optionally re-fetch) history data."""
    log.info("Loading history data...")
    history = load_history(SETTINGS)
    if not history.empty and symbols:
        history = history[history["symbol"].isin(symbols)].copy()
    log.info("History loaded: %d rows, %d symbols", len(history), history["symbol"].nunique() if "symbol" in history.columns else 0)
    return history


def refresh_summary(symbols: list[str]) -> pd.DataFrame:
    """Load (and optionally re-fetch) summary / fundamentals data."""
    log.info("Loading summary data...")
    summary = load_summary(SETTINGS)
    if not summary.empty and symbols:
        summary = summary[summary["symbol"].isin(symbols)].copy()
    log.info("Summary loaded: %d rows", len(summary))
    return summary


def build_and_save_signals(
    summary: pd.DataFrame,
    history: pd.DataFrame,
    watchlist: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Run the signal engine and persist the snapshot."""
    log.info("Building current signals...")
    signals = build_current_signals(summary, history, watchlist, events, DEFAULT_RADAR_CONFIG)
    if signals.empty:
        log.warning("Signal engine produced no output. Check that summary and history data are available.")
        return signals
    log.info("Signals generated: %d rows", len(signals))
    # Save snapshot for daily change detection
    try:
        SETTINGS.signals_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(SETTINGS.signals_snapshot_path, index=False)
        log.info("Signals snapshot saved to %s", SETTINGS.signals_snapshot_path)
    except Exception as exc:
        log.error("Failed to save signals snapshot: %s", exc)
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh A-share research workbench data")
    parser.add_argument(
        "--only",
        choices=["watchlist", "history", "summary", "signals"],
        help="Only refresh a specific layer",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing")
    args = parser.parse_args()

    log.info("=== Research Workbench Refresh — %s ===", datetime.now().strftime("%Y-%m-%d %H:%M"))

    if args.dry_run:
        log.info("[DRY-RUN] Would refresh: %s", args.only or "all layers")
        log.info("[DRY-RUN] Watchlist path: %s", SETTINGS.watchlist_path)
        log.info("[DRY-RUN] Summary path: %s (Supabase: %s)", SETTINGS.summary_path, bool(SETTINGS.database.supabase_url))
        log.info("[DRY-RUN] History path: %s (Supabase: %s)", SETTINGS.history_path, bool(SETTINGS.database.supabase_url))
        return

    watchlist = refresh_watchlist()
    symbols = watchlist_symbols(watchlist)

    if args.only == "watchlist":
        return

    history = refresh_history(symbols)
    if args.only == "history":
        return

    summary = refresh_summary(symbols)
    if args.only == "summary":
        return

    events = load_manual_events(SETTINGS)
    log.info("Events loaded: %d rows", len(events))

    signals = build_and_save_signals(summary, history, watchlist, events)

    # Print brief summary
    if not signals.empty:
        triggered = int((signals.get("signal_state", pd.Series(dtype=str)) == "triggered").sum())
        near_zone = int((signals.get("signal_state", pd.Series(dtype=str)) == "near_zone").sum())
        log.info("=== Summary: %d triggered, %d near-zone, %d total ===", triggered, near_zone, len(signals))
    else:
        log.info("=== No signals generated ===")

    log.info("Refresh complete.")


if __name__ == "__main__":
    main()
