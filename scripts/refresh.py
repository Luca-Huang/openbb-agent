"""Refresh local research data through the layered workbench pipeline.

Usage:
    python scripts/refresh.py
    python scripts/refresh.py --only history
    python scripts/refresh.py --no-fetch
    python scripts/refresh.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from research_workbench.config import default_settings
from research_workbench.ingestion.watchlist import load_watchlist, watchlist_symbols
from research_workbench.outputs.files import (
    load_capital_flow_history,
    load_events,
    load_history,
    load_summary,
    save_auto_events,
    save_capital_flow_history,
    save_earnings_express,
    save_financials,
    save_financials_quarterly,
    save_history,
    save_northbound_holdings,
    save_signals_snapshot,
    save_summary,
    save_valuation_history,
)
from research_workbench.pipelines.refresh import (
    build_all_events,
    build_all_summaries,
    build_all_theses,
    fetch_all_capital_flow,
    fetch_all_history,
    fetch_cn_enrichment,
    merge_capital_flow_history,
    save_all_thesis_reports,
)
from research_workbench.outputs.files import save_business_segments
from research_workbench.signal_engine.radar import DEFAULT_RADAR_CONFIG, build_current_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("refresh")

SETTINGS = default_settings(ROOT)


def load_all_watchlists() -> pd.DataFrame:
    watchlist = load_watchlist(SETTINGS.watchlist_path, market="CN")
    for market in ("HK", "US"):
        scoped = load_watchlist(SETTINGS.watchlist_path, market=market)
        if not scoped.empty:
            watchlist = pd.concat([watchlist, scoped], ignore_index=True)
    return watchlist


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh research workbench data")
    parser.add_argument(
        "--only",
        choices=["watchlist", "history", "summary", "signals"],
        help="Only refresh a specific layer",
    )
    parser.add_argument("--no-fetch", action="store_true", help="Use existing local CSV data")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing")
    args = parser.parse_args()

    log.info("=== Research Workbench Refresh - %s ===", date.today().isoformat())

    log.info("Loading watchlist from %s", SETTINGS.watchlist_path)
    watchlist = load_all_watchlists()
    symbols = watchlist_symbols(watchlist)
    log.info("Watchlist: %d symbols - %s", len(symbols), ", ".join(symbols[:10]))

    if args.dry_run:
        log.info("[DRY-RUN] Would refresh: %s", args.only or "all layers")
        markets = watchlist["market"].unique().tolist() if "market" in watchlist.columns else []
        log.info("[DRY-RUN] Markets: %s", markets)
        return

    if args.only == "watchlist":
        return

    if args.no_fetch:
        log.info("Loading cached history (--no-fetch)")
        history = load_history(SETTINGS)
        if not history.empty and symbols:
            history = history[history["symbol"].isin(symbols)].copy()
    else:
        log.info("Fetching history from Longbridge CLI...")
        history = fetch_all_history(watchlist)
        if not history.empty:
            save_history(history, SETTINGS)
            log.info("Saved %d history rows to %s", len(history), SETTINGS.history_path.name)

    log.info(
        "History: %d rows, %d symbols",
        len(history),
        history["symbol"].nunique() if "symbol" in history.columns else 0,
    )

    # Capital flow snapshot — appends today's row to the accumulating history.
    if not args.no_fetch:
        log.info("Fetching capital-flow snapshot for all symbols...")
        snapshot = fetch_all_capital_flow(watchlist)
        if not snapshot.empty:
            history_cf = load_capital_flow_history(SETTINGS)
            merged = merge_capital_flow_history(snapshot, history_cf)
            save_capital_flow_history(merged, SETTINGS)
            log.info("Capital flow: +%d new rows, %d total in %s",
                     len(snapshot), len(merged),
                     SETTINGS.capital_flow_history_path.name)

    if args.only == "history":
        return

    if args.no_fetch:
        log.info("Loading cached summary (--no-fetch)")
        summary = load_summary(SETTINGS)
        if not summary.empty and symbols:
            summary = summary[summary["symbol"].isin(symbols)].copy()
        enrichment = None
    else:
        log.info("Fetching CN deep enrichment via AKShare (AKTools)...")
        enrichment = fetch_cn_enrichment(watchlist)
        if not enrichment.financials.empty:
            save_financials(enrichment.financials, SETTINGS)
            log.info("Saved %d annual financial rows to %s",
                     len(enrichment.financials), SETTINGS.financials_path.name)
        if not enrichment.financials_quarterly.empty:
            save_financials_quarterly(enrichment.financials_quarterly, SETTINGS)
            log.info("Saved %d all-period financial rows to %s",
                     len(enrichment.financials_quarterly),
                     SETTINGS.financials_quarterly_path.name)
        if not enrichment.valuation_history.empty:
            save_valuation_history(enrichment.valuation_history, SETTINGS)
            log.info("Saved %d valuation-history rows to %s",
                     len(enrichment.valuation_history),
                     SETTINGS.valuation_history_path.name)
        if not enrichment.earnings_express.empty:
            save_earnings_express(enrichment.earnings_express, SETTINGS)
            log.info("Saved %d earnings-express rows to %s",
                     len(enrichment.earnings_express),
                     SETTINGS.earnings_express_path.name)
        if not enrichment.northbound_holdings.empty:
            save_northbound_holdings(enrichment.northbound_holdings, SETTINGS)
            log.info("Saved %d northbound-holding rows to %s",
                     len(enrichment.northbound_holdings),
                     SETTINGS.northbound_holdings_path.name)
        if not enrichment.business_segments.empty:
            save_business_segments(enrichment.business_segments, SETTINGS)
            log.info("Saved %d business-segment rows to %s",
                     len(enrichment.business_segments),
                     SETTINGS.business_segments_path.name)

        auto_events = build_all_events(enrichment)
        if not auto_events.empty:
            save_auto_events(auto_events, SETTINGS)
            log.info("Saved %d auto events to %s",
                     len(auto_events), SETTINGS.auto_events_path.name)

        log.info("Building summaries from Longbridge + AKShare enrichment...")
        summary = build_all_summaries(watchlist, history, enrichment)
        if not summary.empty:
            save_summary(summary, SETTINGS)
            log.info("Saved %d summary rows to %s", len(summary), SETTINGS.summary_path.name)

    log.info("Summary: %d rows", len(summary))

    if args.only == "summary":
        return

    events = load_events(SETTINGS)
    log.info("Events: %d rows (manual + auto)", len(events))

    # Render per-symbol thesis Markdown reports (uses summary, enrichment, capital flow).
    if not summary.empty:
        cf_history = load_capital_flow_history(SETTINGS)
        theses = build_all_theses(watchlist, history, summary, enrichment, cf_history)
        if theses:
            n = save_all_thesis_reports(theses, SETTINGS.thesis_reports_dir)
            log.info("Saved %d thesis reports to %s/", n, SETTINGS.thesis_reports_dir.name)

    log.info("Building current signals...")
    signals = build_current_signals(summary, history, watchlist, events, DEFAULT_RADAR_CONFIG)
    if signals.empty:
        log.warning("Signal engine produced no output.")
    else:
        log.info("Signals generated: %d rows", len(signals))
        try:
            save_signals_snapshot(signals, SETTINGS)
            log.info("Signals snapshot saved to %s", SETTINGS.signals_snapshot_path)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to save signals snapshot: %s", exc)

    if not signals.empty:
        triggered = int((signals.get("signal_state", pd.Series(dtype=str)) == "triggered").sum())
        near_zone = int((signals.get("signal_state", pd.Series(dtype=str)) == "near_zone").sum())
        log.info("=== Result: %d triggered, %d near-zone, %d total ===", triggered, near_zone, len(signals))
    else:
        log.info("=== No signals generated ===")

    log.info("Refresh complete.")


if __name__ == "__main__":
    main()
