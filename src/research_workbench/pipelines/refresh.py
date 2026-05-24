"""Per-stage pipeline functions for the data refresh.

The pipeline has four logical stages:
  1. fetch_all_history    — Longbridge daily OHLCV for every market.
  2. fetch_cn_enrichment  — AKShare deep data for CN symbols only
                            (financials / valuation history / earnings forecast /
                            dividends / shareholder changes).
  3. build_all_summaries  — combine history + Longbridge fundamentals + CN
                            enrichment into one row per symbol with scores.
  4. build_all_events     — synthesize EVENT_SCHEMA rows from the CN
                            enrichment frames for downstream signal use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from research_workbench.analysis.summary import build_summary_row
from research_workbench.data_sources.indicators import add_technical_indicators
from research_workbench.data_sources.longbridge import (
    LOOKBACK_DAYS,
    LongbridgeCLIProvider,
    get_provider,
)

log = logging.getLogger("pipelines.refresh")


# ---------------------------------------------------------------------------
# Stage 1: history
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stage 1b: capital flow snapshot (one row per symbol per refresh)
# ---------------------------------------------------------------------------


def fetch_all_capital_flow(watchlist: pd.DataFrame) -> pd.DataFrame:
    """Fetch today's capital-flow snapshot for every symbol in the watchlist.

    Returns one row per symbol with the date stamped to today. Caller is
    expected to append + deduplicate against any prior history file.
    """
    if watchlist.empty:
        return pd.DataFrame()

    # Local-date timestamp: utcnow().normalize() can land on the wrong day for
    # CST-evening runs (UTC is 8h behind). pd.Timestamp.now() is naive local.
    today = pd.Timestamp.now().normalize()
    rows: list[dict] = []
    for _, item in watchlist.iterrows():
        market = str(item.get("market", "")).upper()
        symbol = str(item["symbol"])
        try:
            provider = get_provider(market)
        except ValueError:
            continue
        # Only LongbridgeCLIProvider exposes fetch_capital_flow_snapshot.
        if not isinstance(provider, LongbridgeCLIProvider):
            continue
        try:
            snap = provider.fetch_capital_flow_snapshot(symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("capital flow %s failed: %s", symbol, exc)
            continue
        if not snap:
            continue
        snap["date"] = today
        snap["data_source"] = "longbridge"
        # Drop the raw CLI timestamp; date is the canonical PK.
        snap.pop("timestamp", None)
        rows.append(snap)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def merge_capital_flow_history(
    new_snapshot: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    """Append today's snapshot to history, dropping duplicates on (symbol, date)."""
    if new_snapshot.empty:
        return existing
    if existing is None or existing.empty:
        return new_snapshot
    combined = pd.concat([existing, new_snapshot], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date", "symbol"])
    combined = (
        combined.sort_values(["symbol", "date"])
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    return combined


# ---------------------------------------------------------------------------
# Stage 2: CN deep enrichment (financials / valuation / forecast / events)
# ---------------------------------------------------------------------------


@dataclass
class CNEnrichment:
    """Concatenated CN deep data, indexable per symbol via :meth:`for_symbol`.

    ``financials`` is annual-only (the input scoring depends on); the
    ``financials_quarterly`` frame holds every period (Q1/H1/Q3/annual) for
    cadence analysis and storage. Both come from one Sina fetch per symbol.
    """

    financials: pd.DataFrame = field(default_factory=pd.DataFrame)
    financials_quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)
    valuation_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    earnings_forecasts: pd.DataFrame = field(default_factory=pd.DataFrame)
    earnings_express: pd.DataFrame = field(default_factory=pd.DataFrame)
    dividends: pd.DataFrame = field(default_factory=pd.DataFrame)
    shareholder_changes: pd.DataFrame = field(default_factory=pd.DataFrame)
    northbound_holdings: pd.DataFrame = field(default_factory=pd.DataFrame)

    def for_symbol(self, symbol: str) -> dict[str, pd.DataFrame | None]:
        sym = symbol.upper()

        def _slice(df: pd.DataFrame) -> pd.DataFrame | None:
            if df is None or df.empty or "symbol" not in df.columns:
                return None
            sub = df[df["symbol"] == sym]
            return sub if not sub.empty else None

        return {
            "financials": _slice(self.financials),
            "valuation_history": _slice(self.valuation_history),
            "earnings_forecast": _slice(self.earnings_forecasts),
            "dividend_history": _slice(self.dividends),
            "shareholder_changes": _slice(self.shareholder_changes),
        }


def _latest_fiscal_year_end(today: date | None = None) -> str:
    """Return the YYYY1231 string of the most recently completed fiscal year."""
    today = today or date.today()
    # Preannouncements for year N typically land Jan-Apr of year N+1, so we
    # always look up the prior year's 12-31 to maximize hit rate.
    return f"{today.year - 1}1231"


def fetch_cn_enrichment(watchlist: pd.DataFrame) -> CNEnrichment:
    """Pull deep AKShare data for all CN symbols in the watchlist.

    Each per-symbol fetch is wrapped in try/except so one symbol's failure
    (commonly the push2 flap on eastmoney) doesn't take down the others.
    Returns concatenated frames across symbols; downstream callers slice
    via :meth:`CNEnrichment.for_symbol`.
    """
    if watchlist.empty:
        return CNEnrichment()

    cn = watchlist[watchlist.get("market", "").astype(str).str.upper() == "CN"]
    if cn.empty:
        log.info("No CN symbols in watchlist; skipping deep enrichment.")
        return CNEnrichment()

    # Lazy import keeps Longbridge-only deployments from needing aktools.
    try:
        from research_workbench.data_sources.akshare_cn import get_provider as ak_get_provider
    except ImportError as exc:
        log.warning("AKShare provider unavailable, skipping CN enrichment: %s", exc)
        return CNEnrichment()

    akp = ak_get_provider("CN")
    report_date = _latest_fiscal_year_end()

    # Pull the market-wide tables once, then slice per symbol. Calling each
    # per-symbol variant N times would re-download ~7000-11000 rows N times.
    log.info("CN enrichment: pre-fetching market yjyg + yjbb tables")
    try:
        market_yjyg = akp.fetch_earnings_forecast_market(report_date)
        log.info("  market yjyg: %d rows", len(market_yjyg))
    except Exception as exc:  # noqa: BLE001
        log.warning("  market yjyg fetch failed: %s", exc)
        market_yjyg = pd.DataFrame()
    try:
        market_yjbb = akp.fetch_earnings_express_market(report_date)
        log.info("  market yjbb: %d rows", len(market_yjbb))
    except Exception as exc:  # noqa: BLE001
        log.warning("  market yjbb fetch failed: %s", exc)
        market_yjbb = pd.DataFrame()

    fin_all_frames: list[pd.DataFrame] = []
    val_frames: list[pd.DataFrame] = []
    fcst_frames: list[pd.DataFrame] = []
    express_frames: list[pd.DataFrame] = []
    div_frames: list[pd.DataFrame] = []
    chg_frames: list[pd.DataFrame] = []
    north_frames: list[pd.DataFrame] = []

    for _, item in cn.iterrows():
        symbol = str(item["symbol"])
        log.info("CN enrichment: %s", symbol)

        def _try(label, fn, store):
            try:
                df = fn()
                if df is not None and not df.empty:
                    store.append(df)
                else:
                    log.debug("  %s: %s empty", symbol, label)
            except Exception as exc:  # noqa: BLE001
                log.warning("  %s: %s failed - %s", symbol, label, exc)

        # All-period financials (annual subset derived later in one filter).
        _try("financials_all", lambda: akp.fetch_financials_all(symbol), fin_all_frames)
        _try("valuation_history", lambda: akp.fetch_historical_valuation(symbol), val_frames)
        _try("earnings_forecast",
             lambda: akp.fetch_earnings_preannouncement(symbol, report_date, market_df=market_yjyg),
             fcst_frames)
        _try("earnings_express",
             lambda: akp.fetch_earnings_express(symbol, report_date, market_df=market_yjbb),
             express_frames)
        _try("dividend_history", lambda: akp.fetch_dividend_history(symbol), div_frames)
        _try("shareholder_changes", lambda: akp.fetch_shareholder_changes(symbol), chg_frames)
        _try("northbound_holding", lambda: akp.fetch_northbound_holding(symbol), north_frames)

    def _concat(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fin_all = _concat(fin_all_frames)
    fin_annual = (
        fin_all[fin_all["fiscal_period"].dt.month == 12].reset_index(drop=True)
        if not fin_all.empty
        else pd.DataFrame()
    )
    return CNEnrichment(
        financials=fin_annual,
        financials_quarterly=fin_all,
        valuation_history=_concat(val_frames),
        earnings_forecasts=_concat(fcst_frames),
        earnings_express=_concat(express_frames),
        dividends=_concat(div_frames),
        shareholder_changes=_concat(chg_frames),
        northbound_holdings=_concat(north_frames),
    )


# ---------------------------------------------------------------------------
# Stage 3: summaries
# ---------------------------------------------------------------------------


def _industry_peer_pe_median(
    symbol: str,
    yjbb_market_df: pd.DataFrame | None = None,
) -> float | None:
    """Best-effort peer-PE median lookup; returns None on any failure.

    ``yjbb_market_df`` (already fetched market-wide 业绩快报 table) is used as
    the fallback industry-classification source when push2 is flapping.
    """
    try:
        from research_workbench.data_sources.akshare_cn import get_provider as ak_get_provider
    except ImportError:
        return None
    akp = ak_get_provider("CN")
    try:
        info = akp.fetch_industry_classification(symbol, yjbb_market_df=yjbb_market_df)
        industry = info.get("industry") if info else None
        if not industry:
            return None
        peers = akp.fetch_industry_peers(str(industry))
        if peers is None or peers.empty or "pe_ttm" not in peers.columns:
            return None
        median = pd.to_numeric(peers["pe_ttm"], errors="coerce").dropna().median()
        if pd.isna(median) or median <= 0:
            return None
        return float(median)
    except Exception as exc:  # noqa: BLE001
        log.debug("peer-PE lookup failed for %s: %s", symbol, exc)
        return None


def build_all_summaries(
    watchlist: pd.DataFrame,
    history: pd.DataFrame,
    enrichment: CNEnrichment | None = None,
) -> pd.DataFrame:
    """Build summary rows; per-symbol CN enrichment is sliced from ``enrichment``."""
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

        kwargs: dict = {}
        if market == "CN" and enrichment is not None:
            sliced = enrichment.for_symbol(symbol)
            kwargs.update(sliced)
            # Pass the already-fetched market yjbb table as a fallback source
            # for industry classification when push2.eastmoney is flapping.
            kwargs["industry_peer_pe_median"] = _industry_peer_pe_median(
                symbol,
                yjbb_market_df=enrichment.earnings_express if enrichment is not None else None,
            )

        row = build_summary_row(item, history, fund_data, **kwargs)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Stage 4: synthesize events from CN enrichment
# ---------------------------------------------------------------------------


def _yjyg_to_events(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Convert earnings preannouncement rows to EVENT_SCHEMA shape."""
    if forecasts is None or forecasts.empty:
        return pd.DataFrame()
    df = forecasts.copy()
    # Keep only the parent-net-income rows where yoy_pct is informative.
    if "indicator" in df.columns:
        df = df[df["indicator"].astype(str).str.contains("归属", na=False)]
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "symbol": df.get("symbol"),
        "event_date": pd.to_datetime(df.get("announce_date"), errors="coerce"),
        "event_type": "业绩预告",
        "importance": "high",
        "impact": np.where(
            pd.to_numeric(df.get("yoy_pct"), errors="coerce").fillna(0) > 0,
            "positive", "negative",
        ),
        "summary": df.get("change_text"),
        "event_value": pd.to_numeric(df.get("yoy_pct"), errors="coerce"),
        "event_source": "akshare_em",
    })
    return out.dropna(subset=["event_date"]).reset_index(drop=True)


def _dividends_to_events(dividends: pd.DataFrame) -> pd.DataFrame:
    """Convert dividend rows (实施 only) to EVENT_SCHEMA shape."""
    if dividends is None or dividends.empty:
        return pd.DataFrame()
    df = dividends.copy()
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "实施"]
    if df.empty:
        return pd.DataFrame()

    cash = pd.to_numeric(df.get("cash_dividend"), errors="coerce")
    out = pd.DataFrame({
        "symbol": df.get("symbol"),
        "event_date": pd.to_datetime(df.get("ex_div_date"), errors="coerce"),
        "event_type": "分红",
        "importance": "medium",
        "impact": "positive",
        "summary": cash.map(lambda v: f"每10股派 {v} 元" if pd.notna(v) else "派息"),
        "event_value": cash,
        "event_source": "akshare_sina",
    })
    return out.dropna(subset=["event_date"]).reset_index(drop=True)


def _shareholder_changes_to_events(changes: pd.DataFrame) -> pd.DataFrame:
    """Convert shareholder buy/sell rows to EVENT_SCHEMA shape."""
    if changes is None or changes.empty:
        return pd.DataFrame()
    df = changes.copy()

    shares = pd.to_numeric(df.get("change_shares"), errors="coerce")
    out = pd.DataFrame({
        "symbol": df.get("symbol"),
        "event_date": pd.to_datetime(df.get("announce_date"), errors="coerce"),
        "event_type": "股东增减持",
        "importance": shares.abs().map(
            lambda v: "high" if pd.notna(v) and v >= 1e7 else "medium"
        ),
        "impact": shares.map(
            lambda v: "positive" if pd.notna(v) and v > 0 else "negative"
        ),
        "summary": (
            df.get("shareholder").fillna("") + " " + df.get("change_text").fillna("")
        ).str.strip(),
        "event_value": shares,
        "event_source": "akshare_ths",
    })
    return out.dropna(subset=["event_date"]).reset_index(drop=True)


def build_all_events(enrichment: CNEnrichment | None) -> pd.DataFrame:
    """Synthesize EVENT_SCHEMA-conformant rows from CN enrichment frames."""
    if enrichment is None:
        return pd.DataFrame()
    parts = [
        _yjyg_to_events(enrichment.earnings_forecasts),
        _dividends_to_events(enrichment.dividends),
        _shareholder_changes_to_events(enrichment.shareholder_changes),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["symbol", "event_date"], ascending=[True, False]).reset_index(drop=True)
