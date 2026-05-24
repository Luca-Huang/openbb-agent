"""Per-symbol scoring + summary row construction.

The scoring functions below take optional enrichment frames (financials,
historical valuation, earnings forecast, industry peer median, dividend
history) and degrade gracefully to zero when the data is absent — so a
symbol with only quote/fundamentals data still produces a valid row.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual score functions (each returns 0..max for its dimension)
# ---------------------------------------------------------------------------


def _yoy_series(values: pd.Series) -> pd.Series:
    """Compute year-over-year change ratio from a chronological series."""
    return values.pct_change()


def score_growth_quality(financials: pd.DataFrame | None) -> float:
    """Reward sustained revenue + net-income growth and consistent profitability.

    Inputs: annual ``financials`` with ``revenue``, ``net_income_parent`` and
    ``fiscal_period``. Range: 0..15.
    """
    if financials is None or financials.empty:
        return 0.0
    df = financials.sort_values("fiscal_period").copy()
    if "revenue" not in df.columns or "net_income_parent" not in df.columns:
        return 0.0

    score = 0.0
    # net-income YoY (mean of last 3 yearly growth rates)
    ni_yoy = _yoy_series(df["net_income_parent"]).dropna().tail(3)
    if not ni_yoy.empty:
        avg = ni_yoy.mean()
        if avg >= 0.20:
            score += 5
        elif avg >= 0.10:
            score += 3
        elif avg >= 0:
            score += 1

    # revenue YoY (mean of last 3)
    rev_yoy = _yoy_series(df["revenue"]).dropna().tail(3)
    if not rev_yoy.empty:
        avg = rev_yoy.mean()
        if avg >= 0.20:
            score += 5
        elif avg >= 0.10:
            score += 3
        elif avg >= 0:
            score += 1

    # Profitability consistency: positive net income in 3 of last 5 years
    recent = df["net_income_parent"].dropna().tail(5)
    if len(recent) >= 3 and (recent > 0).sum() >= 3:
        score += 5

    return min(15.0, score)


def score_balance_sheet(financials: pd.DataFrame | None) -> float:
    """Reward low leverage + net cash + positive free cash flow.

    Inputs: latest annual row in ``financials``. Range: 0..15.
    """
    if financials is None or financials.empty:
        return 0.0
    latest = financials.sort_values("fiscal_period").iloc[-1]

    score = 0.0
    alr = latest.get("asset_liability_ratio")
    if pd.notna(alr):
        if alr < 0.30:
            score += 7
        elif alr < 0.50:
            score += 5
        elif alr < 0.70:
            score += 2

    cash = latest.get("cash_and_equivalents") or 0
    st_debt = latest.get("short_term_debt") or 0
    lt_debt = latest.get("long_term_debt") or 0
    bonds = latest.get("bonds_payable") or 0
    total_debt = (st_debt or 0) + (lt_debt or 0) + (bonds or 0)
    if pd.notna(cash) and cash > 0:
        if total_debt == 0 or cash > total_debt:
            score += 5
        elif cash > 0.5 * total_debt:
            score += 3

    fcf = latest.get("free_cash_flow")
    if pd.notna(fcf) and fcf > 0:
        score += 3

    return min(15.0, score)


def score_hist_valuation(
    valuation_history: pd.DataFrame | None,
    *,
    window_days: int = 252 * 5,
) -> float:
    """Reward a low percentile of current PE-TTM within the trailing window.

    Lower percentile = cheaper than history. Range: 0..15.
    """
    if valuation_history is None or valuation_history.empty:
        return 0.0
    pe = pd.to_numeric(valuation_history.get("pe_ttm"), errors="coerce").dropna()
    if pe.empty:
        return 0.0
    window = pe.tail(window_days)
    if len(window) < 60:
        # Not enough history to be meaningful.
        return 0.0
    current = pe.iloc[-1]
    if current <= 0:  # negative-earnings makes percentile meaningless
        return 0.0
    pct = float((window < current).mean())  # share of history cheaper than now
    if pct < 0.20:
        return 15.0
    if pct < 0.40:
        return 10.0
    if pct < 0.60:
        return 5.0
    if pct < 0.80:
        return 2.0
    return 0.0


def score_peer_valuation(
    current_pe: float | None,
    peer_pe_median: float | None,
) -> float:
    """Reward a current PE meaningfully below the industry-peer median.

    Range: 0..10.
    """
    if current_pe is None or peer_pe_median is None:
        return 0.0
    if (
        not pd.notna(current_pe)
        or not pd.notna(peer_pe_median)
        or current_pe <= 0
        or peer_pe_median <= 0
    ):
        return 0.0
    ratio = current_pe / peer_pe_median
    if ratio < 0.6:
        return 10.0
    if ratio < 0.8:
        return 6.0
    if ratio < 1.0:
        return 3.0
    return 0.0


def score_peg(
    current_pe: float | None,
    earnings_forecast: pd.DataFrame | None,
) -> float:
    """Reward a low PEG = PE / forward net-income growth (%).

    ``earnings_forecast`` is the per-symbol slice of ``stock_yjyg_em``; we use
    the row whose indicator is the parent-company net income to read
    ``yoy_pct``. Range: 0..10.
    """
    if current_pe is None or not pd.notna(current_pe) or current_pe <= 0:
        return 0.0
    if earnings_forecast is None or earnings_forecast.empty:
        return 0.0

    df = earnings_forecast.copy()
    if "indicator" in df.columns:
        parent_mask = df["indicator"].astype(str).str.contains("归属")
        if parent_mask.any():
            df = df[parent_mask]
    if "yoy_pct" not in df.columns:
        return 0.0
    yoy = pd.to_numeric(df["yoy_pct"], errors="coerce").dropna()
    if yoy.empty:
        return 0.0
    growth_pct = float(yoy.iloc[0])  # most recent first
    if growth_pct <= 0:
        return 0.0
    peg = current_pe / growth_pct  # PE / growth-in-percent (Lynch convention)
    if peg < 0.5:
        return 10.0
    if peg < 1.0:
        return 7.0
    if peg < 1.5:
        return 3.0
    return 0.0


def score_shareholder_return(
    dividend_yield: float | None,
    dividend_history: pd.DataFrame | None,
) -> float:
    """Reward both current yield and multi-year payout consistency.

    Range: 0..15.
    """
    score = 0.0
    if dividend_yield is not None and pd.notna(dividend_yield):
        if dividend_yield > 0.04:
            score += 6
        elif dividend_yield > 0.02:
            score += 4
        elif dividend_yield > 0:
            score += 1

    if dividend_history is not None and not dividend_history.empty:
        if "announce_date" in dividend_history.columns and "cash_dividend" in dividend_history.columns:
            recent = dividend_history.sort_values("announce_date", ascending=False).head(5)
            paid_years = (recent["cash_dividend"].fillna(0) > 0).sum()
            if paid_years >= 4:
                score += 5
            elif paid_years >= 2:
                score += 2

            last3 = recent.head(3)["cash_dividend"].fillna(0).sum()
            if last3 > 0:
                score += 2

    return min(15.0, score)


def score_segment_health(business_segments: pd.DataFrame | None) -> float:
    """Score business-line health from 主营构成 (revenue concentration + margin trend).

    Inputs the long-form ``business_segments`` (one symbol's slice). Looks
    at the top-level 合计 classification only — by-product / by-region rows
    are too granular to assess company-wide health. Range: 0..10.

    Sub-components:
      • Concentration sanity (0-4):
          top segment 50-90% of revenue → +4 (healthy core)
          < 50%                          → +2 (diversified, possibly weak)
          ≥ 90%                          → 0  (over-concentrated risk)
      • Top-segment gross-margin trend over the most recent 3 annual reports
          rising or flat                 → +3
          mixed                          → +1
          consistently declining         → 0  (margin compression risk)
      • Secondary-line diversification: count of non-top segments with > 5%
          revenue contribution
          ≥ 2                            → +3 (multiple revenue streams)
          1                              → +1
          0                              → 0
    """
    if business_segments is None or business_segments.empty:
        return 0.0
    if "classification" not in business_segments.columns:
        return 0.0
    summary = business_segments[business_segments["classification"] == "合计"].copy()
    if summary.empty:
        return 0.0
    summary = summary.dropna(subset=["fiscal_period"])
    if summary.empty:
        return 0.0

    # Restrict to annual reports — quarterly breakdowns are often incomplete
    # (one segment over-represented) and would mislead the concentration check.
    annual = summary[summary["fiscal_period"].dt.month == 12]
    if annual.empty:
        return 0.0
    latest_period = annual["fiscal_period"].max()
    latest = annual[annual["fiscal_period"] == latest_period]
    if latest.empty:
        return 0.0
    latest = latest.sort_values("revenue_ratio", ascending=False)
    top_row = latest.iloc[0]
    top_name = top_row.get("segment")
    top_share = pd.to_numeric(top_row.get("revenue_ratio"), errors="coerce")

    score = 0.0

    # (a) Concentration sanity
    if pd.notna(top_share):
        if 0.50 <= top_share < 0.90:
            score += 4
        elif top_share < 0.50:
            score += 2
        # else (>= 0.90): no points

    # (b) Top-segment margin trend over 3 annual reports
    if top_name:
        annual_top = summary[summary["segment"] == top_name].copy()
        annual_top = annual_top[annual_top["fiscal_period"].dt.month == 12]
        annual_top = annual_top.sort_values("fiscal_period").tail(3)
        margins = pd.to_numeric(annual_top.get("gross_margin"), errors="coerce").dropna()
        if len(margins) >= 2:
            diffs = margins.diff().dropna()
            rising_or_flat = (diffs >= -0.005).all()  # tolerance 0.5pp
            declining_all = (diffs < 0).all()
            if rising_or_flat:
                score += 3
            elif declining_all and len(diffs) >= 2:
                pass  # 0 points - margin compression risk
            else:
                score += 1

    # (c) Secondary-line diversification — count segments with > 5% revenue
    others = latest[latest["segment"] != top_name]
    others_share = pd.to_numeric(others.get("revenue_ratio"), errors="coerce").fillna(0)
    sig_count = int((others_share > 0.05).sum())
    if sig_count >= 2:
        score += 3
    elif sig_count == 1:
        score += 1

    return min(10.0, score)


def score_abs_valuation(fund: dict[str, Any]) -> float:
    """Reward a low absolute PE and PB. Range: 0..20 (existing logic preserved)."""
    pe = fund.get("end_pe")
    pb = fund.get("current_pb")
    score = 0.0
    if pd.notna(pe):
        if pe < 15:
            score += 10
        elif pe < 25:
            score += 5
    if pd.notna(pb):
        if pb < 2:
            score += 10
        elif pb < 5:
            score += 5
    return min(20.0, score)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def score_fundamentals(
    fund: dict[str, Any],
    *,
    financials: pd.DataFrame | None = None,
    valuation_history: pd.DataFrame | None = None,
    earnings_forecast: pd.DataFrame | None = None,
    industry_peer_pe_median: float | None = None,
    dividend_history: pd.DataFrame | None = None,
    business_segments: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute all nine fundamental scores; missing inputs degrade to 0."""
    current_pe = fund.get("end_pe")
    return {
        "score_abs_valuation": score_abs_valuation(fund),
        "score_hist_valuation": score_hist_valuation(valuation_history),
        "score_peer_valuation": score_peer_valuation(current_pe, industry_peer_pe_median),
        "score_peg": score_peg(current_pe, earnings_forecast),
        "score_growth_quality": score_growth_quality(financials),
        "score_balance_sheet": score_balance_sheet(financials),
        "score_segment_health": score_segment_health(business_segments),
        "score_shareholder_return": score_shareholder_return(
            fund.get("dividend_yield"), dividend_history
        ),
        "score_sentiment": 5.0,  # placeholder — no sentiment input wired yet
    }


def build_summary_row(
    item: pd.Series,
    history: pd.DataFrame,
    fund_data: dict[str, Any],
    *,
    financials: pd.DataFrame | None = None,
    valuation_history: pd.DataFrame | None = None,
    earnings_forecast: pd.DataFrame | None = None,
    industry_peer_pe_median: float | None = None,
    dividend_history: pd.DataFrame | None = None,
    business_segments: pd.DataFrame | None = None,
    shareholder_changes: pd.DataFrame | None = None,  # currently unused, accepted for plumbing
) -> dict[str, Any] | None:
    """Assemble a single summary row by combining history stats, fundamentals,
    and the deep enrichment frames (CN-only, optional)."""
    symbol = str(item["symbol"])
    market = str(item.get("market", "CN")).upper()
    sym_hist = history[history["symbol"] == symbol]
    if sym_hist.empty:
        return None

    ordered = sym_hist.sort_values("date")
    latest = ordered.iloc[-1]
    latest_close = float(latest["close"])
    start_close = float(ordered.iloc[0]["close"])
    pct_change = (latest_close - start_close) / start_close if start_close > 0 else 0

    scores = score_fundamentals(
        fund_data,
        financials=financials,
        valuation_history=valuation_history,
        earnings_forecast=earnings_forecast,
        industry_peer_pe_median=industry_peer_pe_median,
        dividend_history=dividend_history,
        business_segments=business_segments,
    )
    value_score = sum(scores.values())

    return {
        "symbol": symbol,
        "name": item.get("name", ""),
        "name_cn": item.get("name", ""),
        "name_en": "",
        "market": market,
        "theme_category": item.get("sector", ""),
        "end_close": latest_close,
        "pct_change": pct_change,
        "value_score": value_score,
        "value_score_tier": "reasonable" if value_score > 30 else "watch",
        **fund_data,
        **scores,
    }
