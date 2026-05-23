from __future__ import annotations

from typing import Any

import pandas as pd


def score_fundamentals(fund: dict[str, Any]) -> dict[str, float]:
    """Compute lightweight scores from normalized valuation fields."""
    pe = fund.get("end_pe")
    pb = fund.get("current_pb")
    dividend_yield = fund.get("dividend_yield") or 0

    score_valuation = 0.0
    if pd.notna(pe):
        if pe < 15:
            score_valuation += 10
        elif pe < 25:
            score_valuation += 5
    if pd.notna(pb):
        if pb < 2:
            score_valuation += 10
        elif pb < 5:
            score_valuation += 5

    score_ret = 10.0 if dividend_yield > 0.02 else 0.0

    return {
        "score_abs_valuation": min(20.0, score_valuation),
        "score_balance_sheet": 0.0,
        "score_shareholder_return": score_ret,
        "score_hist_valuation": 0.0,
        "score_peer_valuation": 0.0,
        "score_peg": 0.0,
        "score_growth_quality": 0.0,
        "score_sentiment": 5.0,
    }


def build_summary_row(item: pd.Series, history: pd.DataFrame, fund_data: dict[str, Any]) -> dict[str, Any] | None:
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
    scores = score_fundamentals(fund_data)
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
