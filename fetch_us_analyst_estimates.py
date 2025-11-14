#!/usr/bin/env python3
"""Fetch past five years of analyst estimates for selected US tech stocks."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

API_KEY = os.environ.get("FMP_API_KEY", "6GfWNQdQxymNoUiM2Be61I9oPDCzeNor")
BASE_URL = "https://financialmodelingprep.com/stable/analyst-estimates"
OUTPUT_DIR = Path(__file__).parent / "openbb_outputs"
YEARS = 5

STOCKS: List[Dict[str, str]] = [
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com, Inc."},
    {"symbol": "META", "name": "Meta Platforms, Inc."},
    {"symbol": "ADBE", "name": "Adobe Inc."},
    {"symbol": "CRM", "name": "Salesforce, Inc."},
    {"symbol": "INTU", "name": "Intuit Inc."},
    {"symbol": "NOW", "name": "ServiceNow, Inc."},
    {"symbol": "ZM", "name": "Zoom Video Communications, Inc."},
]


def fetch_symbol(symbol: str) -> pd.DataFrame:
    """Fetch analyst estimates for a single symbol."""
    resp = requests.get(
        BASE_URL,
        params={"symbol": symbol, "period": "annual", "apikey": API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty or "date" not in df:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    today = pd.Timestamp(date.today())
    df = df[df["date"] <= today]
    df = df.sort_values("date", ascending=False).head(YEARS)
    df = df.sort_values("date")
    df["symbol"] = symbol
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    frames: List[pd.DataFrame] = []
    failures: List[str] = []

    for stock in STOCKS:
        print(f"Fetching {stock['symbol']} ...")
        try:
            df = fetch_symbol(stock["symbol"])
        except requests.HTTPError as exc:
            failures.append(f"{stock['symbol']}: HTTP {exc.response.status_code}")
            continue
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{stock['symbol']}: {exc}")
            continue
        if df.empty:
            failures.append(f"{stock['symbol']}: no data")
            continue
        df["name"] = stock["name"]
        frames.append(df)

    if not frames:
        raise SystemExit("No data fetched for any symbol.")

    result = pd.concat(frames, ignore_index=True)

    rename_map = {
        "name": "name（公司名称）",
        "symbol": "symbol（股票代码）",
        "date": "date（财年/报告期）",
        "revenueLow": "revenueLow（营收下限）",
        "revenueAvg": "revenueAvg（营收均值）",
        "revenueHigh": "revenueHigh（营收上限）",
        "ebitdaLow": "ebitdaLow（EBITDA下限）",
        "ebitdaAvg": "ebitdaAvg（EBITDA均值）",
        "ebitdaHigh": "ebitdaHigh（EBITDA上限）",
        "ebitLow": "ebitLow（EBIT下限）",
        "ebitAvg": "ebitAvg（EBIT均值）",
        "ebitHigh": "ebitHigh（EBIT上限）",
        "netIncomeLow": "netIncomeLow（净利润下限）",
        "netIncomeAvg": "netIncomeAvg（净利润均值）",
        "netIncomeHigh": "netIncomeHigh（净利润上限）",
        "sgaExpenseLow": "sgaExpenseLow（销售与管理费用下限）",
        "sgaExpenseAvg": "sgaExpenseAvg（销售与管理费用均值）",
        "sgaExpenseHigh": "sgaExpenseHigh（销售与管理费用上限）",
        "epsLow": "epsLow（每股收益下限）",
        "epsAvg": "epsAvg（每股收益均值）",
        "epsHigh": "epsHigh（每股收益上限）",
        "numAnalystsRevenue": "numAnalystsRevenue（营收预测分析师数）",
        "numAnalystsEps": "numAnalystsEps（EPS预测分析师数）",
    }

    result = result.rename(columns=rename_map)
    output_path = OUTPUT_DIR / "us_analyst_estimates.csv"
    result.to_csv(output_path, index=False)

    print(f"Saved {len(result)} rows to {output_path}")
    if failures:
        print("Warnings:")
        for item in failures:
            print(f" - {item}")


if __name__ == "__main__":
    main()
