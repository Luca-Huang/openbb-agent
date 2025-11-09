#!/usr/bin/env python3
"""Fetch enriched price history for selected tickers via OpenBB."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from openbb import obb


@dataclass
class EquitySummary:
    name: str
    symbol: str
    start_date: date
    end_date: date
    start_close: float
    end_close: float
    abs_change: float
    pct_change: float
    best_close: float
    best_close_date: date
    end_pe: Optional[float]
    end_close_percentile: Optional[float]

    def as_pct(self) -> float:
        return self.pct_change * 100


TICKERS: tuple[Dict[str, str], ...] = (
    {"name": "Perfect World", "symbol": "002624.SZ", "provider": "tushare"},
    {"name": "Xiaomi Group", "symbol": "1810.HK", "provider": "yfinance"},
    {"name": "Meta Platforms", "symbol": "META", "provider": "fmp"},
)

OUTPUT_DIR = Path("openbb_outputs")
LOOKBACK_DAYS = 90
PERCENTILE_LOOKBACK_DAYS = 5 * 365


def fetch_history(
    symbol: str,
    provider: str,
    start: date,
    end: date,
    fallback_provider: str = "yfinance",
) -> pd.DataFrame:
    """Fetch daily price history via OpenBB and return as DataFrame."""
    for current_provider in (provider, fallback_provider):
        try:
            response = obb.equity.price.historical(
                symbol=symbol,
                provider=current_provider,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            )
            df = response.to_dataframe().reset_index().sort_values("date")
            if df.empty:
                raise RuntimeError("No rows returned")
            df["date"] = pd.to_datetime(df["date"])
            df["close_norm"] = df["close"] / df["close"].iloc[0]
            if current_provider != provider:
                print(
                    f"Warning: {symbol} via {provider} failed, "
                    f"using {current_provider} instead."
                )
            return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Warning: {symbol} via {current_provider} failed -> {exc}")
            if current_provider == fallback_provider:
                break
    raise RuntimeError(f"Unable to fetch {symbol}: {last_error}")


def add_close_percentile(
    short_df: pd.DataFrame, long_df: pd.DataFrame
) -> pd.DataFrame:
    """Add percentile column showing where each close sits within long history."""
    if long_df.empty:
        short_df = short_df.copy()
        short_df["close_percentile"] = np.nan
        return short_df

    closes = np.sort(long_df["close"].dropna().to_numpy())
    total = len(closes)

    def percentile(value: float) -> float:
        idx = np.searchsorted(closes, value, side="right")
        return idx / total

    enriched = short_df.copy()
    enriched["close_percentile"] = enriched["close"].map(percentile)
    return enriched


def add_daily_pe(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Attach trailing EPS and daily PE using Yahoo Finance fundamentals."""
    ticker = yf.Ticker(symbol)
    fundamentals = ticker.quarterly_financials
    if fundamentals.empty or "Diluted EPS" not in fundamentals.index:
        enriched = df.copy()
        enriched["ttm_eps"] = np.nan
        enriched["pe"] = np.nan
        return enriched

    eps_series = (
        fundamentals.loc["Diluted EPS"].dropna().sort_index().rename("eps").to_frame()
    )
    if len(eps_series) < 4:
        enriched = df.copy()
        enriched["ttm_eps"] = np.nan
        enriched["pe"] = np.nan
        return enriched

    eps_series["ttm_eps"] = eps_series["eps"].rolling(window=4).sum()
    ttm = (
        eps_series.dropna(subset=["ttm_eps"])
        .reset_index()
        .rename(columns={"index": "quarter_end"})
    )
    ttm["quarter_end"] = pd.to_datetime(ttm["quarter_end"])

    if ttm.empty:
        enriched = df.copy()
        enriched["ttm_eps"] = np.nan
        enriched["pe"] = np.nan
        return enriched

    merged = pd.merge_asof(
        df.sort_values("date"),
        ttm.sort_values("quarter_end"),
        left_on="date",
        right_on="quarter_end",
        direction="backward",
    )
    merged["pe"] = merged["close"] / merged["ttm_eps"]
    merged.loc[merged["ttm_eps"] == 0, "pe"] = np.nan
    return merged.drop(columns=["eps"]) if "eps" in merged else merged


def summarize(name: str, symbol: str, df: pd.DataFrame) -> EquitySummary:
    """Build a summary row for downstream reporting."""
    start_row = df.iloc[0]
    end_row = df.iloc[-1]
    best_idx = df["close"].idxmax()
    best_row = df.loc[best_idx]
    change = end_row["close"] - start_row["close"]
    pct = change / start_row["close"]
    end_pe = float(end_row["pe"]) if pd.notna(end_row.get("pe")) else None
    percentile = (
        float(end_row["close_percentile"])
        if pd.notna(end_row.get("close_percentile"))
        else None
    )

    return EquitySummary(
        name=name,
        symbol=symbol,
        start_date=pd.to_datetime(start_row["date"]).date(),
        end_date=pd.to_datetime(end_row["date"]).date(),
        start_close=float(start_row["close"]),
        end_close=float(end_row["close"]),
        abs_change=float(change),
        pct_change=float(pct),
        best_close=float(best_row["close"]),
        best_close_date=pd.to_datetime(best_row["date"]).date(),
        end_pe=end_pe,
        end_close_percentile=percentile,
    )


def main() -> None:
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    percentile_start = end - timedelta(days=PERCENTILE_LOOKBACK_DAYS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined: List[pd.DataFrame] = []
    summary_rows: List[EquitySummary] = []

    for info in TICKERS:
        name = info["name"]
        symbol = info["symbol"]
        provider = info["provider"]

        history = fetch_history(symbol, provider, start, end)
        long_history = fetch_history(symbol, provider, percentile_start, end)
        history = add_close_percentile(history, long_history)
        history = add_daily_pe(symbol, history)
        history["name"] = name
        combined.append(
            history[
                [
                    "date",
                    "name",
                    "close",
                    "close_norm",
                    "close_percentile",
                    "ttm_eps",
                    "pe",
                ]
            ].copy()
        )
        summary_rows.append(summarize(name, symbol, history))

    combined_df = pd.concat(combined, ignore_index=True)
    combined_path = OUTPUT_DIR / "three_month_close_history.csv"
    combined_df.to_csv(combined_path, index=False)

    summary_df = pd.DataFrame(
        [asdict(row) | {"pct_change": row.as_pct()} for row in summary_rows]
    )
    summary_path = OUTPUT_DIR / "three_month_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("Saved:", combined_path)
    print("Saved:", summary_path)
    print("\nThree-month performance (% change):")
    pretty = summary_df[
        [
            "name",
            "symbol",
            "pct_change",
            "start_close",
            "end_close",
            "best_close_date",
            "best_close",
            "end_pe",
            "end_close_percentile",
        ]
    ].copy()
    pretty["pct_change"] = pretty["pct_change"].map(lambda v: f"{v:.2f}%")
    pretty["end_pe"] = pretty["end_pe"].map(
        lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"
    )
    pretty["end_close_percentile"] = pretty["end_close_percentile"].map(
        lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "N/A"
    )
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
