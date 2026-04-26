#!/usr/bin/env python3
"""Fetch enriched price history for selected tickers via OpenBB."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    pe_percentile_5y: Optional[float]
    ps_percentile_5y: Optional[float]
    peg_ratio: Optional[float]
    fcf_yield: Optional[float]
    entry_conditions_met: int
    entry_recommendation: str

    def as_pct(self) -> float:
        return self.pct_change * 100


TICKERS: tuple[Dict[str, str], ...] = (
    {"name": "Perfect World", "symbol": "002624.SZ", "provider": "tushare"},
    {"name": "Xiaomi Group", "symbol": "1810.HK", "provider": "yfinance"},
    {"name": "Meta Platforms", "symbol": "META", "provider": "fmp"},
)

OUTPUT_DIR = Path("openbb_outputs")
LOOKBACK_DAYS = 730
PERCENTILE_LOOKBACK_DAYS = 5 * 365


def fetch_history(
    symbol: str,
    provider: str,
    start: date,
    end: date,
    fallback_provider: str = "yfinance",
) -> pd.DataFrame:
    """Fetch daily price history via OpenBB and return as DataFrame."""
    last_error: Exception | None = None
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


def add_daily_pe(
    symbol: str, df: pd.DataFrame, ticker: Optional[yf.Ticker] = None
) -> pd.DataFrame:
    """Attach trailing EPS and daily PE using Yahoo Finance fundamentals."""
    ticker = ticker or yf.Ticker(symbol)
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


def percentile_rank(series: pd.Series, value: Optional[float]) -> Optional[float]:
    """Compute percentile rank of value within series."""
    if value is None or pd.isna(value):
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    rank = (clean <= value).sum() / len(clean)
    return float(rank)


def compute_financial_metrics(ticker: yf.Ticker) -> Dict[str, Optional[float]]:
    """Collect TTM revenue, FCF, shares, and market cap."""
    metrics: Dict[str, Optional[float]] = {
        "ttm_revenue": None,
        "ttm_fcf": None,
        "shares": None,
        "market_cap": None,
    }

    fast_info = getattr(ticker, "fast_info", {}) or {}
    info = getattr(ticker, "info", {}) or {}

    shares = fast_info.get("shares") or fast_info.get("outstandingShares")
    if not shares:
        shares = info.get("sharesOutstanding")
    metrics["shares"] = float(shares) if shares else None

    market_cap = fast_info.get("marketCap") or info.get("marketCap")
    metrics["market_cap"] = float(market_cap) if market_cap else None

    q_fin = ticker.quarterly_financials
    if "Total Revenue" in q_fin.index:
        metrics["ttm_revenue"] = float(
            q_fin.loc["Total Revenue"].dropna().sort_index().tail(4).sum()
        )

    q_cf = ticker.quarterly_cashflow
    if "Free Cash Flow" in q_cf.index:
        metrics["ttm_fcf"] = float(
            q_cf.loc["Free Cash Flow"].dropna().sort_index().tail(4).sum()
        )
    elif (
        "Operating Cash Flow" in q_cf.index
        and "Capital Expenditure" in q_cf.index
    ):
        ocf = q_cf.loc["Operating Cash Flow"].dropna().sort_index().tail(4)
        capex = q_cf.loc["Capital Expenditure"].dropna().sort_index().tail(4)
        if not ocf.empty and not capex.empty:
            metrics["ttm_fcf"] = float((ocf - capex).sum())

    return metrics


def calculate_entry_signals(
    history: pd.DataFrame,
    long_history: pd.DataFrame,
    ticker: yf.Ticker,
) -> Tuple[dict, pd.DataFrame]:
    """Evaluate entry conditions and return metrics along with updated history."""
    latest_row = history.iloc[-1]
    pe_percentile = None
    if "pe" in long_history.columns:
        pe_percentile = percentile_rank(long_history["pe"], latest_row.get("pe"))

    peg_ratio = None
    if "ttm_eps" in history.columns and pd.notna(latest_row.get("ttm_eps")):
        trailing_eps = history.dropna(subset=["ttm_eps"])
        past_eps = trailing_eps[
            trailing_eps["date"] <= latest_row["date"] - pd.Timedelta(days=365)
        ]
        if not past_eps.empty:
            eps_prev = past_eps.iloc[-1]["ttm_eps"]
            if eps_prev and eps_prev != 0:
                growth = (latest_row["ttm_eps"] - eps_prev) / abs(eps_prev)
                if growth > 0:
                    peg_ratio = latest_row["pe"] / (growth * 100)

    metrics = compute_financial_metrics(ticker)
    ps_percentile = None
    if metrics.get("ttm_revenue") and metrics.get("shares"):
        sales_per_share = metrics["ttm_revenue"] / metrics["shares"]
        if sales_per_share and sales_per_share > 0:
            history = history.copy()
            history["ps_ratio"] = history["close"] / sales_per_share
            ps_percentile = percentile_rank(
                history["ps_ratio"].dropna(), history.iloc[-1]["ps_ratio"]
            )
    else:
        history = history.copy()
        history["ps_ratio"] = np.nan

    fcf_yield = None
    if metrics.get("ttm_fcf") and metrics.get("market_cap"):
        if metrics["market_cap"] > 0:
            fcf_yield = metrics["ttm_fcf"] / metrics["market_cap"]

    cond_pe = pe_percentile is not None and pe_percentile <= 0.3
    cond_peg = peg_ratio is not None and peg_ratio <= 1
    cond_ps = ps_percentile is not None and ps_percentile <= 0.3
    cond_fcf = fcf_yield is not None and fcf_yield >= 0.04

    met_count = sum([cond_pe, cond_peg, cond_ps, cond_fcf])
    if met_count == 4:
        recommendation = "建议入场"
    elif met_count >= 2:
        recommendation = "可评估入场"
    else:
        recommendation = "暂不建议入场"

    signals = {
        "pe_percentile_5y": pe_percentile,
        "ps_percentile_5y": ps_percentile,
        "peg_ratio": peg_ratio,
        "fcf_yield": fcf_yield,
        "entry_conditions_met": met_count,
        "entry_recommendation": recommendation,
    }

    return signals, history


def summarize(
    name: str,
    symbol: str,
    df: pd.DataFrame,
    signals: dict,
) -> EquitySummary:
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
        pe_percentile_5y=signals.get("pe_percentile_5y"),
        ps_percentile_5y=signals.get("ps_percentile_5y"),
        peg_ratio=signals.get("peg_ratio"),
        fcf_yield=signals.get("fcf_yield"),
        entry_conditions_met=signals.get("entry_conditions_met", 0),
        entry_recommendation=signals.get("entry_recommendation", "数据不足"),
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
        ticker = yf.Ticker(symbol)
        history = add_close_percentile(history, long_history)
        history = add_daily_pe(symbol, history, ticker=ticker)
        long_history = add_daily_pe(symbol, long_history, ticker=ticker)
        signals, history = calculate_entry_signals(history, long_history, ticker)
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
                    "ps_ratio",
                ]
            ].copy()
        )
        summary_rows.append(summarize(name, symbol, history, signals))

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
    print("\nPerformance overview (% change):")
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
            "pe_percentile_5y",
            "ps_percentile_5y",
            "peg_ratio",
            "fcf_yield",
            "entry_recommendation",
        ]
    ].copy()
    pretty["pct_change"] = pretty["pct_change"].map(lambda v: f"{v:.2f}%")
    pretty["end_pe"] = pretty["end_pe"].map(
        lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"
    )
    pretty["end_close_percentile"] = pretty["end_close_percentile"].map(
        lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "N/A"
    )
    pretty["pe_percentile_5y"] = pretty["pe_percentile_5y"].map(
        lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "N/A"
    )
    pretty["ps_percentile_5y"] = pretty["ps_percentile_5y"].map(
        lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "N/A"
    )
    pretty["peg_ratio"] = pretty["peg_ratio"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
    )
    pretty["fcf_yield"] = pretty["fcf_yield"].map(
        lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "N/A"
    )
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
