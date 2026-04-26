#!/usr/bin/env python3
"""Fetch equity data via FMP (US) and yfinance (CN/HK) and evaluate entry signals.

指标说明：
1. 归一化收盘价 (close_norm)：以观察期首日价格为 1，反映区间内的相对涨跌。
2. 历史分位 (close_percentile)：当前收盘价在过去五年价格分布中的位置。
3. PE（市盈率）：股价 / TTM 每股收益，衡量盈利回本期。
4. PE分位(5年)：当前市盈率在五年历史分布中的百分位。
5. P/S（市销率）：股价 / TTM 每股收入，用于比较收入对应的估值水平。
6. P/S分位(5年)：当前市销率在五年历史分布中的百分位。
7. PEG：市盈率 / EPS 增长率（近一年）；当前数据源只有最近四个季度 EPS，暂无一年以上 TTM EPS，可在补齐历史后启用。
8. 自由现金流收益率：TTM 自由现金流 / 市值，衡量现金回报率。
9. 入场结论：满足四条条件全部为“建议入场”；满足 ≥2 条为“可评估入场”；否则“暂不建议入场”。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


API_KEY = os.environ.get("FMP_API_KEY", "6GfWNQdQxymNoUiM2Be61I9oPDCzeNor")
BASE_URL = "https://financialmodelingprep.com/stable"
OUTPUT_DIR = Path(__file__).parent / "openbb_outputs"
EQUITY_CONFIG_PATH = Path(__file__).parent / "equity_config.json"
RADAR_CONFIG_PATH = Path(__file__).parent / "radar_config.json"
LOOKBACK_DAYS = 730
PERCENTILE_LOOKBACK_DAYS = 5 * 365


SUMMARY_UPLOAD_COLUMNS = [
    "symbol",
    "name_en",
    "name_cn",
    "market",
    "theme_category",
    "investment_reason",
    "value_score",
    "value_score_tier",
    "entry_recommendation",
    "entry_reason",
    "tier_reason",
    "pe_percentile_5y",
    "ps_percentile_5y",
    "peg_ratio",
    "fcf_yield",
    "end_pe",
    "current_ps",
    "forward_pe",
    "pe_coverage_years",
    "ps_coverage_years",
    "refresh_interval_days",
    "next_refresh_date",
    "pct_change",
    "support_level_primary",
    "support_level_secondary",
    "pct_change_7d",
    "pct_change_30d",
]

HISTORY_UPLOAD_COLUMNS = [
    "symbol",
    "as_of_date",
    "name_en",
    "name_cn",
    "market",
    "open",
    "high",
    "low",
    "close",
    "close_norm",
    "close_percentile",
    "support_level_primary",
    "support_level_secondary",
    "ttm_eps",
    "pe",
    "ps_ratio",
    "ma50",
    "ma200",
    "rsi14",
    "fib_38_2",
    "fib_50",
    "fib_61_8",
    "volume",
    "volume_ma20",
    "volume_spike_ratio",
    "obv",
    "vpt",
    "vwap",
    "ad_line",
]

RADAR_UPLOAD_COLUMNS = [
    "as_of_date",
    "priority_rank",
    "symbol",
    "market",
    "trigger_type",
    "trigger_price",
    "stop_price",
    "opportunity_score",
    "risk_flags",
    "reason_1line",
    "value_score",
    "volume_spike_ratio",
    "drawdown_60d",
    "atr14",
    "dollar_volume20",
]

DEFAULT_RADAR_CONFIG: Dict[str, Any] = {
    "markets": ["US", "HK", "CN"],
    "liquidity": {
        "min_price_usd": 2.0,
        "min_dollar_volume20": 5_000_000,
    },
    "triggers": {
        "pullback": {
            "ma50_distance_max": 0.03,
            "volume_spike_min": 1.1,
            "require_support_above_primary": True,
        },
        "breakout": {
            "volume_spike_min": 1.5,
            "lookback_high_days": 20,
        },
    },
    "risk": {
        "mode": "medium_penalty",
        "max_drawdown_60d_penalty_start": 0.22,
    },
    "output": {
        "min_candidates": 8,
        "max_candidates": 15,
    },
}


def merge_nested_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = merge_nested_dict(base_value, value)
        else:
            merged[key] = value
    return merged


def load_radar_config(config_path: Path = RADAR_CONFIG_PATH) -> Dict[str, Any]:
    config = merge_nested_dict(DEFAULT_RADAR_CONFIG, {})
    if not config_path.exists():
        return config
    with config_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Radar config must be an object: {config_path}")
    return merge_nested_dict(config, raw)

def load_equity_universe(config_path: Path = EQUITY_CONFIG_PATH) -> List[Dict[str, str]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Equity config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    equities = payload.get("equities", payload) if isinstance(payload, dict) else payload
    if not equities:
        raise ValueError("Equity config contains no entries.")
    required = {"name_en", "name_cn", "symbol", "provider", "market"}
    validated: List[Dict[str, str]] = []
    for item in equities:
        missing = required - set(item)
        if missing:
            raise ValueError(f"Equity config entry missing keys {missing}: {item}")
        validated.append(item)
    return validated


STOCKS: List[Dict[str, str]] = load_equity_universe()
RADAR_CONFIG: Dict[str, Any] = load_radar_config()





@dataclass
class EquitySummary:
    name_en: str
    name_cn: str
    symbol: str
    market: str
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
    refresh_interval_days: int
    next_refresh_date: date
    current_ps: Optional[float]
    forward_pe: Optional[float]
    pe_coverage_years: Optional[float]
    ps_coverage_years: Optional[float]
    score_hist_valuation: float
    score_abs_valuation: float
    score_peer_valuation: float
    score_peg: float
    score_growth_quality: float
    score_balance_sheet: float
    score_shareholder_return: float
    score_support: float
    score_sentiment: float
    value_score: float
    value_score_tier: str
    entry_reason: str
    tier_reason: str

    def as_pct(self) -> float:
        return self.pct_change * 100


def fmp_get(endpoint: str, params: Optional[dict] = None) -> list | dict:
    params = params or {}
    params.setdefault("apikey", API_KEY)
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_history_fmp(symbol: str, start: date, end: date) -> pd.DataFrame:
    data = fmp_get(
        "historical-price-eod/light",
        {
            "symbol": symbol,
            "from": start.isoformat(),
            "to": end.isoformat(),
        },
    )
    df = pd.DataFrame(data)
    if df.empty:
        raise RuntimeError(f"No price data for {symbol}")
    df["date"] = pd.to_datetime(df["date"])
    if hasattr(df["date"].iloc[0], "tzinfo") and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)
    df = df.sort_values("date")
    if "price" in df.columns and "close" not in df.columns:
        df["close"] = df["price"]
    for col in ["open", "high", "low", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
    df["close_norm"] = df["close"] / df["close"].iloc[0]
    return df[["date", "open", "high", "low", "close", "volume", "close_norm"]]


def fetch_history(stock: Dict[str, str], start: date, end: date) -> pd.DataFrame:
    if stock["provider"] == "fmp":
        try:
            return fetch_history_fmp(stock["symbol"], start, end)
        except requests.HTTPError:
            pass
    # yfinance fallback for CN/HK
    ticker = yf.Ticker(stock["symbol"])
    df = ticker.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No yfinance data for {stock['symbol']}")
    df = df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for col in ["high", "low", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
    df["close_norm"] = df["close"] / df["close"].iloc[0]
    return df[["date", "open", "high", "low", "close", "volume", "close_norm"]]


def add_close_percentile(short_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        out = short_df.copy()
        out["close_percentile"] = np.nan
        return out
    closes = np.sort(long_df["close"].dropna().to_numpy())
    total = len(closes)

    def percentile(value: float) -> float:
        idx = np.searchsorted(closes, value, side="right")
        return idx / total

    enriched = short_df.copy()
    enriched["close_percentile"] = enriched["close"].map(percentile)
    return enriched


def calculate_support_levels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling support levels using local minima percentiles."""
    df = df.sort_values("date").copy()
    rolling_min = df["close"].rolling(window=window, min_periods=1).min()
    # Secondary support: 10% higher than rolling min
    df["support_level"] = rolling_min
    df["support_level_secondary"] = rolling_min * 1.1
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.sort_values("date").copy()
    enriched["ma50"] = enriched["close"].rolling(window=50, min_periods=1).mean()
    enriched["ma200"] = enriched["close"].rolling(window=200, min_periods=1).mean()
    enriched["rsi14"] = compute_rsi(enriched["close"], period=14)
    rolling_high = enriched["close"].rolling(window=120, min_periods=1).max()
    rolling_low = enriched["close"].rolling(window=120, min_periods=1).min()
    diff = (rolling_high - rolling_low).replace(0, np.nan)
    enriched["fib_38_2"] = rolling_high - diff * 0.382
    enriched["fib_50"] = rolling_high - diff * 0.5
    enriched["fib_61_8"] = rolling_high - diff * 0.618
    return enriched


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.sort_values("date").copy()
    enriched["volume"] = pd.to_numeric(enriched.get("volume"), errors="coerce")
    enriched["volume_ma20"] = enriched["volume"].rolling(window=20, min_periods=1).mean()
    enriched["volume_spike_ratio"] = enriched["volume"] / enriched["volume_ma20"]

    close_diff = enriched["close"].diff()
    direction = np.sign(close_diff).fillna(0.0)
    obv = (direction * enriched["volume"].fillna(0.0)).cumsum()
    enriched["obv"] = obv

    pct_change = enriched["close"].pct_change().fillna(0.0)
    enriched["vpt"] = (pct_change * enriched["volume"].fillna(0.0)).cumsum()

    typical_price = enriched["close"]
    if {"high", "low"}.issubset(enriched.columns):
        typical_price = (enriched["high"] + enriched["low"] + enriched["close"]) / 3
    cum_vol = enriched["volume"].fillna(0.0).cumsum()
    cum_tp = (typical_price * enriched["volume"].fillna(0.0)).cumsum()
    enriched["vwap"] = np.where(cum_vol > 0, cum_tp / cum_vol, np.nan)

    if {"high", "low"}.issubset(enriched.columns):
        money_flow_mult = (
            ((enriched["close"] - enriched["low"]) - (enriched["high"] - enriched["close"]))
            / (enriched["high"] - enriched["low"]).replace(0, np.nan)
        )
        money_flow_mult = money_flow_mult.fillna(0.0)
    else:
        money_flow_mult = pd.Series(0.0, index=enriched.index)
    enriched["ad_line"] = (money_flow_mult * enriched["volume"].fillna(0.0)).cumsum()
    return enriched


def add_radar_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.sort_values("date").copy()
    lookback_high = int(
        RADAR_CONFIG.get("triggers", {})
        .get("breakout", {})
        .get("lookback_high_days", 20)
    )
    close = pd.to_numeric(enriched.get("close"), errors="coerce")
    high = pd.to_numeric(enriched.get("high", close), errors="coerce")
    low = pd.to_numeric(enriched.get("low", close), errors="coerce")
    volume = pd.to_numeric(enriched.get("volume"), errors="coerce")

    enriched["high_20d"] = high.rolling(window=lookback_high, min_periods=1).max().shift(1)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    enriched["atr14"] = true_range.rolling(window=14, min_periods=1).mean()
    rolling_peak = close.rolling(window=60, min_periods=1).max()
    enriched["drawdown_60d"] = (rolling_peak - close) / rolling_peak.replace(0, np.nan)
    enriched["dollar_volume20"] = (close * volume).rolling(window=20, min_periods=1).mean()
    return enriched


def get_ttm_eps_series(symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    q_fin = ticker.quarterly_financials
    if q_fin.empty or "Diluted EPS" not in q_fin.index:
        return pd.DataFrame()
    eps = q_fin.loc["Diluted EPS"].dropna().sort_index()
    if eps.empty:
        return pd.DataFrame()
    ttm = eps.rolling(window=4, min_periods=4).sum().dropna()
    if ttm.empty:
        return pd.DataFrame()
    df = ttm.reset_index().rename(columns={"index": "date", 0: "ttm_eps"})
    if "ttm_eps" not in df.columns:
        df = df.rename(columns={df.columns[1]: "ttm_eps"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def trailing_change_map(history_df: pd.DataFrame, days: int) -> Dict[str, float]:
    if history_df.empty:
        return {}
    result: Dict[str, float] = {}
    history_sorted = history_df.sort_values("date")
    grouped = history_sorted.groupby("symbol")
    offset = pd.Timedelta(days=days)
    for symbol, group in grouped:
        if group.empty:
            continue
        last_date = group["date"].iloc[-1]
        last_close = group["close"].iloc[-1]
        past = group[group["date"] <= last_date - offset]
        if past.empty:
            continue
        base_close = past["close"].iloc[-1]
        if base_close:
            result[symbol] = (last_close - base_close) / base_close * 100
    return result


def compute_metrics_yf(symbol: str) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "ttm_revenue": None,
        "ttm_fcf": None,
        "shares": None,
        "market_cap": None,
        "forward_pe": None,
        "revenue_growth": None,
        "profit_growth": None,
        "total_cash": None,
        "total_debt": None,
        "dividend_yield": None,
        "buyback_amount": None,
        "recommendation_key": None,
        "recommendation_mean": None,
    }
    ticker = yf.Ticker(symbol)
    q_fin = ticker.quarterly_financials
    if not q_fin.empty:
        def get_series(row: str) -> pd.Series:
            return q_fin.loc[row].dropna() if row in q_fin.index else pd.Series(dtype=float)

        revenue = get_series("Total Revenue")
        net_income = get_series("Net Income")
        shares = get_series("Diluted Average Shares")
        if not revenue.empty:
            metrics["ttm_revenue"] = float(revenue.iloc[:4].sum())
            if len(revenue) >= 8:
                curr = revenue.iloc[:4].sum()
                prev = revenue.iloc[4:8].sum()
                if prev:
                    metrics["revenue_growth"] = (curr - prev) / abs(prev)
        if not net_income.empty:
            curr_profit = net_income.iloc[:4].sum()
            metrics["ttm_profit"] = float(curr_profit)
            if len(net_income) >= 8:
                prev_profit = net_income.iloc[4:8].sum()
                if prev_profit:
                    metrics["profit_growth"] = (curr_profit - prev_profit) / abs(prev_profit)
        if not shares.empty:
            metrics["shares"] = float(shares.iloc[0])
    q_cf = ticker.quarterly_cashflow
    if not q_cf.empty and "Free Cash Flow" in q_cf.index:
        fcf = q_cf.loc["Free Cash Flow"].dropna()
        if not fcf.empty:
            metrics["ttm_fcf"] = float(fcf.iloc[:4].sum())
    if not q_cf.empty and "Repurchase Of Capital Stock" in q_cf.index:
        buyback = q_cf.loc["Repurchase Of Capital Stock"].dropna()
        if not buyback.empty:
            metrics["buyback_amount"] = float(buyback.iloc[:4].sum())
    fast = getattr(ticker, "fast_info", {}) or {}
    info = getattr(ticker, "info", {}) or {}
    market_cap = fast.get("marketCap") or info.get("marketCap")
    metrics["market_cap"] = float(market_cap) if market_cap else None
    forward_pe = info.get("forwardPE") or fast.get("forwardPE")
    metrics["forward_pe"] = float(forward_pe) if forward_pe else None
    metrics["total_cash"] = float(info.get("totalCash")) if info.get("totalCash") else None
    metrics["total_debt"] = float(info.get("totalDebt")) if info.get("totalDebt") else None
    metrics["dividend_yield"] = info.get("dividendYield")
    metrics["recommendation_key"] = info.get("recommendationKey")
    metrics["recommendation_mean"] = info.get("recommendationMean")
    return metrics


def percentile_rank(series: pd.Series, value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return float((clean <= value).sum() / len(clean))


def score_percentile(pct: Optional[float]) -> float:
    if pct is None or pd.isna(pct):
        return 0.0
    if pct <= 0.10:
        return 10.0
    if pct <= 0.30:
        return 7.0
    if pct <= 0.50:
        return 4.0
    if pct <= 0.70:
        return 1.0
    return 0.0


def score_fcf_yield(value: Optional[float]) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if value >= 0.06:
        return 10.0
    if value >= 0.05:
        return 7.0
    if value >= 0.04:
        return 4.0
    if value >= 0.03:
        return 1.0
    return 0.0


def score_peg_value(peg: Optional[float]) -> float:
    if peg is None or pd.isna(peg):
        return 0.0
    if peg <= 0.8:
        return 15.0
    if peg <= 1.0:
        return 10.0
    if peg <= 1.2:
        return 5.0
    return 0.0


def score_growth_quality(rev_growth: Optional[float], profit_growth: Optional[float]) -> float:
    if rev_growth is None or profit_growth is None:
        return 0.0
    if rev_growth > 0 and profit_growth > rev_growth:
        return 15.0
    if rev_growth > 0 and profit_growth > 0:
        return 10.0
    if rev_growth > 0 and profit_growth <= 0:
        return 5.0
    return 0.0


def score_balance_sheet(total_cash: Optional[float], total_debt: Optional[float]) -> float:
    if total_cash is None or total_debt is None:
        return 4.0
    if total_cash > total_debt:
        return 10.0
    if total_debt <= total_cash * 1.5:
        return 7.0
    if total_debt <= total_cash * 3:
        return 4.0
    return 0.0


def score_shareholder_return(dividend_yield: Optional[float], buyback_amount: Optional[float]) -> float:
    has_buyback = buyback_amount is not None and buyback_amount < 0
    has_dividend = dividend_yield is not None and dividend_yield > 0
    if has_buyback and has_dividend:
        return 10.0
    if has_buyback or has_dividend:
        return 7.0
    return 0.0


def score_support_ratio(ratio: Optional[float]) -> float:
    if ratio is None or pd.isna(ratio):
        return 0.0
    if ratio <= 1.02:
        return 10.0
    if ratio <= 1.05:
        return 5.0
    return 0.0


def score_sentiment(reco_key: Optional[str], reco_mean: Optional[float]) -> float:
    key = (reco_key or "").lower()
    if key in {"sell", "underperform"}:
        return 10.0
    if key == "hold":
        return 3.0
    if key in {"buy", "strong_buy"}:
        return 0.0
    if reco_mean is not None:
        if reco_mean >= 4.5:
            return 10.0
        if reco_mean >= 3.5:
            return 5.0
    return 3.0


def detect_trigger_type(
    latest_row: pd.Series,
    trigger_cfg: Dict[str, Any],
) -> Optional[str]:
    close = pd.to_numeric(latest_row.get("close"), errors="coerce")
    ma50 = pd.to_numeric(latest_row.get("ma50"), errors="coerce")
    high_20d = pd.to_numeric(latest_row.get("high_20d"), errors="coerce")
    volume_spike = pd.to_numeric(latest_row.get("volume_spike_ratio"), errors="coerce")
    support_primary = pd.to_numeric(
        latest_row.get("support_level_primary", latest_row.get("support_level")),
        errors="coerce",
    )

    pullback_cfg = trigger_cfg.get("pullback", {})
    breakout_cfg = trigger_cfg.get("breakout", {})
    pullback_distance_max = float(pullback_cfg.get("ma50_distance_max", 0.03))
    pullback_volume_min = float(pullback_cfg.get("volume_spike_min", 1.1))
    require_support = bool(pullback_cfg.get("require_support_above_primary", True))
    breakout_volume_min = float(breakout_cfg.get("volume_spike_min", 1.5))

    breakout = (
        pd.notna(close)
        and pd.notna(high_20d)
        and close > high_20d
        and (pd.notna(volume_spike) and volume_spike >= breakout_volume_min)
    )
    if breakout:
        return "breakout"

    near_ma50 = (
        pd.notna(close)
        and pd.notna(ma50)
        and ma50 > 0
        and abs(close - ma50) / ma50 <= pullback_distance_max
    )
    volume_ok = pd.notna(volume_spike) and volume_spike >= pullback_volume_min
    support_ok = True
    if require_support:
        support_ok = pd.notna(support_primary) and pd.notna(close) and close >= support_primary
    if near_ma50 and volume_ok and support_ok:
        return "pullback"

    return None


def build_radar_record(
    summary_row: pd.Series,
    latest_row: pd.Series,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    trigger_type = detect_trigger_type(latest_row, cfg.get("triggers", {}))
    if trigger_type is None:
        return None

    close = pd.to_numeric(latest_row.get("close"), errors="coerce")
    support_secondary = pd.to_numeric(
        latest_row.get("support_level_secondary"),
        errors="coerce",
    )
    ma200 = pd.to_numeric(latest_row.get("ma200"), errors="coerce")
    if pd.notna(support_secondary):
        stop_price = support_secondary
    elif pd.notna(ma200):
        stop_price = ma200 * 0.97
    else:
        stop_price = np.nan

    volume_spike = pd.to_numeric(latest_row.get("volume_spike_ratio"), errors="coerce")
    reason = (
        f"{'放量突破' if trigger_type == 'breakout' else '回踩确认'}，"
        f"量能比 {volume_spike:.2f}" if pd.notna(volume_spike) else
        f"{'放量突破' if trigger_type == 'breakout' else '回踩确认'}"
    )
    return {
        "as_of_date": pd.to_datetime(latest_row.get("date"), errors="coerce").date().isoformat()
        if pd.notna(latest_row.get("date"))
        else date.today().isoformat(),
        "symbol": summary_row.get("symbol", latest_row.get("symbol")),
        "market": summary_row.get("market", latest_row.get("market")),
        "trigger_type": trigger_type,
        "trigger_price": close,
        "stop_price": stop_price,
        "opportunity_score": np.nan,
        "risk_flags": "",
        "reason_1line": reason,
    }


def compute_risk_penalty(
    drawdown_60d: Optional[float],
    atr14: Optional[float],
    cfg: Dict[str, Any],
) -> float:
    penalty = 0.0
    dd_start = float(cfg.get("max_drawdown_60d_penalty_start", 0.22))
    if drawdown_60d is not None and not pd.isna(drawdown_60d) and drawdown_60d > dd_start:
        penalty += min(20.0, (drawdown_60d - dd_start) * 100)
    if atr14 is not None and not pd.isna(atr14):
        if atr14 >= 12:
            penalty += 6.0
        elif atr14 >= 8:
            penalty += 3.0
    return penalty


def score_opportunity(
    value_score: Optional[float],
    trigger_type: Optional[str],
    volume_spike_ratio: Optional[float],
    drawdown_60d: Optional[float],
    atr14: Optional[float],
    cfg: Dict[str, Any],
) -> float:
    base = float(value_score) if value_score is not None and not pd.isna(value_score) else 0.0
    trigger_bonus = 0.0
    if trigger_type == "breakout":
        trigger_bonus = 8.0
    elif trigger_type == "pullback":
        trigger_bonus = 6.0

    volume_bonus = 0.0
    if volume_spike_ratio is not None and not pd.isna(volume_spike_ratio):
        volume_bonus = max(0.0, min(12.0, (float(volume_spike_ratio) - 1.0) * 10))
    risk_penalty = compute_risk_penalty(drawdown_60d, atr14, cfg)
    return max(0.0, min(100.0, base + trigger_bonus + volume_bonus - risk_penalty))


def select_radar_candidates(
    radar_df: pd.DataFrame,
    min_n: int,
    max_n: int,
) -> pd.DataFrame:
    if radar_df.empty:
        return radar_df.copy()
    filtered = radar_df.copy()
    filtered = filtered[filtered["trigger_type"].notna()]
    if filtered.empty:
        return filtered

    filtered["opportunity_score"] = pd.to_numeric(
        filtered["opportunity_score"], errors="coerce"
    ).fillna(-1.0)
    filtered = filtered.sort_values("opportunity_score", ascending=False)
    max_n = max(max_n, 1)
    min_n = max(min_n, 0)

    selected_indices: List[int] = []
    if "market" in filtered.columns:
        for market in ["US", "HK", "CN"]:
            top = filtered[filtered["market"] == market].head(1)
            if not top.empty:
                selected_indices.extend(top.index.tolist())

    for idx in filtered.index.tolist():
        if idx in selected_indices:
            continue
        selected_indices.append(idx)
        if len(selected_indices) >= max_n:
            break

    selected = filtered.loc[selected_indices].copy()
    selected = selected.sort_values("opportunity_score", ascending=False)
    if len(selected) < min_n:
        selected = filtered.head(min_n)
    return selected.head(max_n)


def build_opportunity_radar(
    summary_df: pd.DataFrame,
    history_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    if summary_df.empty or history_df.empty:
        return pd.DataFrame(columns=RADAR_UPLOAD_COLUMNS)

    latest = (
        history_df.sort_values("date")
        .dropna(subset=["symbol"])
        .groupby("symbol")
        .last()
    )
    if latest.empty:
        return pd.DataFrame(columns=RADAR_UPLOAD_COLUMNS)

    liquidity_cfg = cfg.get("liquidity", {})
    min_price = float(liquidity_cfg.get("min_price_usd", 2.0))
    min_dollar_volume20 = float(liquidity_cfg.get("min_dollar_volume20", 5_000_000))
    risk_cfg = cfg.get("risk", {})
    dd_start = float(risk_cfg.get("max_drawdown_60d_penalty_start", 0.22))

    radar_records: List[Dict[str, Any]] = []
    for _, summary_row in summary_df.iterrows():
        symbol = summary_row.get("symbol")
        if symbol not in latest.index:
            continue
        latest_row = latest.loc[symbol]
        radar_rec = build_radar_record(summary_row, latest_row, cfg)
        if radar_rec is None:
            continue

        trigger_price = pd.to_numeric(radar_rec.get("trigger_price"), errors="coerce")
        dollar_volume20 = pd.to_numeric(latest_row.get("dollar_volume20"), errors="coerce")
        if pd.isna(trigger_price) or trigger_price < min_price:
            continue
        if pd.isna(dollar_volume20) or dollar_volume20 < min_dollar_volume20:
            continue

        drawdown = pd.to_numeric(latest_row.get("drawdown_60d"), errors="coerce")
        atr14 = pd.to_numeric(latest_row.get("atr14"), errors="coerce")
        volume_spike = pd.to_numeric(latest_row.get("volume_spike_ratio"), errors="coerce")
        opportunity_score = score_opportunity(
            summary_row.get("value_score"),
            radar_rec.get("trigger_type"),
            volume_spike,
            drawdown,
            atr14,
            risk_cfg,
        )
        risk_flags: List[str] = []
        if pd.notna(drawdown) and drawdown > dd_start:
            risk_flags.append("drawdown60d_high")
        if pd.notna(atr14) and atr14 >= 12:
            risk_flags.append("atr_high")

        radar_rec.update(
            {
                "value_score": pd.to_numeric(summary_row.get("value_score"), errors="coerce"),
                "volume_spike_ratio": volume_spike,
                "drawdown_60d": drawdown,
                "atr14": atr14,
                "dollar_volume20": dollar_volume20,
                "opportunity_score": opportunity_score,
                "risk_flags": ",".join(risk_flags),
            }
        )
        radar_records.append(radar_rec)

    radar_df = pd.DataFrame(radar_records)
    if radar_df.empty:
        return pd.DataFrame(columns=RADAR_UPLOAD_COLUMNS)
    radar_df = select_radar_candidates(
        radar_df,
        min_n=int(cfg.get("output", {}).get("min_candidates", 8)),
        max_n=int(cfg.get("output", {}).get("max_candidates", 15)),
    )
    radar_df = radar_df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    radar_df["priority_rank"] = radar_df.index + 1
    for col in RADAR_UPLOAD_COLUMNS:
        if col not in radar_df.columns:
            radar_df[col] = np.nan
    return radar_df[RADAR_UPLOAD_COLUMNS]


def classify_score(score: float) -> str:
    if score >= 80:
        return "黄金坑"
    if score >= 60:
        return "白银坑"
    if score >= 40:
        return "合理区"
    return "观望"


def score_peer_ratio(ratio: Optional[float]) -> float:
    if ratio is None or pd.isna(ratio):
        return 0.0
    if ratio <= 0.8:
        return 10.0
    if ratio <= 1.2:
        return 5.0
    return 0.0


def calculate_entry_signals(
    history: pd.DataFrame,
    long_history: pd.DataFrame,
    metrics: Dict[str, Optional[float]],
) -> Tuple[dict, pd.DataFrame, float, float]:
    latest_row = history.iloc[-1]
    pe_series = long_history["pe"] if "pe" in long_history else pd.Series(dtype=float)
    ps_series = long_history["ps_ratio"] if "ps_ratio" in long_history else pd.Series(dtype=float)
    coverage_pe_years = (
        (pe_series.dropna().index.size / 252) if not pe_series.dropna().empty else 0.0
    )
    coverage_ps_years = (
        (ps_series.dropna().index.size / 252) if not ps_series.dropna().empty else 0.0
    )
    pe_percentile = percentile_rank(pe_series, latest_row.get("pe"))

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
    if metrics.get("ttm_fcf") and metrics.get("market_cap") and metrics["market_cap"] > 0:
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
        "forward_pe": metrics.get("forward_pe"),
    }

    return signals, history, coverage_pe_years, coverage_ps_years


def determine_refresh_interval(percentile: Optional[float]) -> int:
    if percentile is None or pd.isna(percentile):
        return 7
    if percentile >= 0.70:
        return 21
    if percentile >= 0.45:
        return 12
    return 1


def summarize(
    stock: Dict[str, str],
    df: pd.DataFrame,
    signals: dict,
    cov_pe: float,
    cov_ps: float,
    metrics: Dict[str, Optional[float]],
) -> EquitySummary:
    start_row = df.iloc[0]
    end_row = df.iloc[-1]
    best_idx = df["close"].idxmax()
    best_row = df.loc[best_idx]
    change = end_row["close"] - start_row["close"]
    pct = change / start_row["close"]
    end_pe = float(end_row["pe"]) if pd.notna(end_row.get("pe")) else None
    end_ps = float(end_row["ps_ratio"]) if pd.notna(end_row.get("ps_ratio")) else None
    percentile = (
        float(end_row["close_percentile"])
        if pd.notna(end_row.get("close_percentile"))
        else None
    )

    percentile_for_schedule = end_close_percentile = percentile
    interval = determine_refresh_interval(percentile_for_schedule)
    next_refresh = end_row["date"] + pd.Timedelta(days=interval)

    forward_pe_val = signals.get("forward_pe")
    support_ratio = (
        end_row["close"] / end_row["support_level"]
        if end_row.get("support_level")
           and end_row.get("support_level") not in (0, np.nan)
        else None
    )

    pct_values = [signals.get("pe_percentile_5y"), signals.get("ps_percentile_5y")]
    pct_values = [p for p in pct_values if p is not None and not pd.isna(p)]
    hist_pct = min(pct_values) if pct_values else None
    score_hist = score_percentile(hist_pct)
    score_abs = score_fcf_yield(signals.get("fcf_yield"))
    score_peer = 0.0  # placeholder, filled later
    score_peg = score_peg_value(signals.get("peg_ratio"))
    score_growth = score_growth_quality(metrics.get("revenue_growth"), metrics.get("profit_growth"))
    score_balance = score_balance_sheet(metrics.get("total_cash"), metrics.get("total_debt"))
    score_shareholder = score_shareholder_return(metrics.get("dividend_yield"), metrics.get("buyback_amount"))
    score_support_level = score_support_ratio(support_ratio)
    score_sentiment_val = score_sentiment(metrics.get("recommendation_key"), metrics.get("recommendation_mean"))
    value_score_partial = (
        score_hist
        + score_abs
        + score_peer
        + score_peg
        + score_growth
        + score_balance
        + score_shareholder
        + score_support_level
        + score_sentiment_val
    )
    tier = classify_score(value_score_partial)

    return EquitySummary(
        name_en=stock["name_en"],
        name_cn=stock["name_cn"],
        symbol=stock["symbol"],
        market=stock["market"],
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
        refresh_interval_days=interval,
        next_refresh_date=next_refresh.date(),
        current_ps=end_ps,
        forward_pe=forward_pe_val,
        pe_coverage_years=cov_pe,
        ps_coverage_years=cov_ps,
        score_hist_valuation=score_hist,
        score_abs_valuation=score_abs,
        score_peer_valuation=score_peer,
        score_peg=score_peg,
        score_growth_quality=score_growth,
        score_balance_sheet=score_balance,
        score_shareholder_return=score_shareholder,
        score_support=score_support_level,
        score_sentiment=score_sentiment_val,
        value_score=value_score_partial,
        value_score_tier=tier,
        tier_reason=(
            f"{tier}: "
            f"估值{score_hist}+绝对{score_abs}+同行{score_peer}+PEG{score_peg}+增长{score_growth}+"
            f"资产{score_balance}+回报{score_shareholder}+支撑{score_support_level}+情绪{score_sentiment_val}"
        ),
        entry_reason=signals.get("entry_recommendation"),
    )


def main() -> None:
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    percentile_start = end - timedelta(days=PERCENTILE_LOOKBACK_DAYS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined: List[pd.DataFrame] = []
    summary_rows: List[EquitySummary] = []
    failures: List[str] = []

    for stock in STOCKS:
        try:
            history = fetch_history(stock, start, end)
            long_history = fetch_history(stock, percentile_start, end)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{stock['symbol']}: {exc}")
            continue

        history = add_close_percentile(history, long_history)
        long_history = add_close_percentile(long_history, long_history)

        yf_symbol = stock.get("yf_symbol", stock["symbol"])
        try:
            metrics = compute_metrics_yf(yf_symbol)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{stock['symbol']} metrics fetch failed: {exc}")
            metrics = {}
        ttm_series = get_ttm_eps_series(yf_symbol)
        if not ttm_series.empty:
            history = pd.merge_asof(
                history.sort_values("date"),
                ttm_series.sort_values("date"),
                on="date",
                direction="backward",
            )
            long_history = pd.merge_asof(
                long_history.sort_values("date"),
                ttm_series.sort_values("date"),
                on="date",
                direction="backward",
            )
            history["pe"] = history["close"] / history["ttm_eps"]
            long_history["pe"] = long_history["close"] / long_history["ttm_eps"]
            history.loc[history["ttm_eps"] == 0, "pe"] = np.nan
            long_history.loc[long_history["ttm_eps"] == 0, "pe"] = np.nan
        else:
            history["ttm_eps"] = np.nan
            history["pe"] = np.nan
            long_history["ttm_eps"] = np.nan
            long_history["pe"] = np.nan

        signals, history, coverage_pe, coverage_ps = calculate_entry_signals(
            history, long_history, metrics
        )
        display_name = f"{stock['name_en']}（{stock['name_cn']}）"
        history["name_en"] = stock["name_en"]
        history["name_cn"] = stock["name_cn"]
        history["name"] = display_name
        history["market"] = stock["market"]
        history["symbol"] = stock["symbol"]
        history = calculate_support_levels(history)
        history = add_trend_indicators(history)
        history = add_volume_indicators(history)
        history = add_radar_features(history)
        combined.append(
            history[
                [
                    "date",
                    "symbol",
                    "name",
                    "name_en",
                    "name_cn",
                    "market",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "support_level",
                    "support_level_secondary",
                    "close_norm",
                    "close_percentile",
                    "ttm_eps",
                    "pe",
                    "ps_ratio",
                    "ma50",
                    "ma200",
                    "rsi14",
                    "fib_38_2",
                    "fib_50",
                    "fib_61_8",
                    "volume_ma20",
                    "volume_spike_ratio",
                    "obv",
                    "vpt",
                    "vwap",
                    "ad_line",
                    "high_20d",
                    "atr14",
                    "drawdown_60d",
                    "dollar_volume20",
                ]
            ].copy()
        )
        summary_rows.append(summarize(stock, history, signals, coverage_pe, coverage_ps, metrics))

    if not combined:
        raise SystemExit("No data collected. Check logs for failures.")

    combined_df = pd.concat(combined, ignore_index=True)
    combined_path = OUTPUT_DIR / "three_month_close_history.csv"
    combined_df.to_csv(combined_path, index=False)

    summary_df = pd.DataFrame(
        [asdict(row) | {"pct_change": row.as_pct()} for row in summary_rows]
    )
    def apply_peer_scores(df: pd.DataFrame) -> pd.DataFrame:
        for market, group in df.groupby("market"):
            median_pe = group["end_pe"].median()
            median_ps = group["current_ps"].median()
            for idx in group.index:
                ratios = []
                if median_pe and df.at[idx, "end_pe"]:
                    ratios.append(df.at[idx, "end_pe"] / median_pe)
                if median_ps and df.at[idx, "current_ps"]:
                    ratios.append(df.at[idx, "current_ps"] / median_ps)
                ratio = min(ratios) if ratios else None
                peer_score = score_peer_ratio(ratio)
                df.at[idx, "score_peer_valuation"] = peer_score
                df.at[idx, "value_score"] = (
                    df.at[idx, "score_hist_valuation"]
                    + df.at[idx, "score_abs_valuation"]
                    + peer_score
                    + df.at[idx, "score_peg"]
                    + df.at[idx, "score_growth_quality"]
                    + df.at[idx, "score_balance_sheet"]
                    + df.at[idx, "score_shareholder_return"]
                    + df.at[idx, "score_support"]
                    + df.at[idx, "score_sentiment"]
                )
                df.at[idx, "value_score_tier"] = classify_score(df.at[idx, "value_score"])
        return df

    summary_df = apply_peer_scores(summary_df)
    support_levels = (
        combined_df.sort_values("date")
        .groupby("symbol")[["support_level", "support_level_secondary"]]
        .last()
        .rename(
            columns={
                "support_level": "support_level_primary",
                "support_level_secondary": "support_level_secondary",
            }
        )
    )
    summary_df = summary_df.merge(support_levels, on="symbol", how="left")
    change_7d = trailing_change_map(combined_df, 7)
    change_30d = trailing_change_map(combined_df, 30)
    summary_df["pct_change_7d"] = summary_df["symbol"].map(change_7d)
    summary_df["pct_change_30d"] = summary_df["symbol"].map(change_30d)
    summary_df["name"] = summary_df["name_en"] + "（" + summary_df["name_cn"] + "）"
    summary_path = OUTPUT_DIR / "three_month_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    radar_df = build_opportunity_radar(summary_df, combined_df, RADAR_CONFIG)
    radar_path = OUTPUT_DIR / "equity_opportunity_radar.csv"
    radar_df.to_csv(radar_path, index=False)



    print("Saved:", combined_path)
    print("Saved:", summary_path)
    print("Saved:", radar_path)
    if failures:
        print("Warnings:")
        for msg in failures:
            print(" -", msg)


if __name__ == "__main__":
    main()
