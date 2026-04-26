#!/usr/bin/env python3
"""A-Share Data Ingestion Script (Local-first).

Fetches daily price history and basic fundamental data for symbols in `watchlist_cn.json`
using `yfinance`, computes technical indicators and basic value scores, and saves to CSVs.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import os

# Disable proxy BEFORE importing requests/akshare/yfinance
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)
os.environ["NO_PROXY"] = "*"

import yfinance as yf
import akshare as ak
import urllib.request

# Force Python's requests library to ignore Windows Registry proxy settings
urllib.request.getproxies = lambda: {}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("fetch_cn")

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "research_inputs"
OUTPUT_DIR = ROOT_DIR / "openbb_outputs"

WATCHLIST_PATH = INPUT_DIR / "watchlist_cn.json"
HISTORY_PATH = OUTPUT_DIR / "three_month_close_history.csv"
SUMMARY_PATH = OUTPUT_DIR / "three_month_summary.csv"

# Fetch 2 years of history for percentiles and 200ma
LOOKBACK_DAYS = 730


def load_cn_watchlist() -> list[dict]:
    if not WATCHLIST_PATH.exists():
        log.warning(f"Watchlist not found: {WATCHLIST_PATH}")
        return []
    with WATCHLIST_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    # Ensure symbols have proper yahoo finance suffixes if not present
    # e.g. 000001.SZ -> 000001.SZ
    return [item for item in data if item.get("market") == "CN" and item.get("status") != "ignore"]


def fetch_akshare_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    # Convert '002241.SZ' to '002241' for akshare
    ak_symbol = symbol.split('.')[0]
    log.info(f"Fetching history for {symbol} via akshare...")
    try:
        df = ak.stock_zh_a_hist(
            symbol=ak_symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq"
        )
    except Exception as e:
        log.error(f"akshare fetch failed for {symbol}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()
        
    # akshare columns: 日期, 股票代码, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume"
    })
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    
    df["close_norm"] = df["close"] / df["close"].iloc[0]
    return df[["date", "symbol", "open", "high", "low", "close", "volume", "close_norm"]]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("date").copy()
    df["ma50"] = df["close"].rolling(window=50, min_periods=1).mean()
    df["ma200"] = df["close"].rolling(window=200, min_periods=1).mean()
    df["rsi14"] = compute_rsi(df["close"], period=14)
    
    # Support levels
    rolling_min = df["close"].rolling(window=20, min_periods=1).min()
    df["support_level_primary"] = rolling_min
    df["support_level_secondary"] = rolling_min * 1.1
    
    # Volume metrics
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df["volume_ma20"] = df["volume"].rolling(window=20, min_periods=1).mean()
    df["volume_spike_ratio"] = df["volume"] / df["volume_ma20"]
    
    # Percentile mapping
    closes = np.sort(df["close"].dropna().to_numpy())
    if len(closes) > 0:
        df["close_percentile"] = df["close"].map(
            lambda x: np.searchsorted(closes, x, side="right") / len(closes) if pd.notna(x) else np.nan
        )
    else:
        df["close_percentile"] = np.nan
        
    return df


def fetch_fundamentals(symbol: str) -> dict:
    log.info(f"Fetching fundamentals for {symbol} via yfinance...")
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info
        fast = getattr(ticker, "fast_info", {})
    except Exception as e:
        log.warning(f"yfinance fundamentals fetch failed for {symbol}: {e}")
        info = {}
        fast = {}
        
    pe = info.get("trailingPE", info.get("forwardPE", np.nan))
    ps = info.get("priceToSalesTrailing12Months", np.nan)
    fcf = info.get("freeCashflow", np.nan)
    mcap = fast.get("marketCap", info.get("marketCap", np.nan))
    fcf_yield = fcf / mcap if pd.notna(fcf) and pd.notna(mcap) and mcap > 0 else np.nan
    
    total_cash = info.get("totalCash", 0)
    total_debt = info.get("totalDebt", 0)
    
    dividend = info.get("dividendYield", 0)
    
    return {
        "end_pe": pe,
        "current_ps": ps,
        "fcf_yield": fcf_yield,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "dividend_yield": dividend,
    }


def score_fundamentals(fund: dict) -> dict:
    pe = fund.get("end_pe")
    ps = fund.get("current_ps")
    fcf = fund.get("fcf_yield")
    
    # Simple Valuation
    score_valuation = 0.0
    if pd.notna(pe):
        if pe < 15: score_valuation += 10
        elif pe < 25: score_valuation += 5
    if pd.notna(ps):
        if ps < 2: score_valuation += 10
        elif ps < 5: score_valuation += 5
    if pd.notna(fcf) and fcf > 0.05:
        score_valuation += 10
        
    # Simple Balance Sheet
    score_bs = 0.0
    cash = fund.get("total_cash", 0)
    debt = fund.get("total_debt", 0)
    if cash > debt:
        score_bs += 10
    elif debt <= cash * 2:
        score_bs += 5
        
    # Shareholder Return
    score_ret = 10.0 if fund.get("dividend_yield", 0) > 0.02 else 0.0
    
    return {
        "score_abs_valuation": min(20.0, score_valuation),
        "score_balance_sheet": score_bs,
        "score_shareholder_return": score_ret,
        # Default others to 0 to keep signal engine happy
        "score_hist_valuation": 0.0,
        "score_peer_valuation": 0.0,
        "score_peg": 0.0,
        "score_growth_quality": 0.0,
        "score_sentiment": 5.0,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    watchlist = load_cn_watchlist()
    if not watchlist:
        log.error("No valid CN symbols in watchlist.")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    
    all_history = []
    summary_rows = []
    
    for item in watchlist:
        symbol = item["symbol"]
        hist_df = fetch_akshare_history(symbol, start_date, end_date)
        if hist_df.empty:
            log.warning(f"No history data for {symbol}, skipping.")
            continue
            
        hist_df = add_technical_indicators(hist_df)
        all_history.append(hist_df)
        
        fund_data = fetch_fundamentals(symbol)
        scores = score_fundamentals(fund_data)
        
        # Calculate trailing return
        latest_close = hist_df["close"].iloc[-1]
        start_close = hist_df["close"].iloc[0]
        pct_change = (latest_close - start_close) / start_close if start_close > 0 else 0
        
        value_score = sum(scores.values())
        
        summary_row = {
            "symbol": symbol,
            "name_cn": item.get("name", ""),
            "name_en": "",
            "market": "CN",
            "theme_category": item.get("sector", ""),
            "end_close": latest_close,
            "pct_change": pct_change,
            "value_score": value_score,
            "value_score_tier": "合理区" if value_score > 30 else "观望",
            **fund_data,
            **scores,
        }
        summary_rows.append(summary_row)

    if all_history:
        combined_hist = pd.concat(all_history, ignore_index=True)
        combined_hist.to_csv(HISTORY_PATH, index=False)
        log.info(f"Saved {len(combined_hist)} history rows to {HISTORY_PATH.name}")
        
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        # Ensure target_zone_low / high is kept if we wanted to merge, 
        # but radar.py merges watchlist_df anyway, so summary doesn't need it.
        summary_df.to_csv(SUMMARY_PATH, index=False)
        log.info(f"Saved {len(summary_df)} summary rows to {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
