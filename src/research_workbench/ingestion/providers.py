"""Data providers for multi-market stock data ingestion.

Provides a unified interface for fetching history and fundamentals data
across A-share, Hong Kong, and US markets.

Usage::

    provider = get_provider("CN")
    history = provider.fetch_history("002241.SZ", start, end)
    fundamentals = provider.fetch_fundamentals("002241.SZ")
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("ingestion.providers")

# ---------------------------------------------------------------------------
# Proxy routing
# ---------------------------------------------------------------------------
# Domestic A-share endpoints (AKShare's upstreams) must bypass any HTTP(S)
# proxy: from a mainland network the proxy adds latency and flakiness, and a
# flapping proxy node breaks the whole pipeline. Overseas providers
# (yfinance → Yahoo) keep using the ambient proxy, which is required to reach
# Yahoo from the mainland — so we only force *domestic* hosts direct rather
# than disabling proxying globally.
_DOMESTIC_NO_PROXY: tuple[str, ...] = (
    "eastmoney.com", ".eastmoney.com",
    "push2.eastmoney.com", "push2his.eastmoney.com",
    "datacenter-web.eastmoney.com", "datacenter.eastmoney.com",
    "quote.eastmoney.com",
    "sinajs.cn", ".sinajs.cn", "hq.sinajs.cn",
    "sina.com.cn", ".sina.com.cn",
    "xueqiu.com", ".xueqiu.com",
    "10jqka.com.cn", ".10jqka.com.cn",
    "legulegu.com", ".legulegu.com",
)


def _ensure_domestic_direct() -> None:
    """Add domestic data hosts to NO_PROXY so requests connects to them directly.

    Preserves any existing NO_PROXY entries and the ambient HTTP(S)_PROXY (which
    yfinance needs for Yahoo); drops a blunt ``*`` so overseas traffic still uses
    the proxy.
    """
    existing = os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", "")
    entries = [e.strip() for e in existing.split(",") if e.strip() and e.strip() != "*"]
    for host in _DOMESTIC_NO_PROXY:
        if host not in entries:
            entries.append(host)
    value = ",".join(entries)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


_ensure_domestic_direct()


def _retry_call(fn, *, attempts: int = 3, base_delay: float = 1.0, what: str = ""):
    """Call ``fn`` with exponential backoff on transient network failures.

    Domestic data endpoints (eastmoney push2) intermittently reset connections,
    so a couple of retries absorb the flap before we fall back to another source.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — transient network errors → retry
            last_exc = e
            if i < attempts - 1:
                delay = base_delay * (2 ** i)
                log.warning(
                    "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                    what or "call", i + 1, attempts, e, delay,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Technical indicator helpers (shared by all providers)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard technical indicators to a OHLCV dataframe.

    Expected input columns: date, symbol, open, high, low, close, volume.
    Output adds: ma20, ma50, ma200, rsi14, support_level_primary,
    support_level_secondary, volume_ma20, volume_spike_ratio, close_percentile.
    """
    if df.empty:
        return df
    df = df.sort_values("date").copy()
    close = pd.to_numeric(df["close"], errors="coerce")

    # Moving averages
    df["ma20"] = close.rolling(window=20, min_periods=1).mean()
    df["ma50"] = close.rolling(window=50, min_periods=1).mean()
    df["ma200"] = close.rolling(window=200, min_periods=1).mean()

    # RSI
    df["rsi14"] = compute_rsi(close, period=14)

    # Support levels
    rolling_min = close.rolling(window=20, min_periods=1).min()
    df["support_level_primary"] = rolling_min
    df["support_level_secondary"] = rolling_min * 1.1

    # Volume metrics
    volume = pd.to_numeric(df.get("volume"), errors="coerce")
    df["volume_ma20"] = volume.rolling(window=20, min_periods=1).mean()
    df["volume_spike_ratio"] = volume / df["volume_ma20"]

    # Percentile mapping
    closes = np.sort(close.dropna().to_numpy())
    if len(closes) > 0:
        df["close_percentile"] = close.map(
            lambda x: np.searchsorted(closes, x, side="right") / len(closes)
            if pd.notna(x) else np.nan
        )
    else:
        df["close_percentile"] = np.nan

    # Normalized close
    first_valid = close.dropna()
    if not first_valid.empty:
        df["close_norm"] = close / first_valid.iloc[0]
    else:
        df["close_norm"] = np.nan

    return df


def score_fundamentals(fund: dict) -> dict:
    """Compute simple valuation/quality scores from fundamental data."""
    pe = fund.get("end_pe")
    ps = fund.get("current_ps")
    fcf = fund.get("fcf_yield")

    score_valuation = 0.0
    if pd.notna(pe):
        if pe < 15:
            score_valuation += 10
        elif pe < 25:
            score_valuation += 5
    if pd.notna(ps):
        if ps < 2:
            score_valuation += 10
        elif ps < 5:
            score_valuation += 5
    if pd.notna(fcf) and fcf > 0.05:
        score_valuation += 10

    score_bs = 0.0
    cash = fund.get("total_cash", 0) or 0
    debt = fund.get("total_debt", 0) or 0
    if cash > debt:
        score_bs += 10
    elif debt <= cash * 2:
        score_bs += 5

    score_ret = 10.0 if (fund.get("dividend_yield") or 0) > 0.02 else 0.0

    return {
        "score_abs_valuation": min(20.0, score_valuation),
        "score_balance_sheet": score_bs,
        "score_shareholder_return": score_ret,
        "score_hist_valuation": 0.0,
        "score_peer_valuation": 0.0,
        "score_peg": 0.0,
        "score_growth_quality": 0.0,
        "score_sentiment": 5.0,
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """Abstract base for multi-market data providers."""

    market: str = ""

    @abstractmethod
    def fetch_history(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        """Fetch OHLCV history. Returns DataFrame with columns:
        date, symbol, open, high, low, close, volume.
        """
        ...

    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        """Fetch fundamental data. Returns dict with keys like:
        end_pe, current_ps, fcf_yield, total_cash, total_debt, dividend_yield.
        """
        ...


# ---------------------------------------------------------------------------
# A-share via AKShare
# ---------------------------------------------------------------------------

class AKShareCNProvider(DataProvider):
    """A-share data via AKShare (东方财富/新浪 source).

    Falls back to yfinance when AKShare fails or returns empty — yfinance
    natively understands ``002602.SZ`` / ``600519.SH`` style tickers and
    works from networks where AKShare's upstream endpoints are blocked
    or flaky.
    """

    market = "CN"

    def fetch_history(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        df = self._fetch_via_akshare(symbol, start, end)
        if not df.empty:
            return df
        log.warning("AKShare returned no data for %s, falling back to yfinance", symbol)
        return self._fetch_via_yfinance(symbol, start, end)

    @staticmethod
    def _fetch_via_akshare(symbol: str, start: date, end: date) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError:
            log.warning("akshare not installed; skipping primary fetch path")
            return pd.DataFrame()

        ak_symbol = symbol.split(".")[0]
        log.info("AKShare: fetching history for %s (%s → %s)", symbol, start, end)
        try:
            df = _retry_call(
                lambda: ak.stock_zh_a_hist(
                    symbol=ak_symbol,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="qfq",
                ),
                attempts=3,
                base_delay=1.0,
                what=f"AKShare {symbol}",
            )
        except Exception as e:  # noqa: BLE001 — exhausted retries → fall through to yfinance
            log.warning("AKShare fetch failed for %s after retries: %s", symbol, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        time.sleep(1)  # rate limiting
        return df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    @staticmethod
    def _fetch_via_yfinance(symbol: str, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed; cannot fall back")
            return pd.DataFrame()

        log.info("yfinance fallback: fetching history for %s (%s → %s)", symbol, start, end)
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
        except Exception as e:  # noqa: BLE001
            log.error("yfinance fallback failed for %s: %s", symbol, e)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        time.sleep(0.5)
        return df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    def fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        """Fetch fundamentals via yfinance for A-share symbols."""
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed, returning empty fundamentals")
            return {}

        log.info("yfinance: fetching fundamentals for %s", symbol)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            fast = getattr(ticker, "fast_info", {})
        except Exception as e:
            log.warning("yfinance fundamentals failed for %s: %s", symbol, e)
            info = {}
            fast = {}

        pe = info.get("trailingPE", info.get("forwardPE"))
        ps = info.get("priceToSalesTrailing12Months")
        fcf = info.get("freeCashflow")
        mcap = fast.get("marketCap", info.get("marketCap"))
        fcf_yield = (
            fcf / mcap if fcf and mcap and mcap > 0 else None
        )

        return {
            "end_pe": pe,
            "current_ps": ps,
            "fcf_yield": fcf_yield,
            "total_cash": info.get("totalCash", 0),
            "total_debt": info.get("totalDebt", 0),
            "dividend_yield": info.get("dividendYield", 0),
        }


# ---------------------------------------------------------------------------
# HK / US via yfinance
# ---------------------------------------------------------------------------

class YFinanceProvider(DataProvider):
    """HK and US market data via yfinance."""

    def __init__(self, market: str = "US"):
        self.market = market

    def fetch_history(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed")
            return pd.DataFrame()

        log.info("yfinance: fetching history for %s (%s → %s)", symbol, start, end)
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
        except Exception as e:
            log.error("yfinance fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        time.sleep(0.5)  # rate limiting
        return df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    def fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        try:
            import yfinance as yf
        except ImportError:
            return {}

        log.info("yfinance: fetching fundamentals for %s", symbol)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            fast = getattr(ticker, "fast_info", {})
        except Exception as e:
            log.warning("yfinance fundamentals failed for %s: %s", symbol, e)
            return {}

        pe = info.get("trailingPE", info.get("forwardPE"))
        ps = info.get("priceToSalesTrailing12Months")
        fcf = info.get("freeCashflow")
        mcap = fast.get("marketCap", info.get("marketCap"))
        fcf_yield = fcf / mcap if fcf and mcap and mcap > 0 else None

        return {
            "end_pe": pe,
            "current_ps": ps,
            "fcf_yield": fcf_yield,
            "total_cash": info.get("totalCash", 0),
            "total_debt": info.get("totalDebt", 0),
            "dividend_yield": info.get("dividendYield", 0),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[DataProvider]] = {
    "CN": AKShareCNProvider,
    "HK": type("YFinanceHKProvider", (YFinanceProvider,), {"__init__": lambda self: YFinanceProvider.__init__(self, "HK")}),
    "US": type("YFinanceUSProvider", (YFinanceProvider,), {"__init__": lambda self: YFinanceProvider.__init__(self, "US")}),
}


def get_provider(market: str) -> DataProvider:
    """Get a data provider for the given market (CN / HK / US)."""
    market = market.upper()
    cls = _PROVIDERS.get(market)
    if cls is None:
        raise ValueError(f"No provider for market: {market}. Available: {list(_PROVIDERS.keys())}")
    return cls()


LOOKBACK_DAYS = 730  # 2 years of history for percentiles and 200MA
