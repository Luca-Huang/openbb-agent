"""AkShare-backed data source for A-share financial statements.

Uses AKTools (https://github.com/akfamily/aktools) as an HTTP wrapper around
the ``akshare`` Python library. Run the service locally with::

    python -m aktools

and configure the URL via the ``AKTOOLS_URL`` env var (default
``http://127.0.0.1:8080``). Domestic data hosts must bypass any HTTP(S) proxy
on the AKTools side; the helper here only talks to the local AKTools service.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("data_sources.akshare_cn")

DEFAULT_AKTOOLS_URL = "http://127.0.0.1:8080"
SUPPORTED_MARKETS = {"CN"}
DEFAULT_HTTP_TIMEOUT_SECONDS = 30


class AKToolsError(RuntimeError):
    """Raised when the AKTools HTTP service cannot return usable data."""


def _aktools_base_url() -> str:
    return os.environ.get("AKTOOLS_URL", DEFAULT_AKTOOLS_URL).rstrip("/")


def _akshare_sina_symbol(symbol: str) -> str:
    """Convert canonical ``002624.SZ`` to akshare sina format ``sz002624``."""
    if "." not in symbol:
        raise ValueError(f"Symbol must be <CODE>.<MARKET>, got: {symbol!r}")
    code, market = symbol.split(".", 1)
    return f"{market.lower()}{code}"


def _retry_get_json(
    path: str,
    params: dict[str, Any],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS,
) -> Any:
    """GET JSON from AKTools with exponential-backoff retries on transient errors.

    The upstream eastmoney/sina endpoints AKTools wraps occasionally drop
    connections; a couple of retries absorb the flap.
    """
    url = f"{_aktools_base_url()}{path}"
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout_seconds)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if i < attempts - 1:
                delay = base_delay * (2 ** i)
                log.warning(
                    "AKTools %s failed (attempt %d/%d): %s; retrying in %.1fs",
                    path, i + 1, attempts, exc, delay,
                )
                time.sleep(delay)
    raise AKToolsError(
        f"AKTools call to {path} failed after {attempts} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Sina financial statement field mappings (Chinese -> canonical English)
# ---------------------------------------------------------------------------

_INCOME_RENAME = {
    "报告日": "fiscal_period",
    "营业总收入": "revenue",
    "营业收入": "revenue_main",
    "营业成本": "operating_cost",
    "营业利润": "operating_profit",
    "利润总额": "total_profit",
    "净利润": "net_income",
    "归属于母公司所有者的净利润": "net_income_parent",
    "基本每股收益": "basic_eps",
    "稀释每股收益": "diluted_eps",
}

_BALANCE_RENAME = {
    "报告日": "fiscal_period",
    "资产总计": "total_assets",
    "负债合计": "total_liabilities",
    "归属于母公司股东权益合计": "equity_parent",
    "所有者权益(或股东权益)合计": "total_equity",
    "货币资金": "cash_and_equivalents",
    "短期借款": "short_term_debt",
    "长期借款": "long_term_debt",
    "应付债券": "bonds_payable",
    "存货": "inventory",
}

_CASHFLOW_RENAME = {
    "报告日": "fiscal_period",
    "经营活动产生的现金流量净额": "operating_cash_flow_net",
    "投资活动产生的现金流量净额": "investing_cash_flow_net",
    "筹资活动产生的现金流量净额": "financing_cash_flow_net",
    "购建固定资产、无形资产和其他长期资产所支付的现金": "capex",
}


def _fetch_sina_statement(
    sina_symbol: str,
    statement_label: str,
    rename: dict[str, str],
) -> pd.DataFrame:
    """Fetch one of the three Sina statements via AKTools and normalize columns."""
    rows = _retry_get_json(
        "/api/public/stock_financial_report_sina",
        {"stock": sina_symbol, "symbol": statement_label},
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = [c for c in rename if c in df.columns]
    df = df[keep].rename(columns=rename)
    df["fiscal_period"] = pd.to_datetime(
        df["fiscal_period"].astype(str), format="%Y%m%d", errors="coerce"
    )
    for col in df.columns:
        if col != "fiscal_period":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["fiscal_period"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AKShareCNProvider:
    """A-share deep financial data via the local AKTools HTTP service.

    Scope is intentionally narrow: this provider fills the gaps that Longbridge
    does not cover for A-shares (annual financial statements, eventually
    dividends, share-unlock calendars, north-bound holdings, etc.). Daily
    quotes / OHLCV stay with ``data_sources.longbridge``.
    """

    market = "CN"

    def __init__(self, market: str = "CN"):
        market = market.upper()
        if market not in SUPPORTED_MARKETS:
            raise ValueError(
                f"AKShareCNProvider only supports {sorted(SUPPORTED_MARKETS)}, "
                f"got {market!r}"
            )

    def fetch_annual_financials(self, symbol: str) -> pd.DataFrame:
        """Fetch IS + BS + CF as one annual-frequency DataFrame.

        Rows are filtered to fiscal-year-end (12-31) reports only; quarterly
        periods are dropped. Statements are outer-merged on ``fiscal_period``,
        so each row holds all available IS/BS/CF fields for that year. A few
        derived ratios (net margin, asset-liability ratio, free cash flow)
        are pre-computed for convenience.
        """
        log.info("AKShare CN: fetching annual financials for %s", symbol)
        sina_symbol = _akshare_sina_symbol(symbol)

        income = _fetch_sina_statement(sina_symbol, "利润表", _INCOME_RENAME)
        balance = _fetch_sina_statement(sina_symbol, "资产负债表", _BALANCE_RENAME)
        cashflow = _fetch_sina_statement(sina_symbol, "现金流量表", _CASHFLOW_RENAME)

        if income.empty:
            log.warning("AKShare CN: no income statement returned for %s", symbol)
            return pd.DataFrame()

        df = income
        for other in (balance, cashflow):
            if not other.empty:
                df = df.merge(other, on="fiscal_period", how="outer")

        # Annual only (fiscal-year-end is 12-31 for CN A-shares).
        df = df[df["fiscal_period"].dt.month == 12].copy()
        if df.empty:
            return df

        # Derived ratios — computed only when both operands are present.
        # Division by zero yields ±inf; coerce to NaN so callers can skip cleanly.
        def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
            return (num / den).replace([np.inf, -np.inf], np.nan)

        if {"net_income_parent", "revenue"}.issubset(df.columns):
            df["net_margin"] = _safe_div(df["net_income_parent"], df["revenue"])
        if {"total_liabilities", "total_assets"}.issubset(df.columns):
            df["asset_liability_ratio"] = _safe_div(df["total_liabilities"], df["total_assets"])
        if {"operating_cash_flow_net", "capex"}.issubset(df.columns):
            df["free_cash_flow"] = df["operating_cash_flow_net"] - df["capex"]
        if {"net_income_parent", "equity_parent"}.issubset(df.columns):
            df["roe_simple"] = _safe_div(df["net_income_parent"], df["equity_parent"])

        df["symbol"] = symbol.upper()
        df["data_source"] = "akshare_sina"
        df["fiscal_year"] = df["fiscal_period"].dt.year.astype("Int64")

        return df.sort_values("fiscal_period", ascending=False).reset_index(drop=True)


def get_provider(market: str = "CN") -> AKShareCNProvider:
    """Return an AKShareCNProvider — only ``CN`` is supported."""
    return AKShareCNProvider(market)
