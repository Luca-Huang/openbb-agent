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
DEFAULT_HTTP_TIMEOUT_SECONDS = 60
# Market-wide endpoints (full-table downloads) need a longer ceiling.
BULK_HTTP_TIMEOUT_SECONDS = 120


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

    def fetch_financials_all(self, symbol: str) -> pd.DataFrame:
        """Fetch IS + BS + CF for **all** periods (annual + quarterly).

        Returns rows for every fiscal_period the Sina endpoint exposes,
        statements outer-merged. Use :meth:`fetch_annual_financials` for the
        annual-only subset (12-31 rows) when scoring; use the all-periods
        view for sequential / cadence analysis.
        """
        log.info("AKShare CN: fetching all-period financials for %s", symbol)
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

        if df.empty:
            return df

        # Derived ratios — computed only when both operands are present.
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
        # Derive report_type from fiscal-period month: 3 -> q1, 6 -> h1,
        # 9 -> q3 (3-quarter cumulative), 12 -> annual.
        month_map = {3: "q1", 6: "h1", 9: "q3", 12: "annual"}
        df["report_type"] = df["fiscal_period"].dt.month.map(month_map).fillna("other")

        return df.sort_values("fiscal_period", ascending=False).reset_index(drop=True)

    def fetch_annual_financials(self, symbol: str) -> pd.DataFrame:
        """Annual-only subset of :meth:`fetch_financials_all` (12-31 rows).

        Kept as the canonical input to ``analysis.summary`` scoring, which
        depends on year-over-year comparisons across calendar years.
        """
        df = self.fetch_financials_all(symbol)
        if df.empty:
            return df
        return df[df["fiscal_period"].dt.month == 12].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Historical valuation (PE/PB/PS daily series) — for hist-valuation score
    # ------------------------------------------------------------------

    def fetch_historical_valuation(self, symbol: str) -> pd.DataFrame:
        """Fetch daily PE/PB/PS series from eastmoney 'stock value'.

        Returns columns: date, close, pe_ttm, pe_static, pb, peg, ps, market_cap.
        Coverage typically starts 2018 (~2000+ trading days).
        """
        log.info("AKShare CN: fetching historical valuation for %s", symbol)
        code = symbol.split(".", 1)[0]
        rows = _retry_get_json("/api/public/stock_value_em", {"symbol": code})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        rename = {
            "数据日期": "date",
            "当日收盘价": "close",
            "总市值": "market_cap",
            "PE(TTM)": "pe_ttm",
            "PE(静)": "pe_static",
            "市净率": "pb",
            "PEG值": "peg",
            "市现率": "pcf",
            "市销率": "ps",
        }
        keep = [c for c in rename if c in df.columns]
        df = df[keep].rename(columns=rename)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["symbol"] = symbol.upper()
        df["data_source"] = "akshare_em"
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Earnings preannouncement — input to PEG (forward EPS growth)
    # ------------------------------------------------------------------

    _YJYG_RENAME = {
        "股票代码": "code",
        "股票简称": "name",
        "预测指标": "indicator",
        "业绩变动": "change_text",
        "预测数值": "forecast_value",
        "业绩变动幅度": "yoy_pct",
        "预告类型": "forecast_type",
        "上年同期值": "prior_year_value",
        "公告日期": "announce_date",
    }

    @staticmethod
    def _normalize_yjyg(df: pd.DataFrame) -> pd.DataFrame:
        """Rename cols, parse numerics, attach canonical symbol."""
        rename = AKShareCNProvider._YJYG_RENAME
        keep = [c for c in rename if c in df.columns]
        df = df[keep].rename(columns=rename)
        for col in ("forecast_value", "yoy_pct", "prior_year_value"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "announce_date" in df.columns:
            df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
        if "code" in df.columns:
            # Map 6-digit code to canonical .SZ/.SH suffix.
            def _suffix(code: str) -> str:
                code = str(code)
                if code.startswith(("0", "2", "3")):
                    return f"{code}.SZ"
                if code.startswith(("5", "6", "9")):
                    return f"{code}.SH"
                if code.startswith("8"):
                    return f"{code}.BJ"
                return code
            df["symbol"] = df["code"].map(_suffix)
        return df

    def fetch_earnings_forecast_market(self, report_date: str) -> pd.DataFrame:
        """Fetch the FULL market preannouncement table for ``report_date``.

        Use this once per refresh and filter per-symbol downstream — calling
        the per-symbol variant N times re-downloads the same 7000+-row table
        N times. ``report_date`` like ``20251231``.
        """
        log.info("AKShare CN: fetching market yjyg table @ %s", report_date)
        rows = _retry_get_json(
            "/api/public/stock_yjyg_em",
            {"date": report_date},
            timeout_seconds=BULK_HTTP_TIMEOUT_SECONDS,
        )
        if not rows:
            return pd.DataFrame()
        df = self._normalize_yjyg(pd.DataFrame(rows))
        df["fiscal_period"] = pd.to_datetime(report_date, format="%Y%m%d", errors="coerce")
        return df.reset_index(drop=True)

    def fetch_earnings_preannouncement(
        self,
        symbol: str,
        report_date: str,
        market_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Filter the market preannouncement table to a single ``symbol``.

        If ``market_df`` is supplied (already fetched via
        :meth:`fetch_earnings_forecast_market`), it's filtered locally. Otherwise
        this fetches the full table for ``report_date`` — fine for ad-hoc one-off
        queries, but wasteful inside a multi-symbol pipeline loop.
        """
        if market_df is None:
            market_df = self.fetch_earnings_forecast_market(report_date)
        if market_df.empty or "code" not in market_df.columns:
            return pd.DataFrame()
        code = symbol.split(".", 1)[0]
        match = market_df[market_df["code"].astype(str) == code].copy()
        if match.empty:
            return match
        match["symbol"] = symbol.upper()
        match["fiscal_period"] = pd.to_datetime(report_date, format="%Y%m%d", errors="coerce")
        return match.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Earnings express (yjbb) — preliminary results before formal report
    # ------------------------------------------------------------------

    _YJBB_RENAME = {
        "股票代码": "code",
        "股票简称": "name",
        "每股收益": "eps",
        "营业总收入-营业总收入": "revenue",
        "营业总收入-同比增长": "revenue_yoy_pct",
        "营业总收入-季度环比增长": "revenue_qoq_pct",
        "净利润-净利润": "net_income",
        "净利润-同比增长": "net_income_yoy_pct",
        "净利润-季度环比增长": "net_income_qoq_pct",
        "每股净资产": "bps",
        "净资产收益率": "roe_pct",
        "每股经营现金流量": "ocf_per_share",
        "销售毛利率": "gross_margin_pct",
        "所处行业": "industry",
        "最新公告日期": "announce_date",
    }

    @staticmethod
    def _normalize_yjbb(df: pd.DataFrame) -> pd.DataFrame:
        rename = AKShareCNProvider._YJBB_RENAME
        keep = [c for c in rename if c in df.columns]
        df = df[keep].rename(columns=rename)
        numeric_cols = (
            "eps", "revenue", "revenue_yoy_pct", "revenue_qoq_pct",
            "net_income", "net_income_yoy_pct", "net_income_qoq_pct",
            "bps", "roe_pct", "ocf_per_share", "gross_margin_pct",
        )
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "announce_date" in df.columns:
            df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
        if "code" in df.columns:
            def _suffix(code: str) -> str:
                code = str(code)
                if code.startswith(("0", "2", "3")):
                    return f"{code}.SZ"
                if code.startswith(("5", "6", "9")):
                    return f"{code}.SH"
                if code.startswith("8"):
                    return f"{code}.BJ"
                return code
            df["symbol"] = df["code"].map(_suffix)
        return df

    def fetch_earnings_express_market(self, report_date: str) -> pd.DataFrame:
        """Fetch the FULL market 业绩快报 table for ``report_date``.

        Each row is one stock's preliminary results for that fiscal period
        (typically released 1-4 weeks before the formal annual / quarterly
        report). ``report_date`` like ``20251231``. Use this once per refresh
        and filter per symbol with :meth:`fetch_earnings_express`.
        """
        log.info("AKShare CN: fetching market yjbb table @ %s", report_date)
        rows = _retry_get_json(
            "/api/public/stock_yjbb_em",
            {"date": report_date},
            timeout_seconds=BULK_HTTP_TIMEOUT_SECONDS,
        )
        if not rows:
            return pd.DataFrame()
        df = self._normalize_yjbb(pd.DataFrame(rows))
        df["fiscal_period"] = pd.to_datetime(report_date, format="%Y%m%d", errors="coerce")
        return df.reset_index(drop=True)

    def fetch_earnings_express(
        self,
        symbol: str,
        report_date: str,
        market_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Slice the market 业绩快报 table down to a single ``symbol``.

        Same batch-vs-ad-hoc tradeoff as :meth:`fetch_earnings_preannouncement`.
        """
        if market_df is None:
            market_df = self.fetch_earnings_express_market(report_date)
        if market_df.empty or "code" not in market_df.columns:
            return pd.DataFrame()
        code = symbol.split(".", 1)[0]
        match = market_df[market_df["code"].astype(str) == code].copy()
        if match.empty:
            return match
        match["symbol"] = symbol.upper()
        return match.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Industry peer snapshot — for peer-valuation score
    # ------------------------------------------------------------------

    def fetch_industry_classification(
        self,
        symbol: str,
        yjbb_market_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Return the industry name and listing exchange for ``symbol``.

        Primary source is ``stock_individual_info_em`` (push2.eastmoney). That
        endpoint is intermittently RST-prone from mainland networks, so this
        falls back to the ``industry`` column of the market-wide
        :meth:`fetch_earnings_express_market` table when a ``yjbb_market_df``
        is supplied — the fallback only fills the industry name, not the rest.
        """
        log.info("AKShare CN: fetching industry classification for %s", symbol)
        code = symbol.split(".", 1)[0]
        try:
            rows = _retry_get_json(
                "/api/public/stock_individual_info_em",
                {"symbol": code},
            )
        except AKToolsError as exc:
            log.warning("stock_individual_info_em failed for %s: %s", symbol, exc)
            rows = None

        if rows:
            # akshare returns a list of {item, value} rows
            out: dict[str, Any] = {}
            for row in rows:
                item = row.get("item") or row.get("项目") or row.get("indicator")
                value = row.get("value") or row.get("值") or row.get("VALUE")
                if item:
                    out[str(item)] = value
            return {
                "industry": out.get("行业") or out.get("INDUSTRY") or "",
                "listing_date": out.get("上市时间") or "",
                "total_shares": out.get("总股本") or "",
                "circ_shares": out.get("流通股") or "",
                "data_source": "akshare_em_individual",
                "raw": out,
            }

        # Fallback: pluck the industry name out of the market yjbb table.
        if yjbb_market_df is not None and not yjbb_market_df.empty:
            if "code" in yjbb_market_df.columns and "industry" in yjbb_market_df.columns:
                match = yjbb_market_df[yjbb_market_df["code"].astype(str) == code]
                if not match.empty:
                    industry = match.iloc[0]["industry"]
                    if industry:
                        log.info("  industry fallback from yjbb: %s -> %s", symbol, industry)
                        return {
                            "industry": str(industry),
                            "listing_date": "",
                            "total_shares": "",
                            "circ_shares": "",
                            "data_source": "akshare_yjbb_fallback",
                            "raw": {},
                        }
        return {}

    def fetch_industry_peers(self, industry_name: str) -> pd.DataFrame:
        """List all A-share constituents of an eastmoney industry board.

        Returns columns: code, name, price, change_pct, market_cap, pe, pb, ...
        (whatever eastmoney returns for the board).
        """
        log.info("AKShare CN: fetching industry peers for %s", industry_name)
        rows = _retry_get_json(
            "/api/public/stock_board_industry_cons_em",
            {"symbol": industry_name},
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        rename = {
            "代码": "code",
            "名称": "name",
            "最新价": "last",
            "涨跌幅": "change_pct",
            "市盈率-动态": "pe_ttm",
            "市净率": "pb",
            "总市值": "market_cap",
            "流通市值": "circ_market_cap",
        }
        keep = [c for c in rename if c in df.columns]
        if not keep:
            return df  # return raw if shape changed
        df = df[keep].rename(columns=rename)
        for col in df.columns:
            if col not in ("code", "name"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # Northbound holding (沪深港通-北向资金) — daily time series since 2017
    # ------------------------------------------------------------------

    def fetch_northbound_holding(self, symbol: str) -> pd.DataFrame:
        """Fetch the daily northbound (HK-Connect inflow) holding series.

        Returns ~1500-2000 trading-day rows since 2017-03 with daily snapshots
        of foreign holding shares, value, % of A-shares outstanding, plus the
        net buy/sell delta for the day. This is one of the strongest sentiment
        signals on the A-share market (and not available from Longbridge).
        """
        log.info("AKShare CN: fetching northbound holdings for %s", symbol)
        code = symbol.split(".", 1)[0]
        rows = _retry_get_json(
            "/api/public/stock_hsgt_individual_em",
            {"symbol": code},
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        rename = {
            "持股日期": "date",
            "当日收盘价": "close",
            "当日涨跌幅": "change_pct",
            "持股数量": "north_shares",
            "持股市值": "north_value",
            "持股数量占A股百分比": "north_pct_of_outstanding",
            "今日增持股数": "north_net_buy_shares",
            "今日增持资金": "north_net_buy_cny",
            "今日持股市值变化": "north_value_change",
        }
        keep = [c for c in rename if c in df.columns]
        df = df[keep].rename(columns=rename)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["symbol"] = symbol.upper()
        df["data_source"] = "akshare_em"
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Business segments (主营构成) — SOTP valuation input
    # ------------------------------------------------------------------

    def fetch_business_segments(self, symbol: str) -> pd.DataFrame:
        """Fetch the multi-year 主营构成 breakdown (按产品 / 按地区 / 汇总).

        Returns long-form: each row is one (fiscal_period, classification,
        segment) triple with revenue / cost / profit / margin numbers. Use
        for SOTP valuation: aggregate revenue / profit per segment, apply
        per-segment PE multiples.
        """
        log.info("AKShare CN: fetching business segments for %s", symbol)
        code = symbol.split(".", 1)[0]
        suffix = symbol.split(".", 1)[1].upper() if "." in symbol else ""
        # stock_zygc_em wants market-prefixed symbol: 'SZ002624' / 'SH600519'
        prefixed = f"{suffix}{code}" if suffix in ("SZ", "SH", "BJ") else code
        rows = _retry_get_json(
            "/api/public/stock_zygc_em",
            {"symbol": prefixed},
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        rename = {
            "股票代码": "code",
            "报告日期": "fiscal_period",
            "分类类型": "classification",
            "主营构成": "segment",
            "主营收入": "revenue",
            "收入比例": "revenue_ratio",
            "主营成本": "cost",
            "成本比例": "cost_ratio",
            "主营利润": "profit",
            "利润比例": "profit_ratio",
            "毛利率": "gross_margin",
        }
        keep = [c for c in rename if c in df.columns]
        df = df[keep].rename(columns=rename)
        df["fiscal_period"] = pd.to_datetime(df["fiscal_period"], errors="coerce")
        numeric = ("revenue", "revenue_ratio", "cost", "cost_ratio",
                   "profit", "profit_ratio", "gross_margin")
        for col in numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # classification can be NaN for the top-level (合计) rows.
        if "classification" in df.columns:
            df["classification"] = df["classification"].fillna("合计")
        df["symbol"] = symbol.upper()
        df["data_source"] = "akshare_em"
        return df.dropna(subset=["fiscal_period"]).sort_values(
            ["fiscal_period", "classification", "segment"], ascending=[False, True, True]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Events: dividends, shareholder changes
    # ------------------------------------------------------------------

    def fetch_dividend_history(self, symbol: str) -> pd.DataFrame:
        """Fetch dividend history (新浪) with announce / ex-div dates and per-share payout."""
        log.info("AKShare CN: fetching dividend history for %s", symbol)
        code = symbol.split(".", 1)[0]
        rows = _retry_get_json(
            "/api/public/stock_history_dividend_detail",
            {"symbol": code, "indicator": "分红"},
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        rename = {
            "公告日期": "announce_date",
            "送股": "bonus_shares",
            "转增": "transfer_shares",
            "派息": "cash_dividend",
            "进度": "status",
            "除权除息日": "ex_div_date",
            "股权登记日": "record_date",
            "红股上市日": "bonus_listing_date",
        }
        keep = [c for c in rename if c in df.columns]
        df = df[keep].rename(columns=rename)
        for col in ("announce_date", "ex_div_date", "record_date", "bonus_listing_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in ("bonus_shares", "transfer_shares", "cash_dividend"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["symbol"] = symbol.upper()
        return df.sort_values("announce_date", ascending=False, na_position="last").reset_index(drop=True)

    def fetch_shareholder_changes(self, symbol: str) -> pd.DataFrame:
        """Fetch major shareholder buy/sell records (同花顺)."""
        log.info("AKShare CN: fetching shareholder changes for %s", symbol)
        code = symbol.split(".", 1)[0]
        rows = _retry_get_json(
            "/api/public/stock_shareholder_change_ths",
            {"symbol": code},
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        rename = {
            "公告日期": "announce_date",
            "变动股东": "shareholder",
            "变动数量": "change_text",
            "交易均价": "avg_price",
            "剩余股份总数": "remaining_text",
            "变动期间": "change_period",
            "变动途径": "channel",
        }
        keep = [c for c in rename if c in df.columns]
        df = df[keep].rename(columns=rename)
        if "announce_date" in df.columns:
            df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
        if "avg_price" in df.columns:
            df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
        # Parse change_text like "减持990.00万" or "增持1600.00" to signed share count
        if "change_text" in df.columns:
            def _parse(text: Any) -> float | None:
                s = str(text or "")
                sign = 1.0 if "增" in s else (-1.0 if "减" in s else None)
                if sign is None:
                    return None
                # strip non-numeric except dot
                num = "".join(ch for ch in s if ch.isdigit() or ch == ".")
                if not num:
                    return None
                value = float(num)
                if "万" in s:
                    value *= 1e4
                elif "亿" in s:
                    value *= 1e8
                return sign * value
            df["change_shares"] = df["change_text"].map(_parse)
        df["symbol"] = symbol.upper()
        return df.sort_values("announce_date", ascending=False, na_position="last").reset_index(drop=True)


def get_provider(market: str = "CN") -> AKShareCNProvider:
    """Return an AKShareCNProvider — only ``CN`` is supported."""
    return AKShareCNProvider(market)
