#!/usr/bin/env python3
"""Interactive dashboard for the OpenBB three-month dataset."""

from __future__ import annotations

from datetime import date, timedelta
import html
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title='价值锚点监控', layout='wide')

px.defaults.template = "plotly_white"

THEME_COLORS = [
    "#143F6B",
    "#F55353",
    "#FEB139",
    "#3EC1D3",
    "#5D2E8C",
    "#2FA8A4",
    "#FF6F91",
    "#6A67CE",
]
COLOR_UP = "#e63946"  # 涨 -> 红色
COLOR_DOWN = "#2ecc71"  # 跌 -> 绿色
CATEGORY_MAP = {
    "NVDA": "AI算力-芯片",
    "AMD": "AI算力-芯片",
    "TSM": "AI算力-芯片",
    "MSFT": "云平台与服务",
    "AMZN": "云平台与服务",
    "GOOGL": "云平台与服务",
    "AVGO": "硬件/网络支持",
    "MRVL": "硬件/网络支持",
    "ANET": "硬件/网络支持",
    "SMCI": "硬件/网络支持",
    "SNOW": "软件与数据",
    "MDB": "软件与数据",
}
AI_STACK_INFO = [
    {"layer": "AI算力-芯片", "symbol": "NVDA", "company": "英伟达 (NVDA)", "reason": "GPU 与 CUDA 生态构成唯一标准，大模型训练必备"},
    {"layer": "AI算力-芯片", "symbol": "AMD", "company": "AMD (AMD)", "reason": "MI300 系列正在获得云厂商订单，提供英伟达之外的选择"},
    {"layer": "AI算力-芯片", "symbol": "TSM", "company": "台积电 (TSM)", "reason": "先进制程代工霸主，英伟达/AMD 核心芯片由其代工"},
    {"layer": "云平台与服务", "symbol": "MSFT", "company": "微软 (MSFT)", "reason": "Azure + OpenAI 组合，企业级 AI 基础设施领导者"},
    {"layer": "云平台与服务", "symbol": "AMZN", "company": "亚马逊 (AMZN)", "reason": "AWS + 自研芯片 + Bedrock，按需租赁算力"},
    {"layer": "云平台与服务", "symbol": "GOOGL", "company": "谷歌 (GOOGL)", "reason": "TPU + Vertex AI + Gemini，技术栈完整"},
    {"layer": "硬件/网络支持", "symbol": "AVGO", "company": "博通 (AVGO)", "reason": "ASIC 与高速网络芯片，支撑云厂商定制需求"},
    {"layer": "硬件/网络支持", "symbol": "MRVL", "company": "迈威尔 (MRVL)", "reason": "数据中心互连芯片，AI 集群高速网络核心"},
    {"layer": "硬件/网络支持", "symbol": "ANET", "company": "Arista (ANET)", "reason": "高速交换机供应商，AI 数据中心网络必备"},
    {"layer": "硬件/网络支持", "symbol": "SMCI", "company": "超微电脑 (SMCI)", "reason": "AI 服务器系统集成商，深度绑定英伟达"},
    {"layer": "软件与数据", "symbol": "SNOW", "company": "Snowflake (SNOW)", "reason": "云原生数据仓库，AI 所需的数据燃料管家"},
    {"layer": "软件与数据", "symbol": "MDB", "company": "MongoDB (MDB)", "reason": "文档数据库，支撑 AI 应用的非结构化数据"},
]
SYMBOL_REASON = {row["symbol"]: row["reason"] for row in AI_STACK_INFO}

CUSTOM_CSS = """
<style>
body {
    background-color: #f5f6fa;
    font-family: "Inter", "Helvetica Neue", sans-serif;
}
.metric-card {
    background: #ffffff;
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 8px 16px rgba(20, 63, 107, 0.08);
    border: 1px solid #e6ebf3;
    color: #143f6b;
}
.metric-card h2 {
    color: #0d2e50;
    margin: 4px 0 0;
}
.metric-card p {
    color: #5b6b80;
    margin: 0 0 4px;
    font-size: 0.95rem;
}
.badge {
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.badge-green { background: #e8f8f5; color: #117a65; }
.badge-yellow { background: #fcf3cf; color: #9c640c; }
.badge-gray { background: #ecf0f1; color: #566573; }
.badge-red { background: #fdecea; color: #c0392b; }
.badge-blue { background: #e3f2fd; color: #0d47a1; }
.badge-purple { background: #f3e5f5; color: #6a1b9a; }
.tag {
    font-size: 0.78rem;
    padding: 2px 8px;
    border-radius: 999px;
    background: #eef2ff;
    color: #4338ca;
    display: inline-block;
}
.table-wrapper {
    margin-top: 12px;
    overflow-x: auto;
    border-radius: 18px;
    box-shadow: 0 15px 35px rgba(20, 63, 107, 0.08);
}
table {
    width: 100%;
    border-collapse: collapse;
    background: #fff;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(20,63,107,0.05);
}
th, td {
    padding: 10px 12px;
    border-bottom: 1px solid #eef2f7;
    text-align: center;
    color: #1f2a37;
    font-size: 0.92rem;
}
table thead th {
    position: sticky;
    top: 0;
    z-index: 2;
}
table tbody tr:nth-child(even) {
    background: #fbfdff;
}
th {
    background: #f0f4fb;
    color: #143f6b;
    font-weight: 600;
}
tr:hover {
    background: #f8fbff;
}
button {
    border: none;
    background: #143f6b;
    color: #fff;
    padding: 4px 10px;
    border-radius: 8px;
    cursor: pointer;
}
.info-btn {
    border: 1px solid #dbe4f3;
    background: #fff;
    color: #143f6b;
    transition: all 0.2s ease;
    font-size: 0.85rem;
}
.info-btn:hover {
    background: #143f6b;
    color: #fff;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "openbb_outputs"
HISTORY_PATH = DATA_DIR / "three_month_close_history.csv"
SUMMARY_PATH = DATA_DIR / "three_month_summary.csv"
ANALYST_PATH = DATA_DIR / "us_analyst_estimates.csv"
CRYPTO_DIR = DATA_DIR / "crypto"
CRYPTO_DASHBOARD_PATH = CRYPTO_DIR / "crypto_support_dashboard.csv"
DEFAULT_SUPABASE_URL = "https://wpyrevceqirzpwcpulqz.supabase.co"
DEFAULT_SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndweXJldmNlcWlyenB3Y3B1bHF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMzODUzOTEsImV4cCI6MjA3ODk2MTM5MX0.vY-lSpINIwDc80Caq7tX6iQ_zcBaKDflO5AfV79-tZA"
)
SUPABASE_URL = os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL)
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", DEFAULT_SUPABASE_KEY)
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}

TIER_BADGE = {
    "黄金坑": "badge-green",
    "白银坑": "badge-blue",
    "合理区": "badge-yellow",
    "白银": "badge-blue",
    "观望区": "badge-gray",
    "观望": "badge-gray",
    "观望/高估": "badge-gray",
    "高估": "badge-red",
    "高估区": "badge-red",
}

ENTRY_BADGE = {
    "建议入场": "badge-green",
    "可评估入场": "badge-purple",
    "暂不建议入场": "badge-gray",
    "暂不入场": "badge-gray",
}

MARKET_LABEL = {"US": "美股", "HK": "港股", "CN": "A股"}


def badge(text: str | float | None, mapping: dict[str, str], default: str = "badge-gray") -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        label = "N/A"
        key = None
    else:
        label = str(text)
        key = text
    css = mapping.get(key, default)
    return f"<span class='badge {css}'>{label}</span>"


def fmt_percent(value: float | None) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value*100:.1f}%"


def fmt_number(value: float | None, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def fmt_days(value: float | None) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{int(value)} 天"


def fmt_date(value: str | float | None) -> str:
    if pd.isna(value) or not value:
        return "N/A"
    try:
        parsed = pd.to_datetime(value).date()
        return parsed.isoformat()
    except Exception:
        return str(value)


def market_chip(market: str | None) -> str:
    label = MARKET_LABEL.get(market, market or "未知")
    return f"<span class='tag'>{label}</span>"


def bool_chip(value: bool | None) -> str:
    if pd.isna(value):
        return "N/A"
    return "✅" if bool(value) else "—"


def fetch_supabase_table(
    table: str,
    select: str = "*",
    order: Optional[str] = None,
    limit: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    params = {"select": select}
    if order:
        params["order"] = order
    if limit:
        params["limit"] = limit
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Supabase 表 {table} 读取失败：{exc}")
        return None


def align_history_to_summary(history_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict history rows to match the filtered summary universe."""
    if summary_df.empty:
        return history_df.iloc[0:0]
    if {"symbol"}.issubset(history_df.columns) and "symbol" in summary_df.columns:
        allowed = summary_df["symbol"].dropna().unique().tolist()
        return history_df[history_df["symbol"].isin(allowed)]
    if "name" in history_df.columns and "name" in summary_df.columns:
        allowed = summary_df["name"].dropna().unique().tolist()
        return history_df[history_df["name"].isin(allowed)]
    return history_df


@st.cache_data
def load_history() -> pd.DataFrame:
    df = fetch_supabase_table(
        "equity_metrics_history",
        select="symbol,as_of_date,name_en,name_cn,market,theme_category,investment_reason,"
        "value_score,value_score_tier,entry_recommendation,entry_reason,tier_reason,"
        "pe_percentile_5y,ps_percentile_5y,peg_ratio,fcf_yield,end_pe,current_ps,"
        "refresh_interval_days,next_refresh_date,pct_change,support_level_primary,"
        "support_level_secondary,ma50,ma200,rsi14,fib_38_2,fib_50,fib_61_8",
        order="as_of_date.asc",
    )
    if df is None or df.empty:
        df = pd.read_csv(HISTORY_PATH)
    else:
        df = df.rename(columns={"as_of_date": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["name"] = df.apply(
            lambda row: f"{row.get('name_en','')}（{row.get('name_cn','')}）"
            if row.get("name_en")
            else row.get("name_cn", row.get("symbol")),
            axis=1,
        )
        if "support_level_primary" in df.columns and "support_level" not in df.columns:
            df["support_level"] = pd.to_numeric(df["support_level_primary"], errors="coerce")
        if "support_level_secondary" in df.columns:
            df["support_level_secondary"] = pd.to_numeric(
                df["support_level_secondary"], errors="coerce"
            )
    return df


@st.cache_data
def load_summary() -> pd.DataFrame:
    df = fetch_supabase_table("equity_metrics")
    if df is None or df.empty:
        df = pd.read_csv(SUMMARY_PATH)
    else:
        for col in ["start_date", "end_date", "next_refresh_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        df["name"] = df.apply(
            lambda row: f"{row.get('name_en','')}（{row.get('name_cn','')}）"
            if row.get("name_en")
            else row.get("name_cn", row.get("symbol")),
            axis=1,
        )
    return df


@st.cache_data
def load_analyst() -> pd.DataFrame:
    if not ANALYST_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ANALYST_PATH, parse_dates=["date（财年/报告期）"])
    return df


def render_equity_dashboard() -> None:
    st.title("OpenBB 三个月表现仪表盘")
    st.caption(
        "数据来源：`openbb_three_months.py` 和 `fetch_us_analyst_estimates.py` 生成的 CSV 文件。"
    )
    with st.expander("指标说明"):
        st.markdown(
            """
            - **归一化收盘价**：以观察期首日为 1，便于比较不同股票的走势。
            - **历史分位**：当前收盘价在过去五年价格分布中的位置。
            - **支撑位**：以 20 日滚动最低价 (主支撑) 及其上浮 10% (次支撑) 作为参考。
            - **PE（市盈率）/PE分位(5年)**：股价相对盈利的高低及其历史分布位置。
            - **P/S（市销率）/P/S分位(5年)**：股价相对收入的估值水平及其历史分位。
            - **远期PE**：股价除以分析师预计的未来12个月每股收益。
            - **PEG**：市盈率除以 EPS 增长率；目前因缺少一年以上 EPS 数据暂为空。
            - **自由现金流收益率**：TTM 自由现金流 / 市值，衡量现金回报率。
            - **综合结论**：满足 4 条 → 建议入场；≥2 条 → 可评估入场；否则暂不建议入场。
            """
        )

    with st.expander("AI 产业链速览"):
        stack_df = pd.DataFrame(AI_STACK_INFO)
        st.dataframe(stack_df, use_container_width=True)

    history = load_history()
    history["date"] = (
        pd.to_datetime(history["date"], utc=True, errors="coerce").dt.tz_convert(None)
    )
    history = history.dropna(subset=["date"])
    summary = load_summary()
    expected_cols = {
        "refresh_interval_days",
        "next_refresh_date",
        "current_ps",
        "forward_pe",
        "market",
        "score_hist_valuation",
        "score_abs_valuation",
        "score_peer_valuation",
        "score_peg",
        "score_growth_quality",
        "score_balance_sheet",
        "score_shareholder_return",
        "score_support",
        "score_sentiment",
        "value_score",
        "value_score_tier",
        "tier_reason",
        "entry_reason",
    }
    missing_cols = expected_cols - set(summary.columns)
    for col in missing_cols:
        summary[col] = np.nan
    if "market" not in history.columns:
        history["market"] = "Unknown"
    if "market" not in summary.columns:
        summary["market"] = "Unknown"
    summary["theme_category"] = summary["symbol"].map(CATEGORY_MAP).fillna("其他")
    summary["investment_reason"] = summary["symbol"].map(SYMBOL_REASON).fillna("")
    if "symbol" in history.columns:
        history["theme_category"] = history["symbol"].map(CATEGORY_MAP).fillna("其他")
    else:
        history["theme_category"] = "Unknown"
    analyst = load_analyst()

    market_options = sorted(summary["market"].dropna().unique())
    selected_markets = st.multiselect(
        "选择市场",
        market_options,
        default=market_options,
    )
    summary = summary[summary["market"].isin(selected_markets)]
    history = history[history["market"].isin(selected_markets)]
    history = align_history_to_summary(history, summary)
    category_options = sorted(summary["theme_category"].dropna().unique())
    selected_categories = st.multiselect(
        "选择产业链层级",
        category_options,
        default=category_options,
    )
    if selected_categories:
        summary = summary[summary["theme_category"].isin(selected_categories)]
        history = align_history_to_summary(history, summary)

    search_kw = st.text_input("搜索公司/代码")
    if search_kw:
        keyword = search_kw.strip().lower()
        if keyword:
            search_cols = [col for col in ["name", "name_cn", "name_en", "symbol"] if col in summary.columns]
            if search_cols:
                mask = pd.Series(False, index=summary.index)
                for col in search_cols:
                    mask |= summary[col].astype(str).str.lower().str.contains(keyword, na=False)
                summary = summary[mask]
                history = align_history_to_summary(history, summary)

    entry_series = summary["entry_recommendation"] if "entry_recommendation" in summary.columns else pd.Series(dtype=str)
    entry_options = sorted(entry_series.dropna().unique().tolist())
    if entry_options:
        selected_entry = st.multiselect(
            "筛选入场结论",
            entry_options,
            default=entry_options,
        )
        if selected_entry:
            summary = summary[summary["entry_recommendation"].isin(selected_entry)]
            history = align_history_to_summary(history, summary)

    if summary.empty or history.empty:
        st.warning("所选条件暂无数据")
        return

    if "value_score" in summary.columns:
        summary = summary.sort_values("value_score", ascending=False)

    st.subheader("产业链概览")
    cat_summary = (
        summary.groupby("theme_category")
        .agg(
            company_count=("symbol", "nunique"),
            avg_score=("value_score", "mean"),
            top_score=("value_score", "max"),
        )
        .reset_index()
    )
    cat_summary["avg_score"] = cat_summary["avg_score"].round(1)
    cat_summary["top_score"] = cat_summary["top_score"].round(1)
    st.dataframe(cat_summary.rename(columns={"theme_category": "分类", "company_count": "公司数", "avg_score": "平均得分", "top_score": "最高得分"}), use_container_width=True)

    if category_options:
        category_tabs = st.tabs(category_options)
        for cat, tab in zip(category_options, category_tabs):
            subset_cat = summary[summary["theme_category"] == cat]
            if subset_cat.empty:
                tab.info("暂无数据")
                continue
            avg_cat = subset_cat["value_score"].mean()
            metric_text = f"{avg_cat:.1f}" if not np.isnan(avg_cat) else "N/A"
            tab.metric("平均得分", metric_text, help="分类内 Value Score 均值")
            display_cols = [col for col in ["name", "symbol", "value_score", "value_score_tier", "investment_reason"] if col in subset_cat.columns]
            tab.dataframe(subset_cat[display_cols], use_container_width=True)
    else:
        st.info("当前筛选暂无分类数据")

        summary = summary.sort_values("value_score", ascending=False)

    # KPI cards
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    golden = summary[summary["value_score_tier"] == "黄金坑"]
    silver = summary[summary["value_score_tier"] == "白银坑"]
    avg_score = summary["value_score"].mean().round(1) if not summary.empty else 0
    col_kpi1.markdown(
        f"<div class='metric-card'><p>黄金坑数量</p><h2>{len(golden)}</h2></div>",
        unsafe_allow_html=True,
    )
    col_kpi2.markdown(
        f"<div class='metric-card'><p>白银坑数量</p><h2>{len(silver)}</h2></div>",
        unsafe_allow_html=True,
    )
    col_kpi3.markdown(
        f"<div class='metric-card'><p>平均价值得分</p><h2>{avg_score}</h2></div>",
        unsafe_allow_html=True,
    )

    st.subheader("入场信号 & 价值得分")
    tab_eval, tab_score = st.tabs(["入场信号", "价值得分拆解"])

    with tab_eval:
        required_cols = {
            "entry_recommendation",
            "pe_percentile_5y",
            "ps_percentile_5y",
            "peg_ratio",
            "fcf_yield",
        }
        if required_cols.issubset(summary.columns):
            eval_df = summary[
                [
                    "name",
                    "symbol",
                    "market",
                    "entry_recommendation",
                    "value_score",
                    "value_score_tier",
                    "tier_reason",
                    "entry_reason",
                    "pe_percentile_5y",
                    "ps_percentile_5y",
                    "peg_ratio",
                    "fcf_yield",
                    "refresh_interval_days",
                    "next_refresh_date",
                    "end_pe",
                    "current_ps",
                    "forward_pe",
                    "pe_coverage_years",
                    "ps_coverage_years",
                ]
            ].copy()

            signal_kw = st.text_input("搜索入场信号标的", "", key="signal_search")
            if signal_kw:
                keyword = signal_kw.strip().lower()
                if keyword:
                    mask = (
                        eval_df["name"].astype(str).str.lower().str.contains(keyword, na=False)
                        | eval_df["symbol"].astype(str).str.lower().str.contains(keyword, na=False)
                    )
                    eval_df = eval_df[mask]

            eval_df["pe_condition"] = eval_df["pe_percentile_5y"].apply(
                lambda v: np.nan if pd.isna(v) else v <= 0.3
            )
            eval_df["ps_condition"] = eval_df["ps_percentile_5y"].apply(
                lambda v: np.nan if pd.isna(v) else v <= 0.3
            )
            eval_df["peg_condition"] = eval_df["peg_ratio"].apply(
                lambda v: np.nan if pd.isna(v) else v <= 1
            )
            eval_df["fcf_condition"] = eval_df["fcf_yield"].apply(
                lambda v: np.nan if pd.isna(v) else v >= 0.04
            )

            eval_df = eval_df.rename(
                columns={
                    "name": "标的（中英）",
                    "symbol": "代码",
                    "entry_recommendation": "综合结论",
                    "market": "市场",
                    "value_score": "价值得分",
                    "value_score_tier": "得分等级",
                    "tier_reason": "等级说明",
                    "entry_reason": "入场说明",
                    "pe_percentile_5y": "PE分位(5年)",
                    "ps_percentile_5y": "P/S分位(5年)",
                    "peg_ratio": "PEG",
                    "fcf_yield": "自由现金流收益率",
                    "next_refresh_date": "下一次刷新日期",
                    "refresh_interval_days": "刷新间隔(天)",
                    "end_pe": "当前PE",
                    "current_ps": "当前P/S",
                    "forward_pe": "远期PE",
                    "pe_coverage_years": "PE历史覆盖(年)",
                    "ps_coverage_years": "P/S历史覆盖(年)",
                }
            )

            def format_tooltip(row):
                reason = row.get("等级说明") or "暂无说明"
                entry = row.get("入场说明") or "暂无说明"
                return f"价值结论：{reason}\\n入场结论：{entry}"

            eval_df["PE分位(5年)"] = eval_df["PE分位(5年)"].map(fmt_percent)
            eval_df["P/S分位(5年)"] = eval_df["P/S分位(5年)"].map(fmt_percent)
            eval_df["自由现金流收益率"] = eval_df["自由现金流收益率"].map(fmt_percent)
            eval_df["PEG"] = eval_df["PEG"].map(lambda v: fmt_number(v, 2))
            eval_df["当前PE"] = eval_df["当前PE"].map(lambda v: fmt_number(v, 1))
            eval_df["当前P/S"] = eval_df["当前P/S"].map(lambda v: fmt_number(v, 2))
            eval_df["远期PE"] = eval_df["远期PE"].map(lambda v: fmt_number(v, 1))
            eval_df["PE历史覆盖(年)"] = eval_df["PE历史覆盖(年)"].map(lambda v: fmt_number(v, 1))
            eval_df["P/S历史覆盖(年)"] = eval_df["P/S历史覆盖(年)"].map(lambda v: fmt_number(v, 1))
            eval_df["刷新间隔(天)"] = eval_df["刷新间隔(天)"].map(fmt_days)
            eval_df["下一次刷新日期"] = eval_df["下一次刷新日期"].map(fmt_date)
            eval_df["市场"] = eval_df["市场"].apply(market_chip)
            eval_df["综合结论"] = eval_df["综合结论"].apply(lambda v: badge(v, ENTRY_BADGE))
            eval_df["得分等级"] = eval_df["得分等级"].apply(lambda v: badge(v, TIER_BADGE))
            eval_df["价值得分"] = eval_df["价值得分"].map(lambda v: fmt_number(v, 1))
            eval_df["PE分位<=30%"] = eval_df["pe_condition"].map(bool_chip)
            eval_df["P/S分位<=30%"] = eval_df["ps_condition"].map(bool_chip)
            eval_df["PEG<=1"] = eval_df["peg_condition"].map(bool_chip)
            eval_df["FCF收益>=4%"] = eval_df["fcf_condition"].map(bool_chip)

            eval_df["说明"] = eval_df.apply(format_tooltip, axis=1)
            eval_df["详情"] = eval_df["说明"].apply(
                lambda text: f"<button class='info-btn' title='{html.escape(text, quote=True)}'>详情</button>"
            )
            display_df = eval_df.drop(
                columns=[
                    "等级说明",
                    "入场说明",
                    "说明",
                    "pe_condition",
                    "ps_condition",
                    "peg_condition",
                    "fcf_condition",
                ]
            )
            display_columns = [
                "市场",
                "标的（中英）",
                "代码",
                "综合结论",
                "得分等级",
                "价值得分",
                "PE分位(5年)",
                "P/S分位(5年)",
                "当前PE",
                "当前P/S",
                "远期PE",
                "自由现金流收益率",
                "PEG",
                "PE历史覆盖(年)",
                "P/S历史覆盖(年)",
                "刷新间隔(天)",
                "下一次刷新日期",
                "PE分位<=30%",
                "P/S分位<=30%",
                "PEG<=1",
                "FCF收益>=4%",
                "详情",
            ]
            display_df = display_df[display_columns]
            table_html = display_df.to_html(escape=False, index=False)
            st.markdown(f"<div class='table-wrapper'>{table_html}</div>", unsafe_allow_html=True)
            csv_bytes = summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "下载当前筛选结果",
                data=csv_bytes,
                file_name="three_month_summary_filtered.csv",
                mime="text/csv",
            )
        else:
            missing = required_cols - set(summary.columns)
            st.info(
                f"当前汇总缺少字段 {missing}，请确认 `openbb_three_months.py` 版本已更新并重新运行。"
            )

    with tab_score:
        score_cols = [
            "name",
            "symbol",
            "market",
            "value_score",
            "value_score_tier",
            "score_hist_valuation",
            "score_abs_valuation",
            "score_peer_valuation",
            "score_peg",
            "score_growth_quality",
            "score_balance_sheet",
            "score_shareholder_return",
            "score_support",
            "score_sentiment",
        ]
        if set(score_cols).issubset(summary.columns) and not summary.empty:
            score_df = summary[score_cols].copy()
            score_df = score_df.sort_values("value_score", ascending=False)
            score_df["market_label"] = score_df["market"].map(MARKET_LABEL).fillna(score_df["market"])
            score_df = score_df.rename(
                columns={
                    "name": "标的（中英）",
                    "symbol": "代码",
                    "value_score": "综合得分",
                    "value_score_tier": "等级",
                    "market_label": "市场",
                    "score_hist_valuation": "历史估值",
                    "score_abs_valuation": "绝对估值",
                    "score_peer_valuation": "同业比较",
                    "score_peg": "PEG 匹配",
                    "score_growth_quality": "增长质量",
                    "score_balance_sheet": "资产负债",
                    "score_shareholder_return": "股东回报",
                    "score_support": "技术支撑",
                    "score_sentiment": "市场情绪",
                }
            )
            score_df = score_df[
                [
                    "市场",
                    "标的（中英）",
                    "代码",
                    "综合得分",
                    "等级",
                    "历史估值",
                    "绝对估值",
                    "同业比较",
                    "PEG 匹配",
                    "增长质量",
                    "资产负债",
                    "股东回报",
                    "技术支撑",
                    "市场情绪",
                ]
            ]
            st.dataframe(
                score_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "综合得分": st.column_config.NumberColumn("综合得分", format="%.1f"),
                    "历史估值": st.column_config.ProgressColumn("历史估值", min_value=0, max_value=10, format="%d"),
                    "绝对估值": st.column_config.ProgressColumn("绝对估值", min_value=0, max_value=10, format="%d"),
                    "同业比较": st.column_config.ProgressColumn("同业比较", min_value=0, max_value=10, format="%d"),
                    "PEG 匹配": st.column_config.ProgressColumn("PEG 匹配", min_value=0, max_value=15, format="%d"),
                    "增长质量": st.column_config.ProgressColumn("增长质量", min_value=0, max_value=15, format="%d"),
                    "资产负债": st.column_config.ProgressColumn("资产负债", min_value=0, max_value=10, format="%d"),
                    "股东回报": st.column_config.ProgressColumn("股东回报", min_value=0, max_value=10, format="%d"),
                    "技术支撑": st.column_config.ProgressColumn("技术支撑", min_value=0, max_value=10, format="%d"),
                    "市场情绪": st.column_config.ProgressColumn("市场情绪", min_value=0, max_value=10, format="%d"),
                },
            )
        else:
            st.info("当前数据缺少完整的得分拆解字段。")

    st.markdown("---")

    tickers = history["name"].unique().tolist()
    selected = st.multiselect("选择要展示的标的", tickers, default=tickers)
    filtered = history[history["name"].isin(selected)]
    two_years_ago = pd.Timestamp(date.today() - timedelta(days=730))
    filtered = filtered[filtered["date"] >= two_years_ago]

    if filtered.empty:
        st.warning("请选择至少一支股票。")
        return

    st.subheader("归一化收盘价（近两年，首日=1）")
    norm_fig = px.line(
        filtered,
        x="date",
        y="close_norm",
        color="name",
        labels={"close_norm": "Normalized Close", "date": "Date"},
        hover_data={"close": ":.2f"},
        color_discrete_sequence=THEME_COLORS,
    )
    if {"support_level", "support_level_secondary"}.issubset(filtered.columns):
        for name in filtered["name"].unique():
            sub = filtered[filtered["name"] == name]
            norm_fig.add_scatter(
                x=sub["date"],
                y=sub["support_level"] / sub["close"].iloc[0],
                mode="lines",
                line=dict(dash="dot"),
                name=f"{name}-主支撑",
                legendgroup=name,
                showlegend=False,
            )
            norm_fig.add_scatter(
                x=sub["date"],
                y=sub["support_level_secondary"] / sub["close"].iloc[0],
                mode="lines",
                line=dict(dash="dot"),
                name=f"{name}-次支撑",
                legendgroup=name,
                showlegend=False,
            )
    norm_fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(255,255,255,0.6)",
        ),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(norm_fig, use_container_width=True)

    st.subheader("PE 走势（TTM）")
    pe_df = filtered.dropna(subset=["pe"])
    if pe_df.empty:
        st.info("当前数据源缺少 PE，暂无法绘制。")
    else:
        pe_fig = px.line(
            pe_df,
            x="date",
            y="pe",
            color="name",
            labels={"pe": "PE (TTM)", "date": "Date"},
            color_discrete_sequence=THEME_COLORS,
        )
        st.plotly_chart(pe_fig, use_container_width=True)

    st.subheader("历史分位（收盘价相对于过去五年）")
    percentile_fig = px.line(
        filtered,
        x="date",
        y="close_percentile",
        color="name",
        labels={"close_percentile": "Percentile", "date": "Date"},
        hover_data={"close_percentile": ":.2%"},
        color_discrete_sequence=THEME_COLORS,
    )
    percentile_fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(percentile_fig, use_container_width=True)

    st.subheader("汇总对比")
    summary_display = summary.copy()
    summary_display["pct_change"] = summary_display["pct_change"] * 100
    cols = st.columns(3)
    with cols[0]:
        pct_fig = px.bar(
            summary_display,
            x="name",
            y="pct_change",
            title="三个月涨跌幅(%)",
            labels={"pct_change": "% Change"},
            color_discrete_sequence=THEME_COLORS,
        )
        pct_colors = [COLOR_UP if val >= 0 else COLOR_DOWN for val in summary_display["pct_change"]]
        pct_fig.update_traces(marker_color=pct_colors)
        st.plotly_chart(pct_fig, use_container_width=True)
    with cols[1]:
        pe_bar = px.bar(
            summary_display.dropna(subset=["end_pe"]),
            x="name",
            y="end_pe",
            title="期末 PE",
            color_discrete_sequence=THEME_COLORS,
        )
        st.plotly_chart(pe_bar, use_container_width=True)
    with cols[2]:
        perc_bar = px.bar(
            summary_display,
            x="name",
            y="end_close_percentile",
            title="期末分位(%)",
            color_discrete_sequence=THEME_COLORS,
        )
        perc_bar.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        perc_bar.update_yaxes(tickformat=".0%")
        st.plotly_chart(perc_bar, use_container_width=True)

    st.subheader("分析师预测（最近5年）")
    if analyst.empty:
        st.info("未找到 `us_analyst_estimates.csv`，请先运行 `fetch_us_analyst_estimates.py`。")
        return

    analyst_symbols = analyst["symbol（股票代码）"].unique().tolist()
    default_selection = analyst_symbols[:]
    selected_analyst = st.multiselect(
        "选择公司",
        analyst_symbols,
        default=default_selection,
    )

    filtered_analyst = analyst[analyst["symbol（股票代码）"].isin(selected_analyst)]
    if filtered_analyst.empty:
        st.warning("请至少选择一个公司查看分析师预测。")
        return

    eps_fig = px.line(
        filtered_analyst,
        x="date（财年/报告期）",
        y="epsAvg（每股收益均值）",
        color="symbol（股票代码）",
        labels={"epsAvg（每股收益均值）": "EPS均值", "date（财年/报告期）": "财年"},
        markers=True,
        color_discrete_sequence=THEME_COLORS,
    )
    st.plotly_chart(eps_fig, use_container_width=True)

    revenue_fig = px.line(
        filtered_analyst,
        x="date（财年/报告期）",
        y="revenueAvg（营收均值）",
        color="symbol（股票代码）",
        labels={"revenueAvg（营收均值）": "营收均值", "date（财年/报告期）": "财年"},
        markers=True,
        color_discrete_sequence=THEME_COLORS,
    )
    st.plotly_chart(revenue_fig, use_container_width=True)

    st.dataframe(filtered_analyst.sort_values("date（财年/报告期）"), use_container_width=True)


def _load_crypto_dashboard() -> Optional[pd.DataFrame]:
    df = fetch_supabase_table("crypto_supports")
    if df is None or df.empty:
        if not CRYPTO_DASHBOARD_PATH.exists():
            st.warning("未找到 `openbb_outputs/crypto/crypto_support_dashboard.csv`，请先运行 `fetch_crypto_supports.py`。")
            return None
        df = pd.read_csv(CRYPTO_DASHBOARD_PATH)
    numeric_cols = [
        "last_price",
        "ma50",
        "ma200",
        "fib_38_2",
        "fib_50",
        "fib_61_8",
        "rsi14",
        "mvrv",
        "lth_cost_basis",
        "exchange_netflow",
        "asopr",
        "funding_rate",
        "fear_greed",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rsi_divergence" in df.columns:
        df["rsi_divergence"] = df["rsi_divergence"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "retrieved_at" in df.columns:
        df["retrieved_at"] = pd.to_datetime(df["retrieved_at"], errors="coerce")
    if "recent_supports" in df.columns:
        df["recent_supports"] = df["recent_supports"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
        )
    return df


def _format_supports(raw: str | float | None) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "暂无数据"
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return ", ".join(f"{item.get('date')}@{item.get('price'):.2f}" for item in data if isinstance(item, dict))
        except Exception:
            pass
    return str(raw)


def _format_number(value: float, fmt: str = ".2f") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return format(float(value), fmt)


def _format_percent(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


def _colored_percentage(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    color = COLOR_UP if value >= 0 else COLOR_DOWN
    prefix = "+" if value > 0 else ""
    return f"<span style='color:{color};font-weight:600'>{prefix}{value:.2f}%</span>"


def _format_usd(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:.2f}K"
    return f"${value:,.0f}"


def render_crypto_dashboard() -> None:
    st.title("加密货币 - 三维度确认法")
    st.caption("数据来源：`fetch_crypto_supports.py` 生成的 `openbb_outputs/crypto/` 文件")
    with st.expander("名词解释"):
        st.markdown(
            """
- **recent_supports / 支撑次数**：最近三次摆动低点及价格；次数越多说明该区间被多次验证。
- **fib_38_2 / 50 / 61_8**：依据最近 120 天摆动区间计算的斐波那契关键位，常用作反弹目标或支撑。
- **swing_low / swing_high / swing_range**：当前摆动区间的低点、高点与跨度，刻画波动范围。
- **pct_change_7d / 30d**：近 7 / 30 日的累积涨跌幅，用于观察短中期动能。
- **distance_ma50_pct / distance_ma200_pct**：现价相对 50 / 200 日均线的偏离百分比，为负表示跌破均线。
- **volume_avg_30d**：近 30 日平均成交量，衡量筹码活跃度。
- **hvn_zones**：Volume Profile 的高成交量节点，每个节点给出价格上下沿与成交量。
- **fear_greed / funding_rate**：恐惧与贪婪指数、永续合约资金费率，两者共同反映市场情绪。
            """
        )

    df = _load_crypto_dashboard()
    if df is None or df.empty:
        return

    latest_ts = None
    if "retrieved_at" in df.columns:
        ts_series = df["retrieved_at"].dropna()
        if not ts_series.empty:
            latest_ts = ts_series.max()

    numeric_distance = pd.to_numeric(df.get("distance_ma200_pct"), errors="coerce")
    pct_above_ma200 = int((numeric_distance >= 0).sum()) if numeric_distance is not None else 0
    avg_rsi = df["rsi14"].mean(skipna=True)
    fear_numeric = pd.to_numeric(df.get("fear_greed"), errors="coerce")
    extreme_fear = int((fear_numeric <= 25).sum()) if fear_numeric is not None else 0

    overview_cols = st.columns(4)
    overview_cols[0].metric("跟踪币种数", len(df))
    overview_cols[1].metric("高于 200 日均线", pct_above_ma200)
    overview_cols[2].metric("平均 RSI(14)", f"{avg_rsi:.1f}" if not np.isnan(avg_rsi) else "N/A")
    overview_cols[3].metric("极度恐惧币数", extreme_fear)
    if latest_ts is not None:
        st.caption(f"最近行情时间：{latest_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    coins = df["symbol"].tolist()
    default_selection = coins[:4] if len(coins) >= 4 else coins
    selected = st.multiselect("选择币种", coins, default=default_selection)
    if not selected:
        st.info("请选择至少一个币种。")
        return

    subset = df[df["symbol"].isin(selected)].reset_index(drop=True)
    for _, row in subset.iterrows():
        st.subheader(f"{row['symbol']} | {row.get('name', row['symbol'])}")
        ts_val = row.get("retrieved_at")
        if isinstance(ts_val, str):
            ts_display = ts_val
        elif pd.notna(ts_val):
            ts_display = pd.to_datetime(ts_val).strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_display = "N/A"
        price_source = row.get("price_source", "N/A")
        st.caption(f"数据时间：{ts_display} UTC | 行情源：{price_source}")

        col1, col2, col3, col4 = st.columns(4)
        strength_display = row.get("support_strength")
        strength_display = int(strength_display) if strength_display is not None and not pd.isna(strength_display) else 0

        col1.markdown(
            f"<div class='metric-card'><p>现价</p><h2>{_format_number(row.get('last_price'))}</h2>"
            f"<p>7 日：{_colored_percentage(row.get('pct_change_7d'))}</p></div>",
            unsafe_allow_html=True,
        )
        col2.markdown(
            f"<div class='metric-card'><p>30 日涨跌幅</p><h2>{_colored_percentage(row.get('pct_change_30d'))}</h2></div>",
            unsafe_allow_html=True,
        )
        col3.markdown(
            f"<div class='metric-card'><p>相对 50 日均线</p><h2>{_colored_percentage(row.get('distance_ma50_pct'))}</h2></div>",
            unsafe_allow_html=True,
        )
        col4.markdown(
            f"<div class='metric-card'><p>相对 200 日均线</p><h2>{_colored_percentage(row.get('distance_ma200_pct'))}</h2></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"- **RSI(14)**：{_format_number(row.get('rsi14'))}；"
            f"**支撑次数**：{strength_display}"
        )

        trend = row.get("ma_trend", "N/A")
        supports = _format_supports(row.get("recent_supports"))
        fib_text = (
            f"38.2% {_format_number(row.get('fib_38_2'))} | "
            f"50% {_format_number(row.get('fib_50'))} | "
            f"61.8% {_format_number(row.get('fib_61_8'))}"
        )
        st.markdown(
            f"- **趋势**：{trend}（50 日 {_format_number(row.get('ma50'))} / 200 日 {_format_number(row.get('ma200'))}）"
        )
        st.markdown(f"- **最近支撑**：{supports}")
        st.markdown(f"- **斐波那契回撤位**：{fib_text}")
        st.markdown(
            f"- **摆动区间**：低 {_format_number(row.get('swing_low'))} / 高 {_format_number(row.get('swing_high'))} "
            f"(跨度 {_format_number(row.get('swing_range'))})"
        )

        fear_value = row.get("fear_greed_text") or "N/A"
        st.markdown(
            f"- **情绪**：恐惧与贪婪 {row.get('fear_greed', 'N/A')} ({fear_value})；"
            f"资金费率 {_format_number(row.get('funding_rate'), '.4f')}"
        )
        st.markdown(
            f"- **链上指标**：MVRV {_format_number(row.get('mvrv'))} | "
            f"LTH Cost {_format_number(row.get('lth_cost_basis'))} | "
            f"aSOPR {_format_number(row.get('asopr'))}"
        )
        st.markdown(f"- **成交量（30 日均值）**：{_format_number(row.get('volume_avg_30d'))}")
        llama_chain = row.get("llama_chain")
        if llama_chain:
            st.markdown(
                f"- **链上 TVL ({llama_chain})**：{_format_usd(row.get('llama_chain_tvl'))}；"
                f"DEX 24h 量 {_format_usd(row.get('llama_dex_volume_24h'))}；"
                f"费用 {_format_usd(row.get('llama_fees_24h'))}"
            )
        if row.get("llama_stablecoin_supply"):
            st.markdown(
                f"- **稳定币供应 ({llama_chain or 'global'})**：{_format_usd(row.get('llama_stablecoin_supply'))}"
            )
        if bool(row.get("rsi_divergence")):
            st.success("RSI 与价格存在底背离，关注潜在反转。")

        hvn_raw = row.get("hvn_zones")
        zones = []
        if isinstance(hvn_raw, str):
            try:
                zones = json.loads(hvn_raw)
            except Exception:
                zones = []
        if zones:
            st.markdown("**成交量密集区 (HVN)**")
            hvn_df = pd.DataFrame(zones)
            hvn_df = hvn_df.rename(columns={"price_low": "下沿", "price_high": "上沿", "volume": "成交量"})
            st.dataframe(hvn_df, use_container_width=True)
        st.divider()

    st.subheader("原始数据")
    display_cols = [
        "symbol",
        "name",
        "last_price",
        "ma50",
        "ma200",
        "ma_trend",
        "price_source",
        "retrieved_at",
        "recent_supports",
        "support_strength",
        "fib_38_2",
        "fib_50",
        "fib_61_8",
        "swing_low",
        "swing_high",
        "swing_range",
        "rsi14",
        "rsi_divergence",
        "pct_change_7d",
        "pct_change_30d",
        "distance_ma50_pct",
        "distance_ma200_pct",
        "llama_chain",
        "llama_chain_tvl",
        "llama_dex_volume_24h",
        "llama_fees_24h",
        "llama_stablecoin_supply",
        "mvrv",
        "lth_cost_basis",
        "exchange_netflow",
        "asopr",
        "funding_rate",
        "volume_avg_30d",
        "fear_greed",
        "fear_greed_text",
    ]
    available_cols = [c for c in display_cols if c in subset.columns]
    st.dataframe(subset[available_cols], use_container_width=True)


def main() -> None:
    view = st.sidebar.radio("选择数据类型", ("股票", "加密货币"))
    if view == "股票":
        render_equity_dashboard()
    else:
        render_crypto_dashboard()


if __name__ == "__main__":
    main()
