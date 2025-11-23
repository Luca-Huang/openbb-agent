#!/usr/bin/env python3
"""价值锚点监控仪表盘：多市场价格、估值与量能监控。"""

from __future__ import annotations

from datetime import date, timedelta
from uuid import uuid4
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

st.set_page_config(page_title='价值锚点监控仪表盘', layout='wide')

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#2E86AB",
    "#E27D60",
    "#E8A87C",
    "#C38D9E",
    "#41B3A3",
    "#6C5B7B",
    "#F67280",
    "#4D4C7D",
]

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --bg-base: #f5f6fb;
    --bg-card: #ffffff;
    --text-main: #0f172a;
    --text-muted: #6b7280;
    --border-soft: 1px solid rgba(15, 23, 42, 0.08);
    --shadow-soft: 0 20px 45px rgba(15, 23, 42, 0.08);
    --brand-blue: #2563eb;
    --brand-green: #22c55e;
    --brand-orange: #fb923c;
}
.stApp, body {
    background: var(--bg-base);
    font-family: "Inter", sans-serif;
    color: var(--text-main);
}
.main > div {
    padding-top: 1rem;
}
.nav-bar {
    background: var(--bg-card);
    border-radius: 20px;
    padding: 18px 30px;
    border: var(--border-soft);
    box-shadow: var(--shadow-soft);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.brand-icon {
    width: 46px;
    height: 46px;
    border-radius: 16px;
    background: radial-gradient(circle at 25% 25%, #8fc5ff, #2563eb);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: white;
    box-shadow: inset 0 1px 6px rgba(255,255,255,0.6);
}
.nav-tabs {
    display: inline-flex;
    gap: 10px;
}
.tab-pill {
    padding: 6px 18px;
    border-radius: 999px;
    border: 1px solid rgba(15,23,42,0.12);
    font-weight: 600;
    color: var(--text-muted);
}
.tab-pill.active {
    background: rgba(37,99,235,0.12);
    color: var(--brand-blue);
    border-color: transparent;
}
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: rgba(34,197,94,0.15);
    color: var(--brand-green);
    border-radius: 999px;
    font-weight: 600;
}
.live-indicator span {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: var(--brand-green);
    box-shadow: 0 0 5px rgba(34,197,94,0.6);
}
.section-heading {
    margin: 0.5rem 0 0.25rem;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}
.metric-card {
    background: var(--bg-card);
    padding: 26px;
    border-radius: 18px;
    border: var(--border-soft);
    box-shadow: var(--shadow-soft);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 16px;
    min-height: 140px;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 25px 50px rgba(15, 23, 42, 0.15);
}
.metric-card h2 {
    color: #475569;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-card p {
    color: var(--text-main);
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
}
.badge {
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid transparent;
}
.badge-green { background: #ecfdf5; color: #059669; border-color: #bbf7d0; }
.badge-red { background: #fef2f2; color: #dc2626; border-color: #fecdd3; }
.badge-blue { background: #eef2ff; color: #2563eb; border-color: #c7d2fe; }
.badge-yellow { background: #fffbeb; color: #d97706; border-color: #fde68a; }
.badge-gray { background: #f1f5f9; color: #475569; border-color: #e2e8f0; }
.tag {
    display: inline-flex;
    align-items: center;
    padding: 2px 10px;
    border-radius: 999px;
    background: rgba(148,163,184,0.15);
    color: #475569;
    font-size: 0.75rem;
    border: 1px solid rgba(148,163,184,0.25);
}
.chip-row {
    background: var(--bg-card);
    border-radius: 18px;
    padding: 18px 22px;
    border: var(--border-soft);
    box-shadow: var(--shadow-soft);
    margin: 10px 0 22px;
}
.chip-row h4 {
    margin: 0 0 6px;
    font-size: 0.95rem;
    color: var(--text-muted);
}
.chip-row .chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid rgba(15, 23, 42, 0.08);
    background: #f8fafc;
    margin: 4px 10px 4px 0;
    font-size: 0.85rem;
    color: #0f172a;
}
.chip-row .chip-label {
    background: rgba(37,99,235,0.08);
    border-color: rgba(37,99,235,0.2);
    color: var(--brand-blue);
    font-weight: 600;
    text-transform: uppercase;
}
.chip-row .chip.ghost {
    background: transparent;
    border-style: dashed;
    color: #94a3b8;
}
.highlight-board {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 10px;
}
.highlight-item {
    background: linear-gradient(165deg, #ffffff, #f0f4ff);
    border-radius: 16px;
    border: 1px solid rgba(37,99,235,0.15);
    padding: 18px;
    box-shadow: 0 12px 30px rgba(37,99,235,0.08);
}
.highlight-symbol {
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.04em;
}
.highlight-metrics {
    font-variant-numeric: tabular-nums;
    color: #475569;
    margin-top: 4px;
}
.table-card {
    background: var(--bg-card);
    border-radius: 22px;
    padding: 12px;
    border: var(--border-soft);
    box-shadow: var(--shadow-soft);
    margin-bottom: 20px;
}
.subtle-card {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 18px;
    border: var(--border-soft);
    box-shadow: var(--shadow-soft);
    margin-bottom: 20px;
}
.stTabs [role="tablist"] {
    border-bottom: var(--border-soft);
    padding-bottom: 0.5rem;
}
.stTabs [role="tab"] {
    font-weight: 600;
    padding: 0.6rem 1rem;
}
.dataframe {
    border-radius: 16px !important;
    overflow: hidden;
}
.stDataFrame div[role="table"] {
    border-radius: 16px;
    border: var(--border-soft);
}
.stDataFrame [data-testid="StyledTable"] {
    border-radius: 16px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "openbb_outputs"
HISTORY_PATH = DATA_DIR / "three_month_close_history.csv"
SUMMARY_PATH = DATA_DIR / "three_month_summary.csv"
ANALYST_PATH = DATA_DIR / "us_analyst_estimates.csv"
CRYPTO_CSV_PATH = DATA_DIR / "crypto" / "crypto_support_dashboard.csv"
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
SUPABASE_HOLDINGS_TABLE = os.environ.get("SUPABASE_HOLDINGS_TABLE", "holdings")

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

def render_page_nav(active: str = "equity") -> None:
    equity_class = "tab-pill active" if active == "equity" else "tab-pill"
    crypto_class = "tab-pill active" if active == "crypto" else "tab-pill"
    nav_html = f"""
    <div class="nav-bar">
        <div class="nav-brand">
            <div class="brand-icon">VA</div>
            <div>
                <strong>价值锚点 Value Anchor</strong><br/>
                <span style="font-size:0.85rem;color:var(--text-muted);">Equity & Crypto Monitor</span>
            </div>
        </div>
        <div class="nav-tabs">
            <div class="{equity_class}">股票资产 (Equity)</div>
            <div class="{crypto_class}">加密资产 (Crypto)</div>
        </div>
        <div class="live-indicator">
            <span></span> Live Feed
        </div>
    </div>
    """
    st.markdown(nav_html, unsafe_allow_html=True)


def render_overview_cards(summary_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    total_watchlist = len(summary_df)
    entry_series = summary_df.get("entry_recommendation", pd.Series(dtype=str))
    entry_ready = int((entry_series == "建议入场").sum())
    entry_watch = int((entry_series == "可评估入场").sum())
    avg_score = summary_df["value_score"].mean() if "value_score" in summary_df.columns else np.nan
    refresh_due = 0
    if "next_refresh_date" in summary_df.columns:
        today = date.today()
        refresh_due = int(
            summary_df["next_refresh_date"]
            .apply(lambda val: pd.to_datetime(val, errors="coerce"))
            .dt.date
            .lt(today)
            .sum()
        )
    spike_alerts = 0
    if {"symbol", "volume_spike_ratio"}.issubset(history_df.columns):
        latest_spike = (
            history_df.sort_values("date")
            .dropna(subset=["symbol"])
            .groupby("symbol")["volume_spike_ratio"]
            .last()
        )
        spike_alerts = int(latest_spike.ge(1.5).sum())
    cards = [
        ("监控标的", f"{total_watchlist}", "Watchlist"),
        ("建议入场", f"{entry_ready}", "满足全部条件"),
        ("可评估入场", f"{entry_watch}", "≥2 条件"),
        (
            "平均得分",
            f"{avg_score:.1f}" if not np.isnan(avg_score) else "N/A",
            f"{refresh_due} 条待刷新" if refresh_due else "刷新正常",
        ),
        ("量能突增", f"{spike_alerts}", "Spike ≥ 1.5x"),
    ]
    cols = st.columns(len(cards))
    for col, (title, value, subtitle) in zip(cols, cards):
        card_html = f"""
        <div class='metric-card'>
            <h2>{title}</h2>
            <p>{value}</p>
            <span class='badge badge-blue'>{subtitle}</span>
        </div>
        """
        col.markdown(card_html, unsafe_allow_html=True)


def render_active_filters(
    markets: list[str],
    categories: list[str],
    entry_status: list[str],
    keyword: str,
) -> None:
    def build_group(title: str, items: list[str]) -> str:
        if not items:
            return ""
        chips = "".join(f"<span class='chip'>{html.escape(item)}</span>" for item in items)
        return f"<div><span class='chip chip-label'>{title}</span>{chips}</div>"

    keyword = (keyword or "").strip()
    html_parts = ["<div class='chip-row'><h4>当前筛选</h4>"]
    html_parts.append(build_group("市场", markets))
    html_parts.append(build_group("产业链", categories))
    html_parts.append(build_group("入场结论", entry_status))
    if keyword:
        html_parts.append(
            f"<div><span class='chip chip-label'>关键字</span><span class='chip'>{html.escape(keyword)}</span></div>"
        )
    if len("".join(html_parts)) <= len("<div class='chip-row'><h4>当前筛选</h4></div>"):
        html_parts.append("<span class='chip ghost'>未选择任何筛选条件</span>")
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_highlight_section(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    summary = summary.copy()
    entry_rank = {"建议入场": 0, "可评估入场": 1}
    summary["entry_rank"] = summary.get("entry_recommendation", "").map(entry_rank).fillna(2)
    leaders = summary.sort_values(["entry_rank", "value_score"], ascending=[True, False]).head(4)
    if leaders.empty:
        return
    cards = []
    for _, row in leaders.iterrows():
        symbol = row.get("symbol", "--")
        name = row.get("name_cn") or row.get("name_en") or ""
        verdict = row.get("entry_recommendation", "待观察")
        score = row.get("value_score")
        support = row.get("support_level_primary")
        pct = row.get("end_close_percentile")
        cards.append(
            f"""
            <div class='highlight-item'>
                <div class='highlight-symbol'>{html.escape(symbol)}</div>
                <div style='color:#94a3b8;font-size:0.85rem;'>{html.escape(name)}</div>
                <div class='highlight-metrics'>得分 {fmt_number(score,1)} · 支撑位 {fmt_number(support,2)} · 分位 {fmt_percent(pct)}</div>
                <div style='margin-top:8px;'>{badge(verdict, ENTRY_BADGE)}</div>
            </div>
            """
        )
    st.markdown("<div class='highlight-board'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_fundamental_section(summary: pd.DataFrame) -> None:
    sample = summary.copy()
    if sample.empty:
        st.info("暂无基础数据")
        return
    sample["price"] = sample.get("end_close")
    sample["fair_value_est"] = sample.get("support_level_primary", np.nan) * 1.15
    sample["fair_value_est"].fillna(sample["price"], inplace=True)
    sample["margin_of_safety"] = (
        (sample["fair_value_est"] - sample["price"]) / sample["price"]
    ) * 100
    display_cols = [
        "symbol",
        "theme_category",
        "price",
        "fair_value_est",
        "margin_of_safety",
        "end_pe",
        "entry_recommendation",
    ]
    for col in display_cols:
        if col not in sample.columns:
            sample[col] = np.nan
    sample = sample.sort_values("margin_of_safety", ascending=False).head(12)
    st.dataframe(
        sample[display_cols].rename(
            columns={
                "symbol": "Symbol",
                "theme_category": "Sector",
                "price": "Price",
                "fair_value_est": "Fair Value",
                "margin_of_safety": "Margin of Safety (%)",
                "end_pe": "P/E",
                "entry_recommendation": "Verdict",
            }
        ),
        use_container_width=True,
    )


def render_sector_gap(summary: pd.DataFrame) -> None:
    if "theme_category" not in summary.columns or summary.empty:
        return
    sector = (
        summary.groupby("theme_category")["value_score"]
        .agg(["mean", "count", "max"])
        .reset_index()
        .rename(
            columns={
                "theme_category": "Sector",
                "mean": "Avg Score",
                "count": "Companies",
                "max": "Top Score",
            }
        )
    )
    st.dataframe(sector, use_container_width=True)
    fig = px.bar(
        sector,
        x="Sector",
        y="Avg Score",
        color="Avg Score",
        color_continuous_scale="Greens",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"sector-gap-chart-{uuid4()}",
    )


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


def fetch_holdings() -> pd.DataFrame:
    df = fetch_supabase_table(SUPABASE_HOLDINGS_TABLE)
    if df is None:
        return pd.DataFrame()
    for col in ["cost_price", "shares", "target_shares"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].str.upper()
    return df


def upsert_holding(record: dict) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("未配置 Supabase，无法保存持仓。")
        return False
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_HOLDINGS_TABLE}"
    headers = {
        **SUPABASE_HEADERS,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    try:
        resp = requests.post(url, headers=headers, json=[record], timeout=15)
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"保存持仓失败：{exc}")
        return False


def delete_holding(symbol: str) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("未配置 Supabase，无法删除持仓。")
        return False
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_HOLDINGS_TABLE}"
    headers = SUPABASE_HEADERS
    try:
        resp = requests.delete(
            url,
            headers=headers,
            params={"symbol": f"eq.{symbol}"},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"删除持仓失败：{exc}")
        return False


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


def render_volume_section(history: pd.DataFrame) -> None:
    st.subheader("量能指标速览")
    latest = (
        history.sort_values("date")
        .dropna(subset=["symbol"])
        .groupby("name")
        .last()
    )
    if latest.empty:
        st.info("当前数据源缺少量能指标。")
        return
    min_ratio = st.slider(
        "突增比率阈值（仅显示高于该值的标的）",
        min_value=1.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
    )
    metrics = latest[
        [
            "volume",
            "volume_ma20",
            "volume_spike_ratio",
            "obv",
            "vpt",
            "vwap",
            "ad_line",
        ]
    ].reset_index()
    filtered_metrics = metrics[metrics["volume_spike_ratio"] >= min_ratio].copy()
    if filtered_metrics.empty:
        filtered_metrics = metrics.copy()
    st.dataframe(
        filtered_metrics.rename(
            columns={
                "name": "标的",
                "volume": "当日成交量",
                "volume_ma20": "20日均量",
                "volume_spike_ratio": "突增比率",
                "obv": "OBV",
                "vpt": "VPT",
                "vwap": "VWAP",
                "ad_line": "A/D线",
            }
        ),
        use_container_width=True,
        column_config={
            "突增比率": st.column_config.ProgressColumn(
                "突增比率", min_value=0.0, max_value=3.0, format="%.2f"
            )
        },
    )
    volume_long = metrics.melt(
        id_vars="name",
        value_vars=["volume", "volume_ma20"],
        var_name="类型",
        value_name="数值",
    )
    vol_fig = px.bar(
        volume_long,
        x="name",
        y="数值",
        color="类型",
        barmode="group",
        title="当日成交量 vs 20 日均量",
        color_discrete_map={"volume": COLOR_UP, "volume_ma20": COLOR_DOWN},
    )
    vol_fig.update_layout(legend_title_text="类型")
    st.plotly_chart(vol_fig, use_container_width=True)
    spike_df = metrics.dropna(subset=["volume_spike_ratio"])
    if not spike_df.empty:
        colors = [
            COLOR_UP if val >= 2 else ("#f39c12" if val >= 1.5 else "#95a5a6")
            for val in spike_df["volume_spike_ratio"]
        ]
        spike_fig = px.bar(
            spike_df,
            x="name",
            y="volume_spike_ratio",
            title="成交量突增比率 (>1 表示高于20日均量)",
            color_discrete_sequence=THEME_COLORS,
        )
        spike_fig.update_traces(marker_color=colors)
        st.plotly_chart(spike_fig, use_container_width=True)
    obv_df = history.dropna(subset=["obv"])
    if not obv_df.empty:
        obv_fig = px.line(
            obv_df,
            x="date",
            y="obv",
            color="name",
            title="OBV（能量潮）走势",
            color_discrete_sequence=THEME_COLORS,
        )
        st.plotly_chart(obv_fig, use_container_width=True)


@st.cache_data
def load_history() -> pd.DataFrame:
    supabase_df = fetch_supabase_table(
        "equity_metrics_history",
        order="as_of_date.desc",
        limit=5000,
    )
    use_fallback = False
    if supabase_df is not None and not supabase_df.empty:
        df = supabase_df.rename(columns={"as_of_date": "date"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # 只保留最近 5000 条后按日期排序，避免 1000 行默认上限导致拿到旧数据
        df = df.sort_values("date", ascending=True)
    required_cols = [
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
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    essential = ["close", "close_norm", "close_percentile"]
    if all(df[col].isna().all() for col in essential):
        st.warning(
            "Supabase 历史表缺少关键价格字段，临时改用仓库中的 CSV。请运行 `fetch_equities_fmp.py` "
            "并上传到 Supabase，以恢复云端历史行情。"
        )
        use_fallback = True
    else:
        if "name" not in df.columns:
            df["name"] = df.apply(
                lambda row: f"{row.get('name_en','')}（{row.get('name_cn','')}）"
                if row.get("name_en")
                else row.get("name_cn", row.get("symbol")),
                axis=1,
            )
        if "support_level" not in df.columns and "support_level_primary" in df.columns:
            df["support_level"] = df["support_level_primary"]
        if "date" in df.columns:
            df = df.sort_values("date")
        return df

    # Fallback to repo CSV
    fallback = pd.read_csv(HISTORY_PATH)
    fallback["date"] = pd.to_datetime(fallback["date"], errors="coerce")
    if "support_level" not in fallback.columns and "support_level_primary" in fallback.columns:
        fallback["support_level"] = fallback["support_level_primary"]
    if "name" not in fallback.columns:
        fallback["name"] = fallback.apply(
            lambda row: f"{row.get('name_en','')}（{row.get('name_cn','')}）"
            if row.get("name_en")
            else row.get("name_cn", row.get("symbol")),
            axis=1,
        )
    fallback = fallback.sort_values("date")
    return fallback


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
    for col in ["end_close", "end_close_percentile", "pct_change", "pct_change_7d", "pct_change_30d"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


@st.cache_data
def load_analyst() -> pd.DataFrame:
    """保留占位，未来如需重启分析师预测模块再启用。"""
    return pd.DataFrame()


@st.cache_data
def load_crypto_supports() -> pd.DataFrame:
    df = fetch_supabase_table("crypto_supports")
    if df is None or df.empty:
        if CRYPTO_CSV_PATH.exists():
            df = pd.read_csv(CRYPTO_CSV_PATH)
        else:
            return pd.DataFrame()
    for col in [
        "last_price",
        "ma50",
        "ma200",
        "support_strength",
        "fib_38_2",
        "fib_50",
        "fib_61_8",
        "swing_low",
        "swing_high",
        "swing_range",
        "pct_change_7d",
        "pct_change_30d",
        "distance_ma50_pct",
        "distance_ma200_pct",
        "volume_latest",
        "volume_ma20",
        "volume_spike_ratio",
        "obv",
        "vpt",
        "vwap",
        "ad_line",
        "volume_avg_30d",
        "mvrv",
        "lth_cost_basis",
        "exchange_netflow",
        "asopr",
        "funding_rate",
        "fear_greed",
        "stablecoin_exchange_balance",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def render_equity_content() -> None:
    st.caption("数据来源：Supabase（`equity_metrics` / `equity_metrics_history` / `holdings`）")
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
    summary_full = summary.copy()
    history_full = history.copy()
    render_overview_cards(summary_full, history_full)

    st.markdown("### 快速筛选")
    expected_cols = {
        "refresh_interval_days",
        "next_refresh_date",
        "current_ps",
        "forward_pe",
        "end_close",
        "end_close_percentile",
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

    st.markdown("<div class='subtle-card'>", unsafe_allow_html=True)
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.2, 1.2, 1, 1])
    market_options = sorted(summary["market"].dropna().unique())
    with filter_col1:
        selected_markets = st.multiselect(
            "市场区域",
            market_options,
            default=market_options,
        )
    if selected_markets:
        summary = summary[summary["market"].isin(selected_markets)]
        history = history[history["market"].isin(selected_markets)]
    else:
        summary = summary.iloc[0:0]
        history = history.iloc[0:0]

    category_options = sorted(summary["theme_category"].dropna().unique())
    with filter_col2:
        selected_categories = st.multiselect(
            "产业链层级",
            category_options,
            default=category_options,
        )
    if selected_categories:
        summary = summary[summary["theme_category"].isin(selected_categories)]
        history = align_history_to_summary(history, summary)

    entry_series = summary["entry_recommendation"] if "entry_recommendation" in summary.columns else pd.Series(dtype=str)
    entry_options = sorted(entry_series.dropna().unique().tolist())
    selected_entry: list[str] = entry_options.copy()
    with filter_col3:
        if entry_options:
            selected_entry = st.multiselect(
                "入场结论",
                entry_options,
                default=entry_options,
            )
        else:
            st.caption("暂无入场分类")
    if selected_entry:
        summary = summary[summary["entry_recommendation"].isin(selected_entry)]
        history = align_history_to_summary(history, summary)

    with filter_col4:
        search_kw = st.text_input("🔍 搜索公司/代码", placeholder="输入公司名称或股票代码")
    st.markdown("</div>", unsafe_allow_html=True)

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

    render_active_filters(selected_markets, selected_categories, selected_entry if selected_entry else [], search_kw or "")

    if summary.empty or history.empty:
        st.warning("所选条件暂无数据")
        return

    if "value_score" in summary.columns:
        summary = summary.sort_values("value_score", ascending=False)

    render_highlight_section(summary)

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

    filtered_categories = sorted(summary["theme_category"].dropna().unique())
    if filtered_categories:
        category_tabs = st.tabs(filtered_categories)
        for cat, tab in zip(filtered_categories, category_tabs):
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

    st.markdown("### 估值锚点")
    render_fundamental_section(summary)
    st.markdown("### 行业得分分布")
    render_sector_gap(summary)

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
    st.plotly_chart(norm_fig, width="stretch")

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
        st.plotly_chart(pe_fig, width="stretch")

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
    st.plotly_chart(percentile_fig, width="stretch")

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
        st.plotly_chart(pct_fig, width="stretch")
    with cols[1]:
        pe_bar = px.bar(
            summary_display.dropna(subset=["end_pe"]),
            x="name",
            y="end_pe",
            title="期末 PE",
            color_discrete_sequence=THEME_COLORS,
        )
        st.plotly_chart(pe_bar, width="stretch")
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
        st.plotly_chart(perc_bar, width="stretch")

    render_fundamental_section(summary)
    render_sector_gap(summary)
    render_volume_section(filtered)
    render_holdings_panel(summary_full, history_full)


def render_equity_dashboard() -> None:
    render_page_nav("equity")
    st.title("价值锚点监控仪表盘")
    equity_tab, crypto_tab = st.tabs(["股票面板", "加密面板"])
    with equity_tab:
        render_equity_content()
    with crypto_tab:
        render_crypto_dashboard()

def render_holdings_panel(summary: pd.DataFrame, history: pd.DataFrame) -> None:
    st.subheader("持仓与风控模块")
    holdings = fetch_holdings()
    if summary.empty:
        st.info("暂无股票数据")
        return
    # 去重后再建索引，避免同一 symbol 多行导致 Series/ndarray 异形
    summary_unique = summary.drop_duplicates(subset=["symbol"], keep="last")
    symbol_options = sorted(summary_unique["symbol"].dropna().unique())
    with st.form("holding_form"):
        c1, c2, c3, c4 = st.columns(4)
        selected_symbol = c1.selectbox("标的代码", symbol_options)
        cost_price = c2.number_input("成本价", min_value=0.0, value=0.0, step=0.1)
        shares = c3.number_input("当前持仓(股)", min_value=0.0, value=0.0, step=10.0)
        target_shares = c4.number_input("目标持仓(股)", min_value=0.0, value=shares, step=10.0)
        submitted = st.form_submit_button("保存持仓")
        if submitted:
            record = {
                "symbol": selected_symbol.upper(),
                "cost_price": cost_price or None,
                "shares": shares or None,
                "target_shares": target_shares or None,
            }
            if upsert_holding(record):
                st.success("持仓已保存")
                st.experimental_rerun()

    if holdings.empty:
        st.info("尚未录入持仓")
        return

    capital = st.number_input(
        "战术资金净值（用于计算最大亏损 1.5%）",
        min_value=0.0,
        value=200000.0,
        step=10000.0,
    )

    latest_history = (
        history.sort_values("date").dropna(subset=["symbol"]).groupby("symbol").last()
    )
    summary_lookup = summary_unique.set_index("symbol")
    rows = []

    for _, row in holdings.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        cost = row.get("cost_price")
        share_count = row.get("shares")
        target = row.get("target_shares")

        summary_row = summary_lookup.loc[symbol] if symbol in summary_lookup.index else None
        hist_row = latest_history.loc[symbol] if symbol in latest_history.index else None
        current_price = summary_row["end_close"] if summary_row is not None and "end_close" in summary_row else None
        ma200 = hist_row["ma200"] if hist_row is not None and "ma200" in hist_row else None
        support_primary = hist_row["support_level"] if hist_row is not None and "support_level" in hist_row else None
        support_secondary = hist_row["support_level_secondary"] if hist_row is not None and "support_level_secondary" in hist_row else None
        entry_price = (
            np.nanmean([current_price, ma200]) if current_price is not None and ma200 is not None else current_price
        )
        stop_price = ma200 * 0.97 if ma200 and not pd.isna(ma200) else None
        tp_50 = cost * 1.5 if cost else None
        tp_100 = cost * 2 if cost else None
        max_loss_capital = capital * 0.015 if capital else None
        allowed_shares = None
        if cost and stop_price and cost > stop_price and max_loss_capital:
            allowed_shares = max_loss_capital / (cost - stop_price)

        notes = []
        if summary_row is not None:
            if summary_row.get("entry_recommendation"):
                notes.append(summary_row["entry_recommendation"])
            if summary_row.get("value_score"):
                notes.append(f"得分 {summary_row['value_score']:.1f}")

        rows.append(
            {
                "标的": symbol,
                "当前价": current_price,
                "成本价": cost,
                "建议首仓": entry_price,
                "200日均线": ma200,
                "主支撑": support_primary,
                "次支撑": support_secondary,
                "建议止损": stop_price,
                "止盈50%": tp_50,
                "止盈100%": tp_100,
                "当前持仓(股)": share_count,
                "目标持仓(股)": target,
                "建议最大仓位(股)": allowed_shares,
                "备注": " / ".join(notes),
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    for _, row in holdings.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        if st.button(f"删除 {symbol}", key=f"delete_{symbol}"):
            if delete_holding(symbol):
                st.success(f"{symbol} 已删除")
                st.experimental_rerun()


def render_crypto_dashboard() -> None:
    st.subheader("加密资产支撑面板")
    crypto_df = load_crypto_supports()
    if crypto_df.empty:
        st.info("暂无加密资产数据。请先运行 `python3 fetch_crypto_supports.py` 并同步到 Supabase。")
        return
    crypto_df["symbol"] = crypto_df["symbol"].str.upper()
    trend_options = sorted(crypto_df["ma_trend"].dropna().unique().tolist())
    if trend_options:
        selected_trend = st.multiselect("趋势筛选", trend_options, default=trend_options)
        if selected_trend:
            crypto_df = crypto_df[crypto_df["ma_trend"].isin(selected_trend)]
    search_kw = st.text_input("搜索加密资产")
    if search_kw:
        k = search_kw.strip().lower()
        if k:
            name_series = (
                crypto_df["name"].astype(str)
                if "name" in crypto_df.columns
                else pd.Series("", index=crypto_df.index)
            )
            crypto_df = crypto_df[
                crypto_df["symbol"].str.lower().str.contains(k)
                | name_series.str.lower().str.contains(k)
            ]
    options = sorted(crypto_df["symbol"].unique())
    selected = st.multiselect("选择资产", options, default=options)
    if selected:
        crypto_df = crypto_df[crypto_df["symbol"].isin(selected)]
    display_cols = [
        "symbol",
        "name",
        "pct_change_7d",
        "pct_change_30d",
        "volume_spike_ratio",
        "obv",
        "vpt",
        "vwap",
        "support_strength",
        "fear_greed",
        "funding_rate",
        "recent_supports",
    ]
    missing = [c for c in display_cols if c not in crypto_df.columns]
    for col in missing:
        crypto_df[col] = np.nan
    # Price summary cards
    price_cols = st.columns(min(len(crypto_df), 4) or 1)
    for idx, (_, row) in enumerate(crypto_df.head(4).iterrows()):
        delta = row.get("pct_change_7d")
        delta_str = f"{delta:.2f}%" if pd.notna(delta) else "N/A"
        price_cols[idx % len(price_cols)].metric(
            f"{row['symbol']} 现价",
            f"${row['last_price']:.2f}" if pd.notna(row["last_price"]) else "N/A",
            f"7日 {delta_str}",
        )

    st.dataframe(
        crypto_df[display_cols].rename(
            columns={
                "symbol": "代码",
                "name": "资产",
                "pct_change_7d": "7日涨跌%",
                "pct_change_30d": "30日涨跌%",
                "volume_spike_ratio": "成交量突增比率",
                "support_strength": "支撑强度",
                "fear_greed": "恐贪指数",
                "funding_rate": "资金费率",
                "recent_supports": "近期支撑",
            }
        ),
        use_container_width=True,
        column_config={
            "7日涨跌%": st.column_config.NumberColumn("7日涨跌%", format="%.2f"),
            "30日涨跌%": st.column_config.NumberColumn("30日涨跌%", format="%.2f"),
            "成交量突增比率": st.column_config.ProgressColumn(
                "成交量突增比率", min_value=0.0, max_value=3.0, format="%.2f"
            ),
        },
    )
    if {"volume_spike_ratio", "symbol"}.issubset(crypto_df.columns):
        ratio_fig = px.bar(
            crypto_df,
            x="symbol",
            y="volume_spike_ratio",
            title="加密资产成交量突增比率",
            color_discrete_sequence=THEME_COLORS,
        )
        st.plotly_chart(ratio_fig, use_container_width=True)
    symbols = crypto_df["symbol"].unique().tolist()
    if {"symbol", "last_price", "ma50", "ma200"}.issubset(crypto_df.columns):
        st.markdown("### 现价 / MA50 / MA200（按币种分表）")
        # 按币种分成最多 4 个表，避免价格量级差异放在一个表里难以阅读
        groups: dict[int, list[str]] = {}
        for idx, sym in enumerate(sorted(symbols)):
            groups.setdefault(idx % 4, []).append(sym)
        cols = st.columns(min(4, len(groups)))
        base_cols = [
            "symbol",
            "last_price",
            "ma50",
            "ma200",
            "distance_ma50_pct",
            "distance_ma200_pct",
        ]
        for slot, syms in groups.items():
            subset = crypto_df[crypto_df["symbol"].isin(syms)][base_cols].copy()
            subset = subset.rename(
                columns={
                    "symbol": "资产",
                    "last_price": "现价(USD)",
                    "ma50": "MA50",
                    "ma200": "MA200",
                    "distance_ma50_pct": "距MA50%",
                    "distance_ma200_pct": "距MA200%",
                }
            )
            subset["现价(USD)"] = subset["现价(USD)"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
            subset["MA50"] = subset["MA50"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
            subset["MA200"] = subset["MA200"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
            subset["距MA50%"] = subset["距MA50%"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
            subset["距MA200%"] = subset["距MA200%"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
            cols[slot].dataframe(subset, hide_index=True, use_container_width=True)
            # 图表展示：同组币种的现价/MA50/MA200 对比
            numeric_subset = crypto_df[crypto_df["symbol"].isin(syms)][["symbol", "last_price", "ma50", "ma200"]]
            price_long = numeric_subset.melt(
                id_vars="symbol",
                value_vars=["last_price", "ma50", "ma200"],
                var_name="指标",
                value_name="数值",
            )
            chart = px.bar(
                price_long,
                x="symbol",
                y="数值",
                color="指标",
                barmode="group",
                title="现价 / MA50 / MA200 对比",
                color_discrete_sequence=THEME_COLORS,
            )
            cols[slot].plotly_chart(chart, use_container_width=True, key=f"crypto-ma-chart-{slot}")



def main() -> None:
    render_equity_dashboard()


if __name__ == "__main__":
    main()
