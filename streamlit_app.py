#!/usr/bin/env python3
"""Interactive dashboard for the OpenBB three-month dataset."""

from __future__ import annotations

from datetime import date, timedelta
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="价值锚点仪表盘", layout="wide")

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


@st.cache_data
def load_history() -> pd.DataFrame:
    df = pd.read_csv(HISTORY_PATH)
    return df


@st.cache_data
def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH)
    return df


@st.cache_data
def load_analyst() -> pd.DataFrame:
    if not ANALYST_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ANALYST_PATH, parse_dates=["date（财年/报告期）"])
    return df


def main() -> None:
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
    analyst = load_analyst()

    market_options = sorted(summary["market"].dropna().unique())
    selected_markets = st.multiselect(
        "选择市场",
        market_options,
        default=market_options,
    )
    summary = summary[summary["market"].isin(selected_markets)]
    history = history[history["market"].isin(selected_markets)]
    if summary.empty or history.empty:
        st.warning("所选市场暂无数据。")
        return

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

    st.subheader("入场信号评估")
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
    else:
        missing = required_cols - set(summary.columns)
        st.info(
            f"当前汇总缺少字段 {missing}，请确认 `openbb_three_months.py` 版本已更新并重新运行。"
        )

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


if __name__ == "__main__":
    main()
