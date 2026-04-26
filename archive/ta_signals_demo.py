#!/usr/bin/env python3
"""Minimal Streamlit page to view TradingAgents signals without touching main app."""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

import pandas as pd
import requests
import streamlit as st

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


def fetch_table(
    table: str, select: str = "*", order: Optional[str] = None, limit: Optional[int] = None
) -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("请先设置环境变量 SUPABASE_URL / SUPABASE_KEY")
        st.stop()
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    params: dict[str, str | int] = {"select": select}
    if order:
        params["order"] = order
    if limit:
        params["limit"] = limit
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=headers,
        params=params,
        timeout=30,
    )
    if resp.status_code >= 400:
        st.error(f"{table} 读取失败：{resp.text}")
        st.stop()
    data = resp.json()
    return pd.DataFrame(data)


def main() -> None:
    st.set_page_config(page_title="TA Signals Demo", layout="wide")
    st.title("TradingAgents 信号预览（独立页面）")
    st.caption("数据来源：Supabase `ta_signals` / `equity_metrics`，不影响主应用")

    if not SUPABASE_URL or not SUPABASE_KEY:
        st.warning("请在终端导出 SUPABASE_URL / SUPABASE_KEY 后再运行。")
        return

    signals = fetch_table("ta_signals", order="as_of_date.desc", limit=500)
    if signals.empty:
        st.info("`ta_signals` 当前为空。请先运行适配脚本写入信号。")
        return

    signals["as_of_date"] = pd.to_datetime(signals["as_of_date"], errors="coerce").dt.date
    latest_date = signals["as_of_date"].max()
    st.write(f"最新信号日期：{latest_date or '未知'} · 总行数：{len(signals)}")

    # Filters
    symbols = sorted(signals["symbol"].dropna().unique())
    cols = st.columns(3)
    with cols[0]:
        selected_symbols = st.multiselect("筛选标的", symbols, default=symbols)
    with cols[1]:
        min_conf = st.slider("最低置信度", 0.0, 1.0, 0.0, 0.05)
    with cols[2]:
        only_latest = st.checkbox("仅看最新日期", value=True)

    df = signals.copy()
    if selected_symbols:
        df = df[df["symbol"].isin(selected_symbols)]
    if "confidence" in df.columns:
        df = df[df["confidence"].fillna(0) >= min_conf]
    if only_latest and latest_date:
        df = df[df["as_of_date"] == latest_date]

    df = df.sort_values(["as_of_date", "symbol"], ascending=[False, True])
    st.dataframe(
        df.rename(
            columns={
                "symbol": "代码",
                "as_of_date": "日期",
                "signal": "信号",
                "confidence": "置信度",
                "rationale": "理由",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # 简单对接概要表，展示最近收盘/支撑
    st.markdown("### 参考：最新价/支撑 (equity_metrics)")
    summary = fetch_table(
        "equity_metrics",
        select="symbol,end_close,support_level_primary,support_level_secondary,value_score,entry_recommendation",
    )
    summary = summary.rename(
        columns={
            "symbol": "代码",
            "end_close": "最新收盘",
            "support_level_primary": "主支撑",
            "support_level_secondary": "次支撑",
            "value_score": "价值得分",
            "entry_recommendation": "入场结论",
        }
    )
    if selected_symbols:
        summary = summary[summary["代码"].isin(selected_symbols)]
    st.dataframe(summary, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
