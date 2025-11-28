#!/usr/bin/env python3
"""
临时适配脚本（示例）：从 Supabase 拉取最新股票特征 -> 生成简单信号 -> 回写 ta_signals。

说明：
- 未调用 TradingAgents-CN 的 LLM，仅做占位/验通路。
- 使用 Supabase REST（需 SUPABASE_URL / SUPABASE_KEY 环境变量）。
- 依赖 ta_signals 表已存在：
    create table if not exists ta_signals (
      symbol text not null,
      as_of_date date not null,
      signal text,
      confidence numeric,
      rationale text,
      created_at timestamptz default now(),
      primary key (symbol, as_of_date)
    );
- 不修改主应用任何逻辑。
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import List, Dict

import numpy as np
import pandas as pd
import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
}


def fetch_table(table: str, select: str = "*", limit: int | None = None) -> pd.DataFrame:
    params = {"select": select}
    if limit:
        params["limit"] = limit
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data)


def generate_stub_signals(summary: pd.DataFrame, history: pd.DataFrame) -> List[Dict]:
    """简单规则信号：仅为打通通路，后续可替换为 TradingAgents-CN LLM 输出。"""
    out: List[Dict] = []
    if "symbol" not in summary.columns or summary.empty:
        return out
    today = date.today().isoformat()

    # 取最新价/支撑/均线
    if not history.empty:
        history["as_of_date"] = pd.to_datetime(history.get("as_of_date", history.get("date")), errors="coerce").dt.date
        latest = history.sort_values("as_of_date").dropna(subset=["symbol"]).groupby("symbol").last()
    else:
        latest = pd.DataFrame()

    for _, row in summary.iterrows():
        sym = row.get("symbol")
        if not sym:
            continue
        score = row.get("value_score")
        support = None
        ma200 = None
        close = None
        if not latest.empty and sym in latest.index:
            close = latest.loc[sym].get("close")
            support = latest.loc[sym].get("support_level_primary") or latest.loc[sym].get("support_level")
            ma200 = latest.loc[sym].get("ma200")

        # 简单规则：有支撑且分位低则建议关注
        signal = "hold"
        confidence = 0.3
        rationale_parts = []
        if pd.notna(score):
            rationale_parts.append(f"value_score={score:.1f}")
        if pd.notna(close) and pd.notna(support):
            dist = (close - support) / close if close else np.nan
            rationale_parts.append(f"close={close:.2f},support={support:.2f},dist={dist:.2% if pd.notna(dist) else 'N/A'}")
            if pd.notna(dist) and dist <= 0.1:
                signal = "watch"
                confidence = 0.5
        if pd.notna(score) and score >= 30:
            signal = "watch"
            confidence = max(confidence, 0.6)
        if pd.notna(ma200) and pd.notna(close) and close > ma200:
            rationale_parts.append("above_ma200")
            confidence = max(confidence, 0.7)

        out.append(
            {
                "symbol": sym,
                "as_of_date": today,
                "signal": signal,
                "confidence": round(confidence, 2),
                "rationale": "; ".join(rationale_parts) if rationale_parts else "stub rule",
            }
        )
    return out


def upsert_signals(records: List[Dict]) -> None:
    if not records:
        print("No records to upsert")
        return
    params = {"on_conflict": "symbol,as_of_date"}
    headers = {
        **HEADERS,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/ta_signals",
        headers=headers,
        params=params,
        data=json.dumps(records),
        timeout=30,
    )
    resp.raise_for_status()
    print(f"Upserted {len(records)} rows to ta_signals")


def main() -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY env vars")
    summary = fetch_table("equity_metrics")
    history = fetch_table("equity_metrics_history", select="symbol,as_of_date,date,close,support_level_primary,support_level,ma200")
    records = generate_stub_signals(summary, history)
    upsert_signals(records)


if __name__ == "__main__":
    main()
