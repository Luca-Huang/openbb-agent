#!/usr/bin/env python3
"""
LLM 版适配脚本（独立运行，不改主应用）：
1) 从 Supabase 拉取股票特征（equity_metrics + equity_metrics_history 最新一行）
2) 组装 Prompt 调用 OpenAI 兼容的聊天接口，生成信号
3) 将结果写入 Supabase 表 `ta_signals`（需提前创建）

环境变量：
- SUPABASE_URL / SUPABASE_KEY
- OPENAI_API_KEY            （必须）
- OPENAI_BASE_URL           （可选，默认 https://api.openai.com/v1，第三方兼容接口可覆盖）
- MODEL_NAME                （可选，默认 gpt-4o-mini）

注意：请自行设置 API Key，不要将 Key 写入代码或提交仓库。
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import requests


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
}


def require_env() -> None:
    missing = [k for k, v in {
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_KEY": SUPABASE_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }.items() if not v]
    if missing:
        raise SystemExit(f"缺少环境变量：{', '.join(missing)}")


def fetch_table(table: str, select: str = "*", order: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    params: Dict[str, str | int] = {"select": select}
    if order:
        params["order"] = order
    if limit:
        params["limit"] = limit
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=SUPABASE_HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def call_llm(messages: list[dict]) -> str:
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content


def build_prompt(records: List[Dict]) -> list[dict]:
    sys = (
        "你是交易助手，请基于给定的结构化特征生成简洁信号。"
        "仅输出 JSON 数组，每个元素包含 symbol, signal, confidence, rationale。"
        "signal 取 buy/watch/hold/sell 四选一，confidence 0-1 之间。"
    )
    user_lines = []
    for r in records:
        user_lines.append(
            f"symbol={r['symbol']}, close={r['close']}, ma200={r['ma200']}, "
            f"support={r['support']}, value_score={r['value_score']}, entry={r['entry_recommendation']}"
        )
    user = "请生成信号，格式示例: " \
           '[{"symbol":"ABC","signal":"watch","confidence":0.7,"rationale":"简要理由"}]. 数据：\n' + "\n".join(user_lines)
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def parse_llm_json(text: str) -> List[Dict]:
    try:
        return json.loads(text)
    except Exception:
        # 尝试截取代码块
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                try:
                    return json.loads(p)
                except Exception:
                    continue
        raise ValueError("LLM 输出无法解析为 JSON")


def upsert_signals(records: List[Dict]) -> None:
    if not records:
        print("No signals to upsert")
        return
    params = {"on_conflict": "symbol,as_of_date"}
    headers = {
        **SUPABASE_HEADERS,
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
    require_env()
    summary = fetch_table(
        "equity_metrics",
        select="symbol,value_score,entry_recommendation",
    )
    history = fetch_table(
        "equity_metrics_history",
        select="symbol,as_of_date,close,support_level_primary,support_level,ma200",
        order="as_of_date.desc",
        limit=5000,
    )
    if summary.empty or history.empty:
        raise SystemExit("summary/history 为空，请先准备数据")
    history["as_of_date"] = pd.to_datetime(
        history.get("as_of_date", history.get("date")), errors="coerce"
    ).dt.date
    latest = history.sort_values("as_of_date").dropna(subset=["symbol"]).groupby("symbol").last()

    prompt_records: List[Dict] = []
    for _, row in summary.iterrows():
        sym = row.get("symbol")
        if not sym or sym not in latest.index:
            continue
        latest_row = latest.loc[sym]
        prompt_records.append(
            {
                "symbol": sym,
                "close": latest_row.get("close"),
                "ma200": latest_row.get("ma200"),
                "support": latest_row.get("support_level_primary") or latest_row.get("support_level"),
                "value_score": row.get("value_score"),
                "entry_recommendation": row.get("entry_recommendation"),
            }
        )
    if not prompt_records:
        raise SystemExit("无可用标的生成 Prompt")

    messages = build_prompt(prompt_records[:30])  # 限制一次请求的标的数，避免超长
    llm_text = call_llm(messages)
    llm_records = parse_llm_json(llm_text)

    today = date.today().isoformat()
    out = []
    for r in llm_records:
        sym = r.get("symbol")
        if not sym:
            continue
        out.append(
            {
                "symbol": sym,
                "as_of_date": today,
                "signal": r.get("signal"),
                "confidence": r.get("confidence"),
                "rationale": r.get("rationale"),
            }
        )
    upsert_signals(out)


if __name__ == "__main__":
    main()
