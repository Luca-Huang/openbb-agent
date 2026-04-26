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
import re
import os
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import requests


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
# 可通过环境变量覆盖路径，某些兼容网关需要 /openai/v1/chat/completions
OPENAI_ENDPOINT_PATH = os.environ.get("OPENAI_ENDPOINT_PATH", "/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/responses")
ARK_API_KEY = os.environ.get("ARK_API_KEY")
ARK_TIMEOUT = int(os.environ.get("ARK_TIMEOUT", "180"))
ARK_RETRIES = int(os.environ.get("ARK_RETRIES", "2"))

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


def _full_url(path: str) -> str:
    base = OPENAI_BASE_URL.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def call_llm(messages: list[dict]) -> str:
    # 分支：字节火山 Ark 接口
    if LLM_PROVIDER == "ark":
        if not ARK_API_KEY:
            raise RuntimeError("缺少 ARK_API_KEY 环境变量")
        headers = {
            "Authorization": f"Bearer {ARK_API_KEY}",
            "Content-Type": "application/json",
        }
        # 将所有消息合并为一个 user 文本，简单串联
        texts: list[str] = []
        for m in messages:
            if m.get("content"):
                texts.append(str(m["content"]))
        merged = "\n".join(texts) if texts else "请生成交易信号"
        body = {
            "model": MODEL_NAME,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": merged},
                    ],
                }
            ],
        }
        last_err: Exception | None = None
        for attempt in range(ARK_RETRIES + 1):
            try:
                resp = requests.post(ARK_BASE_URL, headers=headers, data=json.dumps(body), timeout=ARK_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                text = None
                outputs = data.get("output") or []
                # 先找 type=message，再找其他
                for item in outputs:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "message":
                        continue
                    for c in item.get("content") or []:
                        if isinstance(c, dict) and c.get("text"):
                            text = c["text"]
                            break
                    if text:
                        break
                # 如果没找到 message，再回退 summary / 其他 content
                if not text:
                    for item in outputs:
                        if not isinstance(item, dict):
                            continue
                        for c in item.get("content") or []:
                            if isinstance(c, dict) and c.get("text"):
                                text = c["text"]
                                break
                        if text:
                            break
                        for s in item.get("summary") or []:
                            if isinstance(s, dict) and s.get("text"):
                                text = s["text"]
                                break
                        if text:
                            break
                if not text and data.get("output_text"):
                    text = data["output_text"]
                if not text:
                    raise KeyError("content/summary/output_text missing")
                return text
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < ARK_RETRIES:
                    continue
                raise RuntimeError(f"无法解析 Ark 响应或调用失败: {data if 'data' in locals() else exc}") from exc

    # 默认：OpenAI 兼容接口
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

    paths_to_try = [OPENAI_ENDPOINT_PATH]
    # 兼容部分网关使用 /openai/v1/chat/completions
    if OPENAI_ENDPOINT_PATH != "/openai/v1/chat/completions":
        paths_to_try.append("/openai/v1/chat/completions")

    last_error = None
    for path in paths_to_try:
        try:
            resp = requests.post(_full_url(path), headers=headers, data=json.dumps(body), timeout=180)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise last_error if last_error else RuntimeError("LLM 调用失败")


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
    def _try_load(s: str):
        s = s.strip()
        return json.loads(s)

    # 1) 直接尝试整体 JSON
    try:
        return _try_load(text)
    except Exception:
        pass

    # 2) 尝试从代码块中提取
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            try:
                return _try_load(p)
            except Exception:
                continue

    # 3) 尝试用正则提取数组片段
    m = re.search(r"\[\s*{.*}\s*]", text, flags=re.DOTALL)
    if m:
        snippet = m.group(0)
        try:
            return _try_load(snippet)
        except Exception:
            pass

    # 4) 退而求其次：截取首尾中括号，尝试单引号转双引号
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        for candidate in (snippet, snippet.replace("'", '"')):
            try:
                return _try_load(candidate)
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
    # 拉取概要表；用 * 避免因缺列报 400，然后在下面补齐缺失字段
    summary = fetch_table(
        "equity_metrics",
        select="*",
    )
    history = fetch_table(
        "equity_metrics_history",
        select="*",
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
        close_val = latest_row.get("close") or row.get("end_close")
        prompt_records.append(
            {
                "symbol": sym,
                "close": close_val,
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
