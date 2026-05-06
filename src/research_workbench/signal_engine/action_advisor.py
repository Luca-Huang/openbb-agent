"""Per-stock action advisor for all watchlist symbols.

Produces a structured "next action" recommendation for every stock in the
watchlist, regardless of whether it's held.  The action is derived from the
signal state, trigger type, conviction trajectory, and price position.

Usage::

    from research_workbench.signal_engine.action_advisor import advise_actions
    actions_df = advise_actions(signals_df, history_df)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .radar import compute_exit_plan


# ---------------------------------------------------------------------------
# Action classification
# ---------------------------------------------------------------------------

def _advise_single(row: pd.Series) -> dict[str, Any]:
    """Produce an action recommendation for a single signal row."""

    state = str(row.get("signal_state", "watch"))
    trigger = row.get("trigger_type") or ""
    conviction = pd.to_numeric(row.get("conviction_score"), errors="coerce")
    zone_dist = pd.to_numeric(row.get("zone_distance"), errors="coerce")
    close = pd.to_numeric(row.get("trigger_price", row.get("close")), errors="coerce")
    stop = pd.to_numeric(row.get("stop_price"), errors="coerce")
    target_low = pd.to_numeric(row.get("target_zone_low"), errors="coerce")
    target_high = pd.to_numeric(row.get("target_zone_high"), errors="coerce")
    event_risk = pd.to_numeric(row.get("event_risk_score"), errors="coerce")

    symbol = str(row.get("symbol", ""))
    name = row.get("name", symbol)

    # Base result
    result: dict[str, Any] = {
        "symbol": symbol,
        "name": name,
        "signal_state": state,
        "action": "",
        "action_detail": "",
        "action_priority": 4,  # 1=urgent, 2=important, 3=normal, 4=low
        "suggested_entry": None,
        "suggested_stop": None,
        "suggested_tp1": None,
        "suggested_tp2": None,
        "suggested_trailing": None,
        "position_sizing_note": "",
    }

    # ---- invalidated ----
    if state == "invalidated":
        result["action"] = "暂停关注"
        result["action_detail"] = (
            "价格已跌破止损位，当前结构被破坏。"
            "建议暂时移出重点观察，等待价格重新站上关键均线后再评估。"
        )
        result["action_priority"] = 3
        return result

    # ---- triggered ----
    if state == "triggered":
        # Build an entry plan
        entry_price = float(close) if pd.notna(close) else None
        stop_price = float(stop) if pd.notna(stop) else None

        if entry_price and stop_price and stop_price < entry_price:
            plan = compute_exit_plan(
                entry_price=entry_price,
                stop_price=stop_price,
            )
            result["suggested_entry"] = entry_price
            result["suggested_stop"] = stop_price
            result["suggested_tp1"] = plan["take_profit_1"]
            result["suggested_tp2"] = plan["take_profit_2"]
            result["suggested_trailing"] = plan["trailing_stop"]

            risk_pct = (entry_price - stop_price) / entry_price * 100
            result["position_sizing_note"] = (
                f"单笔风险 {risk_pct:.1f}%。"
                f"建议用总资金的 1-2% 作为本笔最大亏损额来反算仓位。"
            )

        if trigger == "breakout":
            result["action"] = "可考虑入场（突破）"
            result["action_detail"] = (
                f"价格放量突破 20 日新高，触发 breakout 信号。"
                f"如决定入场，建议在当前价附近分批建仓，"
                f"首仓不超过计划仓位的 50%。"
            )
        elif trigger == "pullback":
            result["action"] = "可考虑入场（回踩）"
            result["action_detail"] = (
                f"价格回踩至 MA50 附近且结构未破，触发 pullback 信号。"
                f"这类机会止损空间通常较小，风险收益比较优。"
                f"建议在支撑位附近分批建仓。"
            )
        else:
            result["action"] = "可考虑入场"
            result["action_detail"] = "满足触发条件，可评估入场。"

        # Event risk warning
        if pd.notna(event_risk) and event_risk >= 10:
            result["action_detail"] += (
                f"\n⚠️ 近期有高风险事件（事件风险分={event_risk:.0f}），"
                f"建议等事件落地后再决定。"
            )
            result["action_priority"] = 2
        else:
            result["action_priority"] = 1

        return result

    # ---- near_zone ----
    if state == "near_zone":
        result["action"] = "准备入场计划"

        if pd.notna(zone_dist):
            dist_pct = zone_dist * 100
            result["action_detail"] = (
                f"价格距目标区间仅 {dist_pct:.1f}%，已进入观察窗口。"
                f"建议现在就制定好入场计划：\n"
                f"1. 确认目标买入区间"
            )
            if pd.notna(target_low):
                result["action_detail"] += f"（{target_low:.2f}"
                if pd.notna(target_high):
                    result["action_detail"] += f" ~ {target_high:.2f}"
                result["action_detail"] += "）"
            result["action_detail"] += (
                f"\n2. 预设止损位和仓位大小"
                f"\n3. 设置价格提醒，等待触发信号确认"
            )
        else:
            result["action_detail"] = (
                "接近观察区间，建议制定入场计划并设置价格提醒。"
            )

        result["action_priority"] = 2
        return result

    # ---- watch (default) ----
    result["action_priority"] = 4

    if pd.notna(conviction) and conviction >= 40:
        result["action"] = "重点跟踪"
        result["action_detail"] = (
            f"信心分较高（{conviction:.0f}），虽未触发但基本面和估值有吸引力。"
            f"建议保持关注，等待技术面配合（价格接近支撑/均线回踩）。"
        )
        result["action_priority"] = 3
    elif pd.notna(conviction) and conviction >= 20:
        result["action"] = "常规观察"
        result["action_detail"] = (
            f"信心分中等（{conviction:.0f}），条件尚未成熟。"
            f"保持在自选列表中，每周复查一次即可。"
        )
    else:
        result["action"] = "低优先级"
        result["action_detail"] = (
            "当前无明确的技术或基本面催化剂。"
            "可保留在列表中但无需频繁关注。"
        )

    return result


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def advise_actions(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Generate action recommendations for all signals.

    Returns a DataFrame with columns:
    symbol, name, signal_state, action, action_detail, action_priority,
    suggested_entry, suggested_stop, suggested_tp1, suggested_tp2,
    suggested_trailing, position_sizing_note.
    """
    if signals_df.empty:
        return pd.DataFrame(columns=[
            "symbol", "name", "signal_state",
            "action", "action_detail", "action_priority",
            "suggested_entry", "suggested_stop",
            "suggested_tp1", "suggested_tp2", "suggested_trailing",
            "position_sizing_note",
        ])

    rows = [_advise_single(row) for _, row in signals_df.iterrows()]
    out = pd.DataFrame(rows)
    return out.sort_values("action_priority").reset_index(drop=True)


def format_action_text(action_row: pd.Series) -> str:
    """Format an action row into readable text for UI or email."""
    parts = [
        f"【{action_row['action']}】",
        str(action_row.get("action_detail", "")),
    ]

    entry = action_row.get("suggested_entry")
    stop = action_row.get("suggested_stop")
    tp1 = action_row.get("suggested_tp1")
    tp2 = action_row.get("suggested_tp2")

    if entry and pd.notna(entry):
        parts.append("")
        parts.append(f"入场参考: {float(entry):.2f}")
        if stop and pd.notna(stop):
            parts.append(f"止损位: {float(stop):.2f}")
        if tp1 and pd.notna(tp1):
            parts.append(f"止盈1 (1R): {float(tp1):.2f}")
        if tp2 and pd.notna(tp2):
            parts.append(f"止盈2 (2R): {float(tp2):.2f}")

    sizing = action_row.get("position_sizing_note", "")
    if sizing:
        parts.append("")
        parts.append(sizing)

    return "\n".join(parts)
