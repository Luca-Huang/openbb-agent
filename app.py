from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from research_workbench.config import default_settings
from research_workbench.ingestion.watchlist import load_watchlist, watchlist_symbols
from research_workbench.outputs.files import (
    load_history,
    load_manual_events,
    load_radar,
    load_summary,
)
from research_workbench.signal_engine.action_advisor import advise_actions, format_action_text
from research_workbench.signal_engine.changes import detect_daily_changes
from research_workbench.signal_engine.holdings_tracker import (
    analyze_holding,
    diff_against_previous,
    load_holdings,
    load_previous_snapshot,
    save_snapshot,
)
from research_workbench.signal_engine.radar import (
    DEFAULT_RADAR_CONFIG,
    add_radar_features,
    build_current_signals,
)
from research_workbench.validation.replay import (
    build_symbol_validation,
    build_validation_summary,
    replay_trigger_history,
)

st.set_page_config(page_title="Research Workbench", layout="wide")

SETTINGS = default_settings(ROOT)
EQUITY_CONFIG_PATH = ROOT / "equity_config.json"
HOLDINGS_PATH = ROOT / "research_inputs" / "holdings.json"
HOLDINGS_SNAPSHOT_DIR = ROOT / "outputs" / "holdings_snapshots"

WIRE_CSS = """
<style>
body, .stApp { font-family: Inter, system-ui, sans-serif; }
.rw-card {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 12px;
  background: white;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.rw-title { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
.rw-value { font-size: 1.7rem; font-weight: 700; color: #0f172a; margin-top: 4px; }
.rw-note { color: #475569; font-size: 0.9rem; }
</style>
"""
st.markdown(WIRE_CSS, unsafe_allow_html=True)


def _percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def _load_previous_signals() -> pd.DataFrame:
    """Load yesterday's signal snapshot for change detection."""
    if SETTINGS.signals_snapshot_path.exists():
        try:
            return pd.read_csv(SETTINGS.signals_snapshot_path)
        except Exception:
            pass
    return pd.DataFrame()


def _save_signals_snapshot(signals: pd.DataFrame) -> None:
    """Persist current signals so tomorrow's run can detect changes."""
    if not signals.empty:
        try:
            signals.to_csv(SETTINGS.signals_snapshot_path, index=False)
        except Exception:
            pass


def _load_context() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    watchlist = load_watchlist(SETTINGS.watchlist_path, EQUITY_CONFIG_PATH, market="CN")
    summary = load_summary(SETTINGS)
    history = load_history(SETTINGS)
    events = load_manual_events(SETTINGS)
    radar = load_radar(SETTINGS)
    symbols = watchlist_symbols(watchlist)
    # Always include holdings — the user must be able to see their positions
    # even if they aren't (yet) on the watchlist.
    holding_symbols = [
        _resolve_a_share_symbol(p.code) for p in load_holdings(HOLDINGS_PATH)
    ]
    effective_symbols = list({*symbols, *holding_symbols})
    if effective_symbols:
        summary = summary[summary["symbol"].isin(effective_symbols)].copy()
        history = history[history["symbol"].isin(effective_symbols)].copy()
    signals = build_current_signals(summary, history, watchlist, events, DEFAULT_RADAR_CONFIG)
    if signals.empty and not radar.empty:
        signals = radar.copy()
    previous_signals = _load_previous_signals()
    changes = detect_daily_changes(signals, previous_signals)
    _save_signals_snapshot(signals)
    replay = replay_trigger_history(history, symbols=symbols, cfg=DEFAULT_RADAR_CONFIG)
    return watchlist, summary, history, events, signals, replay, changes


def _resolve_a_share_symbol(bare_code: str) -> str:
    """Map a bare A-share code (e.g. '002602') to the suffixed symbol used in
    the local history dataset (e.g. '002602.SZ').  Already-suffixed codes pass
    through unchanged.
    """
    code = str(bare_code).strip().upper()
    if "." in code:
        return code
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{code}.SZ"
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{code}.SH"
    return code


_HOLDING_PRIORITY_EMOJI = {1: "🔴", 2: "🟠", 3: "🟢", 4: "⚪"}


def _fmt_pct_signed(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:+.2f}%"


def _fmt_price_or_dash(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.2f}"


def render_holdings_section(history: pd.DataFrame) -> None:
    """Daily 加仓/减仓/清仓 directives for every held position, with a one-line
    diff against the previous snapshot so the user can see what changed —
    the strongest anti-impulse signal is "nothing changed, still 持有不动".
    """
    st.subheader("💼 持仓动作")
    positions = load_holdings(HOLDINGS_PATH)
    if not positions:
        st.info(
            f"在 `{HOLDINGS_PATH.relative_to(ROOT)}` 维护你的持仓后会显示每日动作建议。"
            " 格式参考 `research_inputs/holdings.example.json`。"
        )
        return

    previous_snap = load_previous_snapshot(HOLDINGS_SNAPSHOT_DIR)

    snapshots: list = []
    missing: list[str] = []
    for p in positions:
        symbol = _resolve_a_share_symbol(p.code)
        stock_hist = history[history["symbol"] == symbol].copy()
        if stock_hist.empty:
            missing.append(f"{p.name} ({p.code})")
            continue
        # local snapshot uses 'support_level'; holdings_tracker expects
        # 'support_level_primary' to compute pct_to_support and stop_for_plan.
        if (
            "support_level" in stock_hist.columns
            and "support_level_primary" not in stock_hist.columns
        ):
            stock_hist["support_level_primary"] = stock_hist["support_level"]
        enriched = add_radar_features(stock_hist)
        try:
            snapshots.append(analyze_holding(enriched, p))
        except Exception as exc:  # noqa: BLE001 — never block the UI
            st.warning(f"{p.name} ({p.code}) 分析失败：{exc}")

    if missing:
        st.warning(
            "以下持仓在本地历史数据中找不到，请把它们加进 watchlist 后运行 "
            "`python3 scripts/refresh.py`：\n- " + "\n- ".join(missing)
        )

    if not snapshots:
        return

    rows = []
    for snap in sorted(snapshots, key=lambda s: (s.priority, s.code)):
        rows.append({
            "优先级": _HOLDING_PRIORITY_EMOJI.get(snap.priority, "⚪"),
            "股票": f"{snap.name} ({snap.code})",
            "今日动作": snap.action,
            "vs 昨日": diff_against_previous(snap, previous_snap),
            "现价": _fmt_price_or_dash(snap.last_close),
            "浮盈%": _fmt_pct_signed(snap.floating_pnl_pct),
            "止损 / TP1 / TP2": (
                f"{_fmt_price_or_dash(snap.stop_price)} / "
                f"{_fmt_price_or_dash(snap.take_profit_1)} / "
                f"{_fmt_price_or_dash(snap.take_profit_2)}"
            ),
            "理由": snap.note,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "动作基于成本价锚定的止损/TP 与生产 radar 触发规则。'vs 昨日' 显示与上一份快照的差异，"
        "用于避免无变化时的冲动操作。"
    )

    # Persist today's analysis so tomorrow's UI can render the diff.
    try:
        save_snapshot(snapshots, HOLDINGS_SNAPSHOT_DIR)
    except Exception:  # noqa: BLE001 — snapshot save is best-effort
        pass


def _render_card(column, title: str, value: str, note: str) -> None:
    column.markdown(
        f"<div class='rw-card'><div class='rw-title'>{title}</div><div class='rw-value'>{value}</div><div class='rw-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def render_home(signals: pd.DataFrame, summary: pd.DataFrame, events: pd.DataFrame, replay: pd.DataFrame, changes: pd.DataFrame, history: pd.DataFrame) -> None:
    cols = st.columns(4)
    triggered = int((signals.get("signal_state", pd.Series(dtype=str)) == "triggered").sum()) if not signals.empty else 0
    near_zone = int((signals.get("signal_state", pd.Series(dtype=str)) == "near_zone").sum()) if not signals.empty else 0
    avg_conviction = signals["conviction_score"].mean() if "conviction_score" in signals.columns and not signals.empty else None
    validation_summary = build_validation_summary(replay)
    total_samples = int(validation_summary["sample_count"].sum()) if not validation_summary.empty else 0
    _render_card(cols[0], "Watchlist", str(len(summary)), "当前纳入研究工作台的 A 股")
    _render_card(cols[1], "Triggered", str(triggered), "已触发可重点复查条件")
    _render_card(cols[2], "Near Zone", str(near_zone), "接近观察区间，等待确认")
    _render_card(cols[3], "Validation Samples", str(total_samples), "历史同类触发样本数")

    # Holdings advisor sits at the top of the home page — it's the most personal
    # surface and the strongest anti-impulse anchor ("nothing changed, hold").
    render_holdings_section(history)

    st.subheader("今日重点变化")
    if changes.empty:
        st.info("今日暂无显著状态变化。首次运行时，所有 triggered/near_zone 标的会被标记为新事件。")
    else:
        st.dataframe(
            changes[["symbol", "name", "change_type", "detail"]].rename(
                columns={"symbol": "代码", "name": "名称", "change_type": "变化类型", "detail": "详情"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("接近买点")
    if signals.empty:
        st.info("当前没有可展示的信号。先补齐 A 股 watchlist 或重新生成 summary/history 数据。")
    else:
        display_cols = [
            "symbol",
            "name",
            "signal_state",
            "trigger_type",
            "trigger_price",
            "stop_price",
            "risk_unit",
            "take_profit_1",
            "take_profit_2",
            "trailing_stop",
            "target_zone_low",
            "target_zone_high",
            "exit_plan",
            "valuation_score",
            "quality_score",
            "event_risk_score",
            "conviction_score",
            "reasons",
        ]
        present_cols = [col for col in display_cols if col in signals.columns]
        st.caption("止盈说明：TP1 按 1R 分批卖出 30%-50%，TP2 按 2R 再卖 20%-30%，剩余仓位用 MA20 与 ATR 移动止盈跟踪。")
        st.dataframe(
            signals[present_cols].sort_values("conviction_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("后续动作建议")
    if signals.empty:
        st.info("暂无信号，无法生成动作建议。")
    else:
        actions = advise_actions(signals)
        if actions.empty:
            st.info("暂无可展示的动作建议。")
        else:
            _PRIORITY_LABELS = {1: "🔴 紧急", 2: "🟡 重要", 3: "🔵 一般", 4: "⚪ 低"}
            actions_display = actions.copy()
            actions_display["优先级"] = actions_display["action_priority"].map(_PRIORITY_LABELS).fillna("⚪ 低")
            action_cols = ["优先级", "symbol", "name", "signal_state", "action", "action_detail"]
            present = [c for c in action_cols if c in actions_display.columns]
            st.dataframe(
                actions_display[present].rename(columns={
                    "symbol": "代码", "name": "名称", "signal_state": "状态",
                    "action": "建议动作", "action_detail": "详情",
                }),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("基本面过滤概览")
    if summary.empty:
        st.info("暂无 summary 数据。")
    else:
        overview_cols = [
            "symbol",
            "name",
            "value_score",
            "value_score_tier",
            "entry_recommendation",
            "score_hist_valuation",
            "score_abs_valuation",
            "score_peer_valuation",
            "score_growth_quality",
            "score_balance_sheet",
            "score_shareholder_return",
        ]
        present_cols = [col for col in overview_cols if col in summary.columns]
        st.dataframe(summary[present_cols], use_container_width=True, hide_index=True)

    st.subheader("公司 / 财报事件")
    if events.empty:
        st.info(f"暂无事件数据。可在 `{SETTINGS.manual_events_path}` 中维护手动事件。")
    else:
        st.dataframe(events.sort_values("event_date", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("历史验证摘要")
    validation_summary = build_validation_summary(replay)
    if validation_summary.empty:
        st.info("暂无历史触发样本，无法生成验证摘要。")
    else:
        st.dataframe(validation_summary, use_container_width=True, hide_index=True)


def render_stock_detail(symbol: str, history: pd.DataFrame, signals: pd.DataFrame, events: pd.DataFrame, replay: pd.DataFrame) -> None:
    stock_history = history[history["symbol"] == symbol].copy()
    stock_signal = signals[signals["symbol"] == symbol].head(1)
    stock_events = events[events["symbol"] == symbol].copy()
    symbol_validation = build_symbol_validation(replay, symbol)

    if stock_signal.empty:
        st.warning("当前标的暂无结构化信号。")
    else:
        row = stock_signal.iloc[0]
        a, b, c, d = st.columns(4)
        _render_card(a, "Signal State", str(row.get("signal_state", "N/A")), "当前状态")
        _render_card(b, "Trigger", str(row.get("trigger_type", "N/A")), "触发类型")
        _render_card(c, "Conviction", f"{float(row.get('conviction_score', 0.0)):.1f}", "综合信心分")
        _render_card(d, "Event Risk", f"{float(row.get('event_risk_score', 0.0)):.1f}", "近期事件风险")

    # ---- Action Advice ----
    st.subheader("📋 动作建议")
    if not stock_signal.empty:
        actions = advise_actions(stock_signal)
        if not actions.empty:
            act = actions.iloc[0]
            _PRIORITY_COLORS = {1: "#dc2626", 2: "#d97706", 3: "#2563eb", 4: "#6b7280"}
            _PRIORITY_LABELS = {1: "紧急", 2: "重要", 3: "一般", 4: "低"}
            pri = int(act.get("action_priority", 4))
            pri_color = _PRIORITY_COLORS.get(pri, "#6b7280")
            pri_label = _PRIORITY_LABELS.get(pri, "低")

            st.markdown(
                f"""
                <div class='rw-card'>
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-size:1.1rem;font-weight:700;color:#0f172a;">{act['action']}</div>
                        <span style="display:inline-block;padding:3px 10px;border-radius:6px;
                               background:{pri_color};color:white;font-size:0.8rem;">{pri_label}</span>
                    </div>
                    <div style="color:#475569;font-size:0.9rem;margin-top:8px;white-space:pre-line;">{act['action_detail']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Entry plan (only for triggered stocks)
            entry = act.get("suggested_entry")
            if entry is not None and pd.notna(entry):
                st.markdown("**入场计划参考**")
                e1, e2, e3, e4 = st.columns(4)
                _render_card(e1, "入场参考", f"{float(entry):.2f}", "当前收盘价")
                stop = act.get("suggested_stop")
                if stop is not None and pd.notna(stop):
                    risk_pct = (float(entry) - float(stop)) / float(entry) * 100
                    _render_card(e2, "止损位", f"{float(stop):.2f}", f"风险 {risk_pct:.1f}%")
                tp1 = act.get("suggested_tp1")
                if tp1 is not None and pd.notna(tp1):
                    gain_pct = (float(tp1) - float(entry)) / float(entry) * 100
                    _render_card(e3, "止盈1 (1R)", f"{float(tp1):.2f}", f"收益 {gain_pct:.1f}%")
                tp2 = act.get("suggested_tp2")
                if tp2 is not None and pd.notna(tp2):
                    gain_pct2 = (float(tp2) - float(entry)) / float(entry) * 100
                    _render_card(e4, "止盈2 (2R)", f"{float(tp2):.2f}", f"收益 {gain_pct2:.1f}%")

                sizing = act.get("position_sizing_note", "")
                if sizing:
                    st.caption(sizing)
    else:
        st.info("暂无信号，无法生成动作建议。")

    st.subheader("价格证据")
    if stock_history.empty:
        st.info("暂无该标的的历史数据。")
    else:
        fig_df = stock_history.sort_values("date").copy()
        plot_df = fig_df[["date", "close"]].rename(columns={"close": "Close"})
        if "ma50" in fig_df.columns:
            plot_df["MA50"] = fig_df["ma50"]
        if "ma200" in fig_df.columns:
            plot_df["MA200"] = fig_df["ma200"]
        fig = px.line(plot_df, x="date", y=[col for col in plot_df.columns if col != "date"])
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        evidence_cols = [
            "date",
            "close",
            "support_level",
            "support_level_secondary",
            "ma20",
            "ma50",
            "ma200",
            "highest_close_20d",
            "volume_spike_ratio",
            "atr14",
            "drawdown_60d",
        ]
        evidence = stock_history.sort_values("date", ascending=False).head(10)
        present_cols = [col for col in evidence_cols if col in evidence.columns]
        st.dataframe(evidence[present_cols], use_container_width=True, hide_index=True)

    st.subheader("基本面 / 信号证据")
    if stock_signal.empty:
        st.info("暂无可展示证据。")
    else:
        st.caption("止盈/止损不是自动下单：价格位用于复盘和执行计划，实际仓位应按单笔最大亏损控制。")
        st.json(
            {
                "reasons": stock_signal.iloc[0].get("reasons", ""),
                "invalidation_conditions": stock_signal.iloc[0].get("invalidation_conditions", ""),
                "entry_recommendation": stock_signal.iloc[0].get("entry_recommendation", ""),
                "stop_price": stock_signal.iloc[0].get("stop_price"),
                "risk_unit": stock_signal.iloc[0].get("risk_unit"),
                "take_profit_1": stock_signal.iloc[0].get("take_profit_1"),
                "take_profit_2": stock_signal.iloc[0].get("take_profit_2"),
                "trailing_stop": stock_signal.iloc[0].get("trailing_stop"),
                "exit_plan": stock_signal.iloc[0].get("exit_plan", ""),
                "exit_plan_notes": stock_signal.iloc[0].get("exit_plan_notes", ""),
                "target_zone_low": stock_signal.iloc[0].get("target_zone_low"),
                "target_zone_high": stock_signal.iloc[0].get("target_zone_high"),
            }
        )

    st.subheader("公司 / 财报事件")
    if stock_events.empty:
        st.info("暂无该标的的手动事件。")
    else:
        st.dataframe(stock_events.sort_values("event_date", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("历史同类信号")
    if symbol_validation.empty:
        st.info("暂无该标的的历史触发样本。")
    else:
        st.dataframe(symbol_validation, use_container_width=True, hide_index=True)

    with st.expander("解释层（可选）"):
        if stock_signal.empty:
            st.write("暂无解释。")
        else:
            row = stock_signal.iloc[0]
            st.write(
                f"{symbol} 当前状态为 `{row.get('signal_state', 'watch')}`。"
                f" 触发类型：`{row.get('trigger_type', 'none')}`。"
                f" 估值分 `{row.get('valuation_score', 0):.1f}`，质量分 `{row.get('quality_score', 0):.1f}`，"
                f" 事件风险 `{row.get('event_risk_score', 0):.1f}`。"
                " 这段解释只作为阅读辅助，不参与主判断。"
            )


def render_validation(replay: pd.DataFrame) -> None:
    st.subheader("历史验证")
    summary = build_validation_summary(replay)
    if summary.empty:
        st.info("暂无验证结果。")
        return
    st.dataframe(summary, use_container_width=True, hide_index=True)
    fig = px.bar(summary, x="trigger_type", y="avg_return_20d", color="sample_count", text="sample_count")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_settings(watchlist: pd.DataFrame) -> None:
    st.subheader("输入与配置")
    st.write("这部分不是最终配置中心，只是第一版研究工作台的输入边界。")
    st.code(
        "\n".join(
            [
                f"watchlist_path: {SETTINGS.watchlist_path}",
                f"manual_events_path: {SETTINGS.manual_events_path}",
                f"summary_path: {SETTINGS.summary_path}",
                f"history_path: {SETTINGS.history_path}",
                f"radar_path: {SETTINGS.radar_path}",
            ]
        )
    )
    st.write("当前 watchlist：")
    st.dataframe(watchlist, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("个人研究工作台")
    st.caption("第一版目标：证据透明 + 历史验证。UI 仅为可替换骨架。")

    watchlist, summary, history, events, signals, replay, changes = _load_context()

    tabs = st.tabs(["首页", "单票", "验证", "设置"])
    with tabs[0]:
        render_home(signals, summary, events, replay, changes, history)
    with tabs[1]:
        options = watchlist_symbols(watchlist) or (sorted(signals["symbol"].unique().tolist()) if not signals.empty else [])
        if not options:
            st.info("暂无可选标的。")
        else:
            selected = st.selectbox("选择标的", options=options)
            render_stock_detail(selected, history, signals, events, replay)
    with tabs[2]:
        render_validation(replay)
    with tabs[3]:
        render_settings(watchlist)


if __name__ == "__main__":
    main()
