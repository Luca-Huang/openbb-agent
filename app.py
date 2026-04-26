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
from research_workbench.research_store.files import (
    load_history,
    load_manual_events,
    load_radar,
    load_summary,
)
from research_workbench.signal_engine.changes import detect_daily_changes
from research_workbench.signal_engine.radar import DEFAULT_RADAR_CONFIG, build_current_signals
from research_workbench.validation.replay import (
    build_symbol_validation,
    build_validation_summary,
    replay_trigger_history,
)

st.set_page_config(page_title="A-Share Research Workbench", layout="wide")

SETTINGS = default_settings(ROOT)
EQUITY_CONFIG_PATH = ROOT / "equity_config.json"

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
    if symbols:
        summary = summary[summary["symbol"].isin(symbols)].copy()
        history = history[history["symbol"].isin(symbols)].copy()
    signals = build_current_signals(summary, history, watchlist, events, DEFAULT_RADAR_CONFIG)
    if signals.empty and not radar.empty:
        signals = radar.copy()
    previous_signals = _load_previous_signals()
    changes = detect_daily_changes(signals, previous_signals)
    _save_signals_snapshot(signals)
    replay = replay_trigger_history(history, symbols=symbols, cfg=DEFAULT_RADAR_CONFIG)
    return watchlist, summary, history, events, signals, replay, changes


def _render_card(column, title: str, value: str, note: str) -> None:
    column.markdown(
        f"<div class='rw-card'><div class='rw-title'>{title}</div><div class='rw-value'>{value}</div><div class='rw-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def render_home(signals: pd.DataFrame, summary: pd.DataFrame, events: pd.DataFrame, replay: pd.DataFrame, changes: pd.DataFrame) -> None:
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
            "target_zone_low",
            "target_zone_high",
            "valuation_score",
            "quality_score",
            "event_risk_score",
            "conviction_score",
            "reasons",
        ]
        present_cols = [col for col in display_cols if col in signals.columns]
        st.dataframe(
            signals[present_cols].sort_values("conviction_score", ascending=False),
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
            "ma50",
            "ma200",
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
        st.json(
            {
                "reasons": stock_signal.iloc[0].get("reasons", ""),
                "invalidation_conditions": stock_signal.iloc[0].get("invalidation_conditions", ""),
                "entry_recommendation": stock_signal.iloc[0].get("entry_recommendation", ""),
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
    st.title("A 股个人研究工作台")
    st.caption("第一版目标：证据透明 + 历史验证。UI 仅为可替换骨架。")

    watchlist, summary, history, events, signals, replay, changes = _load_context()

    tabs = st.tabs(["首页", "单票", "验证", "设置"])
    with tabs[0]:
        render_home(signals, summary, events, replay, changes)
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

