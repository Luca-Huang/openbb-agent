"""Per-holding daily snapshot for the opening-window radar email.

This sits on top of :mod:`signal_engine.backtest` and produces a HTML "card"
per held position.  By design the card NEVER shows absolute amounts — only
percentages — so the email can be safely re-sent or forwarded without
exposing position size.

Usage::

    positions = load_holdings(Path("research_inputs/holdings.json"))
    snapshots = [analyze_holding(history_df, p) for p in positions]
    html = render_holding_cards_html(snapshots)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .backtest import compute_position_pnl
from .radar import (
    DEFAULT_RADAR_CONFIG,
    compute_exit_plan,
    detect_trigger_type,
)


@dataclass
class Position:
    code: str
    name: str
    trades: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HoldingSnapshot:
    """All numbers shown to the user are percentages or relative; absolutes
    are kept only for internal use (e.g. exit-plan math).
    """
    code: str
    name: str
    last_date: str
    last_close: float

    # Position state — internal, NOT rendered to the email
    holdings: float
    broker_cost: float
    avg_cost: float

    # Public percentages
    floating_pnl_pct: float       # vs broker (摊薄) cost
    pct_to_ma20: float            # (close - ma20) / ma20
    pct_to_ma50: float
    pct_to_ma200: float
    pct_to_support: float

    # Production-rule verdict
    trigger_type: str             # "breakout" | "pullback" | ""
    verdict: str                  # ACCEPT / REJECT
    rsi14: float
    vol_spike: float

    # Suggested exit plan (price levels, not amounts)
    stop_price: float
    take_profit_1: float
    take_profit_2: float
    trailing_stop: float

    action: str                   # high-level recommendation
    note: str                     # one-line explanation


def load_holdings(path: Path) -> list[Position]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[Position] = []
    for raw in payload.get("positions", []):
        out.append(Position(
            code=str(raw["code"]),
            name=str(raw.get("name", raw["code"])),
            trades=list(raw.get("trades", [])),
        ))
    return out


def _safe_pct(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None:
        return float("nan")
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return float("nan")
    return (float(numerator) - float(denominator)) / float(denominator)


def _classify_action(
    pnl: Any,
    holdings: float,
    trigger_type: str,
    pct_to_ma50: float,
    pct_to_ma200: float,
) -> tuple[str, str]:
    """Decide a human-readable recommendation.

    Rules — kept simple on purpose, read top-to-bottom:
      1. No holdings + ACCEPT trigger → '可考虑建仓'
      2. No holdings + no trigger      → '空仓观察'
      3. Holding + ACCEPT trigger      → '可考虑加仓'
      4. Holding + close > MA50 + close > MA200 → '持有不动'
      5. Holding + close < MA50 < MA200          → '风险升高，考虑减仓'
      6. Otherwise (holding, mixed)              → '观察 MA50 突破/破位'
    """
    has_pos = holdings > 0
    accept = trigger_type in {"breakout", "pullback"}
    above_ma50 = pct_to_ma50 > 0
    above_ma200 = pct_to_ma200 > 0

    if not has_pos:
        if accept:
            return "可考虑建仓", f"今日触发 {trigger_type}"
        return "空仓观察", "等待 ACCEPT 信号"
    if accept:
        return "可考虑加仓", f"今日触发 {trigger_type}"
    if above_ma50 and above_ma200:
        return "持有不动", "趋势结构未破"
    if not above_ma50 and not above_ma200:
        return "风险升高，考虑减仓", "已跌破 MA50 且仍在 MA200 之下"
    return "观察 MA50 突破/破位", "处在均线枢纽位"


def analyze_holding(
    history_df: pd.DataFrame,
    position: Position,
    cfg: dict[str, Any] | None = None,
) -> HoldingSnapshot:
    """Build a snapshot for one holding from its history dataframe.

    ``history_df`` must already carry the indicator schema produced by
    ``compute_full_indicators`` / ``add_radar_features`` etc.  The latest bar
    drives all "today" numbers.
    """
    cfg = cfg or DEFAULT_RADAR_CONFIG
    trigger_cfg = cfg.get("triggers", {})
    exit_cfg = cfg.get("exits", {})
    df = history_df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No history data for {position.code}")
    last = df.iloc[-1]
    last_close = float(last["close"])

    pnl = compute_position_pnl(position.trades, last_price=last_close)

    trigger = detect_trigger_type(last, trigger_cfg) or ""
    verdict = "ACCEPT" if trigger in {"breakout", "pullback"} else "REJECT"

    # Exit plan always anchored on the latest close — this answers
    # "if I trade at today's price, where do the levels sit?", which is the
    # relevant question whether the user is adding, building fresh, or just
    # wants a trailing-stop reference for the existing position.
    support_primary = pd.to_numeric(last.get("support_level_primary"), errors="coerce")
    ma50 = pd.to_numeric(last.get("ma50"), errors="coerce")
    ma200 = pd.to_numeric(last.get("ma200"), errors="coerce")
    entry_for_plan = last_close
    # Choose a stop strictly below entry: prefer 20-day support, fall back to
    # MA50 then MA200 then a fixed 8% buffer.
    candidate_stops = [
        v for v in (support_primary, ma50, ma200)
        if pd.notna(v) and float(v) < entry_for_plan
    ]
    stop_for_plan = (
        max(candidate_stops) if candidate_stops else entry_for_plan * 0.92
    )
    plan = compute_exit_plan(
        entry_price=entry_for_plan,
        stop_price=stop_for_plan,
        atr14=pd.to_numeric(last.get("atr14"), errors="coerce"),
        ma20=pd.to_numeric(last.get("ma20"), errors="coerce"),
        highest_close_20d=pd.to_numeric(last.get("highest_close_20d"), errors="coerce"),
        cfg=exit_cfg,
    )

    pct_to_ma20 = _safe_pct(last_close, last.get("ma20"))
    pct_to_ma50 = _safe_pct(last_close, last.get("ma50"))
    pct_to_ma200 = _safe_pct(last_close, last.get("ma200"))
    pct_to_support = _safe_pct(last_close, last.get("support_level_primary"))

    action, note = _classify_action(
        pnl=pnl, holdings=pnl.holdings, trigger_type=trigger,
        pct_to_ma50=pct_to_ma50, pct_to_ma200=pct_to_ma200,
    )

    return HoldingSnapshot(
        code=position.code,
        name=position.name,
        last_date=pd.to_datetime(last["date"]).strftime("%Y-%m-%d"),
        last_close=last_close,
        holdings=pnl.holdings,
        broker_cost=pnl.broker_cost,
        avg_cost=pnl.avg_cost,
        floating_pnl_pct=pnl.floating_pnl_pct if pnl.holdings > 0 else float("nan"),
        pct_to_ma20=pct_to_ma20,
        pct_to_ma50=pct_to_ma50,
        pct_to_ma200=pct_to_ma200,
        pct_to_support=pct_to_support,
        trigger_type=trigger,
        verdict=verdict,
        rsi14=float(last.get("rsi14")) if pd.notna(last.get("rsi14")) else float("nan"),
        vol_spike=float(last.get("volume_spike_ratio")) if pd.notna(last.get("volume_spike_ratio")) else float("nan"),
        # Structural stop = chosen stop_for_plan (support / MA50 / MA200 / 8% buffer).
        # Trailing stop = dynamic trail from compute_exit_plan (may be NaN when
        # MA20 / highest-close minus ATR sit above entry).  Showing both lets
        # the user see the strict stop alongside the active trail.
        stop_price=float(stop_for_plan),
        take_profit_1=float(plan.get("take_profit_1")) if pd.notna(plan.get("take_profit_1")) else float("nan"),
        take_profit_2=float(plan.get("take_profit_2")) if pd.notna(plan.get("take_profit_2")) else float("nan"),
        trailing_stop=float(plan.get("trailing_stop")) if pd.notna(plan.get("trailing_stop")) else float("nan"),
        action=action,
        note=note,
    )


# ---------- HTML rendering (percentages only) ----------

def _fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:+.2f}%"


def _fmt_price(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.2f}"


def _verdict_badge(verdict: str, trigger: str) -> str:
    color = {"ACCEPT": "#28a745", "REJECT": "#6c757d"}.get(verdict, "#6c757d")
    label = verdict if not trigger else f"{verdict} · {trigger}"
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:4px;"
        f"background:{color};color:white;font-size:12px;'>{label}</span>"
    )


def _action_color(action: str) -> str:
    if "建仓" in action or "加仓" in action:
        return "#28a745"
    if "减仓" in action or "风险" in action:
        return "#dc3545"
    if "持有" in action:
        return "#17a2b8"
    return "#6c757d"


def render_holding_card(snap: HoldingSnapshot) -> str:
    """Single holding card.  All quantitative values are percentages."""
    held = snap.holdings > 0
    pnl_label = f"浮盈 {_fmt_pct(snap.floating_pnl_pct)}" if held else "空仓"
    return f"""
    <div style="border:1px solid #e1e4e8;border-radius:6px;padding:12px 16px;
                margin:10px 0;font-family:-apple-system,Segoe UI,sans-serif;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-weight:600;font-size:15px;">
          {snap.name} <span style="color:#666;font-weight:400;">{snap.code}</span>
        </div>
        <div>{_verdict_badge(snap.verdict, snap.trigger_type)}</div>
      </div>
      <div style="margin-top:6px;color:#555;font-size:13px;">
        {snap.last_date} 收 {_fmt_price(snap.last_close)} · {pnl_label}
        · RSI {snap.rsi14:.0f} · 量比 {snap.vol_spike:.2f}
      </div>
      <div style="margin-top:8px;font-size:13px;color:#444;">
        距 MA20 {_fmt_pct(snap.pct_to_ma20)} ·
        距 MA50 {_fmt_pct(snap.pct_to_ma50)} ·
        距 MA200 {_fmt_pct(snap.pct_to_ma200)} ·
        距支撑 {_fmt_pct(snap.pct_to_support)}
      </div>
      <div style="margin-top:8px;font-size:13px;color:#444;">
        止损 {_fmt_price(snap.stop_price)} ·
        TP1 {_fmt_price(snap.take_profit_1)} ·
        TP2 {_fmt_price(snap.take_profit_2)} ·
        移动止损 {_fmt_price(snap.trailing_stop)}
      </div>
      <div style="margin-top:8px;padding:6px 10px;background:#f6f8fa;border-radius:4px;
                  font-size:13px;">
        <span style="color:{_action_color(snap.action)};font-weight:600;">{snap.action}</span>
        <span style="color:#666;"> — {snap.note}</span>
      </div>
    </div>
    """.strip()


def render_holding_cards_html(snapshots: Iterable[HoldingSnapshot]) -> str:
    cards = [render_holding_card(s) for s in snapshots]
    if not cards:
        return ""
    return f"""
    <h3 style="margin-bottom:4px;">持仓追踪</h3>
    <p style="color:#888;font-size:12px;margin-top:0;">
      仅显示百分比。verdict 来自生产 radar 触发规则；操作建议为参考，请结合自有判断。
    </p>
    {''.join(cards)}
    """.strip()


__all__ = [
    "Position",
    "HoldingSnapshot",
    "load_holdings",
    "analyze_holding",
    "render_holding_card",
    "render_holding_cards_html",
]
