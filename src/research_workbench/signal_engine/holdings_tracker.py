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

    # Suggested exit plan (price levels, not amounts).
    # When holdings > 0 these are anchored on the user's avg_cost so that
    # "TP1 reached" / "stop violated" are actually meaningful for the current
    # position.  When holdings == 0 they're anchored on today's close (the
    # hypothetical "if I entered today" plan).
    stop_price: float
    take_profit_1: float
    take_profit_2: float
    trailing_stop: float

    action: str                   # high-level recommendation
    note: str                     # one-line explanation
    priority: int = 4             # 1=urgent (清仓) … 4=low (持有/观察)


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
    *,
    has_holdings: bool,
    last_close: float,
    trigger_type: str,
    floating_pnl_pct: float,
    pct_to_ma50: float,
    pct_to_ma200: float,
    stop_price: float,
    take_profit_1: float,
    take_profit_2: float,
    trailing_stop: float,
) -> tuple[str, str, int]:
    """Decide a clear directive plus a priority bucket.

    Returns ``(action, note, priority)`` where priority is:
      1 = urgent (清仓 / 严重风险)
      2 = important (减仓 / 落袋)
      3 = opportunity (加仓 / 建仓)
      4 = no-op (持有 / 观察)

    The goal is **明确指令** — avoid hedging wording like '考虑' / '可能',
    so the user is either told to do something or told to do nothing.

    Priority-ordered evaluation (first match wins):
      A. No holdings:
         - ACCEPT trigger          → 建仓
         - else                    → 空仓观察
      B. Holdings:
         1. close ≤ stop_price     → 清仓 (stop violated)
         2. close < MA50 < MA200 AND pnl < -10%
                                   → 清仓 (trend broken + heavy loss)
         3. close ≥ TP2            → 减仓 50% (TP2 落袋)
         4. close ≥ TP1            → 减仓 25% (TP1 落袋)
         5. close < trailing_stop  → 减仓 25% (跌破移动止损)
         6. close < MA50 (still above MA200)
                                   → 减仓 25% (结构走弱)
         7. ACCEPT + pullback      → 加仓
         8. close > MA50 AND close > MA200
                                   → 持有不动 (趋势完好)
         9. fallback               → 持有观察
    """
    accept = trigger_type in {"breakout", "pullback"}
    above_ma50 = pct_to_ma50 > 0
    above_ma200 = pct_to_ma200 > 0

    if not has_holdings:
        if accept:
            return ("建仓", f"今日触发 {trigger_type} 信号", 3)
        return ("空仓观察", "等待 ACCEPT 信号", 4)

    # 1. 止损线已破 — 清仓
    if pd.notna(stop_price) and last_close <= float(stop_price):
        return ("清仓", f"已跌破止损线 {stop_price:.2f}", 1)

    # 2. 趋势完全破坏 + 重亏 — 清仓
    if (
        not above_ma50
        and not above_ma200
        and pd.notna(floating_pnl_pct)
        and floating_pnl_pct < -0.10
    ):
        return (
            "清仓",
            f"MA50/MA200 同时跌破且浮亏 {floating_pnl_pct*100:.1f}%，趋势已破",
            1,
        )

    # 3. 触及 TP2 — 减仓 50% 落袋
    if pd.notna(take_profit_2) and last_close >= float(take_profit_2):
        return ("减仓 50%", f"已达 TP2 {take_profit_2:.2f}，落袋一半", 2)

    # 4. 触及 TP1 — 减仓 25%
    if pd.notna(take_profit_1) and last_close >= float(take_profit_1):
        return ("减仓 25%", f"已达 TP1 {take_profit_1:.2f}，落袋 25%", 2)

    # 5. 跌破移动止损 — 减仓 25%（趋势中保护盈利）
    if (
        pd.notna(trailing_stop)
        and pd.notna(stop_price)
        and float(trailing_stop) > float(stop_price)
        and last_close < float(trailing_stop)
    ):
        return ("减仓 25%", f"跌破移动止损 {trailing_stop:.2f}", 2)

    # 6. 跌破 MA50（还在 MA200 上）— 减仓 25%
    if not above_ma50 and above_ma200:
        return ("减仓 25%", "跌破 MA50，结构走弱", 2)

    # 7. ACCEPT pullback — 加仓机会
    if accept and trigger_type == "pullback":
        return ("加仓", "回踩支撑触发 pullback 信号", 3)

    # 8. 趋势完好 — 持有
    if above_ma50 and above_ma200:
        return ("持有不动", "趋势完好，站稳 MA50/MA200", 4)

    # 9. 兜底
    return ("持有观察", "结构中性，等待方向", 4)


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

    # Exit-plan anchor:
    #   • holdings > 0 → anchor on avg_cost so TP / stop are meaningful for
    #     the user's actual position ("did MY position hit TP1?").
    #   • holdings == 0 → anchor on today's close (the hypothetical "if I
    #     entered today, where would the levels sit?").
    support_primary = pd.to_numeric(last.get("support_level_primary"), errors="coerce")
    ma50 = pd.to_numeric(last.get("ma50"), errors="coerce")
    ma200 = pd.to_numeric(last.get("ma200"), errors="coerce")

    if pnl.holdings > 0 and pnl.avg_cost > 0:
        entry_for_plan = float(pnl.avg_cost)
    else:
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

    tp1_val = plan.get("take_profit_1")
    tp2_val = plan.get("take_profit_2")
    trail_val = plan.get("trailing_stop")
    action, note, priority = _classify_action(
        has_holdings=pnl.holdings > 0,
        last_close=last_close,
        trigger_type=trigger,
        floating_pnl_pct=pnl.floating_pnl_pct if pnl.holdings > 0 else float("nan"),
        pct_to_ma50=pct_to_ma50,
        pct_to_ma200=pct_to_ma200,
        stop_price=float(stop_for_plan),
        take_profit_1=float(tp1_val) if pd.notna(tp1_val) else float("nan"),
        take_profit_2=float(tp2_val) if pd.notna(tp2_val) else float("nan"),
        trailing_stop=float(trail_val) if pd.notna(trail_val) else float("nan"),
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
        priority=priority,
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
    if "清仓" in action:
        return "#dc3545"          # red — urgent exit
    if "减仓" in action:
        return "#fd7e14"          # orange — partial trim
    if "建仓" in action or "加仓" in action:
        return "#28a745"          # green — entry / add
    if "持有不动" in action:
        return "#17a2b8"          # teal — keep
    return "#6c757d"              # gray — observe


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


# ---------- Daily snapshot persistence + yesterday diff ----------

# These power the "明确指令 + 昨日分析" UX: every refresh writes a snapshot,
# and the next day's UI / email can show what changed.  The diff lives next
# to the action so the user can see *why* today is different — the strongest
# anti-impulse signal is "nothing changed, still 持有不动".


def _snapshot_record(snap: HoldingSnapshot) -> dict[str, Any]:
    """Subset of HoldingSnapshot needed for the diff.  Kept small and
    JSON-serialisable.  Absolute amounts (holdings, broker_cost) are NOT
    persisted — the diff only needs decision-relevant fields.
    """
    def _maybe_float(x: float) -> float | None:
        return None if x is None or pd.isna(x) else float(x)

    return {
        "code": snap.code,
        "name": snap.name,
        "last_date": snap.last_date,
        "last_close": _maybe_float(snap.last_close),
        "floating_pnl_pct": _maybe_float(snap.floating_pnl_pct),
        "trigger_type": snap.trigger_type,
        "verdict": snap.verdict,
        "pct_to_ma50": _maybe_float(snap.pct_to_ma50),
        "pct_to_ma200": _maybe_float(snap.pct_to_ma200),
        "stop_price": _maybe_float(snap.stop_price),
        "action": snap.action,
        "priority": snap.priority,
        "note": snap.note,
    }


def save_snapshot(
    snapshots: Iterable[HoldingSnapshot],
    outdir: Path,
    as_of: str | None = None,
) -> Path:
    """Write today's snapshot to ``outdir/YYYY-MM-DD.json``.

    ``as_of`` defaults to today (local date).  Overwrites if the same date
    already has a snapshot (idempotent on intra-day re-runs).
    """
    snaps = list(snapshots)
    if as_of is None:
        as_of = pd.Timestamp.today().strftime("%Y-%m-%d")
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": as_of,
        "positions": [_snapshot_record(s) for s in snaps],
    }
    out = outdir / f"{as_of}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def load_previous_snapshot(outdir: Path, before: str | None = None) -> dict[str, Any] | None:
    """Return the most recent snapshot strictly older than ``before`` (today).

    Returns None if no prior snapshot exists.  Filenames are sorted lexically
    which works because the format is YYYY-MM-DD.
    """
    if not outdir.exists():
        return None
    if before is None:
        before = pd.Timestamp.today().strftime("%Y-%m-%d")
    candidates = sorted(
        p for p in outdir.glob("*.json")
        if p.stem < before
    )
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def diff_against_previous(
    today: HoldingSnapshot,
    previous: dict[str, Any] | None,
) -> str:
    """One-line human-readable diff vs the previous snapshot.

    Output examples:
      - 无昨日数据                       (no prior snapshot)
      - 延续 (持有不动)                   (action unchanged)
      - 昨日 持有不动 → 今日 减仓 25%      (action changed)
      - 昨日 持有不动 → 今日 清仓 · 已跌破止损线 17.50  (with note)
    """
    if previous is None:
        return "无昨日数据"

    yesterday_positions = {p["code"]: p for p in previous.get("positions", [])}
    prior = yesterday_positions.get(today.code)
    if prior is None:
        return "新持仓 (昨日无记录)"

    yesterday_action = str(prior.get("action", ""))
    if yesterday_action == today.action:
        return f"延续 ({today.action})"

    as_of = previous.get("as_of", "昨日")
    return f"{as_of} {yesterday_action} → 今日 {today.action}"


__all__ = [
    "Position",
    "HoldingSnapshot",
    "load_holdings",
    "analyze_holding",
    "render_holding_card",
    "render_holding_cards_html",
    "save_snapshot",
    "load_previous_snapshot",
    "diff_against_previous",
]
