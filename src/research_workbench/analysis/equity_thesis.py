"""Equity thesis — structured 6-step analysis report per symbol.

Mirrors the framework distilled from a sample analyst note (catalyst-anchored,
peer-bucketed, quality-differentiated, SOTP-valued, sentiment-aware, action-
prescribed). Outputs a Markdown report; not a numeric score. Each step has
both data-driven auto-fill and manual override read from the watchlist's
``thesis`` block.

Usage::

    thesis = build_thesis(watchlist_item, history, enrichment, fund_data)
    print(render_thesis_markdown(thesis))

Manual overrides live in ``watchlist_cn.json`` under a per-item ``thesis``
object. None of the keys are required — anything missing falls back to data
or stays blank in the report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

log = logging.getLogger("analysis.equity_thesis")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QualityFactor:
    """One concrete differentiating factor with evidence (Step 3)."""
    description: str
    evidence: str = ""
    direction: int = 1  # +1 positive, -1 negative, 0 neutral


@dataclass
class BusinessSegment:
    """One revenue/profit line, optionally with a per-segment PE multiple (Step 4)."""
    name: str
    revenue: float | None = None       # CNY (元)
    revenue_ratio: float | None = None  # 0-1
    profit: float | None = None         # CNY (元)
    profit_ratio: float | None = None   # 0-1
    gross_margin: float | None = None   # 0-1
    pe_multiple: float | None = None    # user-supplied; None = exclude from SOTP
    valuation: float | None = None      # = profit * pe_multiple, computed
    source: str = "auto"                # "auto" / "manual_override"
    note: str = ""


@dataclass
class EquityThesis:
    """Structured 6-step investment thesis for one symbol."""
    symbol: str
    name: str = ""
    as_of: date | None = None

    # ① Catalyst
    catalyst_type: str | None = None
    catalyst_date: date | None = None
    catalyst_summary: str = ""

    # ② Peer Anchoring
    peer_industry: str = ""
    peer_pe_median: float | None = None
    peer_bucket: str = ""
    peer_success: list[str] = field(default_factory=list)
    peer_failure: list[str] = field(default_factory=list)

    # ③ Quality Differentiation
    quality_factors: list[QualityFactor] = field(default_factory=list)

    # ④ SOTP
    segments: list[BusinessSegment] = field(default_factory=list)
    sotp_valuation: float | None = None        # CNY total
    current_market_cap: float | None = None    # CNY
    upside_pct: float | None = None            # (sotp - mkt) / mkt

    # ⑤ Sentiment
    north_30d_net_buy_cny: float | None = None
    north_30d_pct_of_value: float | None = None
    capital_flow_net_main: float | None = None   # 万元 today
    capital_flow_net_total: float | None = None  # 万元 today
    sentiment_notes: str = ""

    # ⑥ Action
    current_price: float | None = None
    action: str = "watch"
    thesis_notes: str = ""

    # Data quality
    auto_filled: list[str] = field(default_factory=list)   # which fields came from data
    missing: list[str] = field(default_factory=list)       # missing inputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:  # noqa: BLE001
        return None


def _yuan_to_yi(value: float | None) -> str:
    """Format CNY (元) as `X.XX亿` for human-readable Markdown."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"{value / 1e8:.2f}亿"


def _pct(value: float | None, *, multiplier: float = 100.0, suffix: str = "%") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"{value * multiplier:.1f}{suffix}"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _latest_segments(business_segments: pd.DataFrame | None) -> pd.DataFrame:
    """Pick the most recent fiscal period and prefer 按产品分类, else 合计."""
    if business_segments is None or business_segments.empty:
        return pd.DataFrame()
    df = business_segments.dropna(subset=["fiscal_period"])
    if df.empty:
        return df
    latest = df["fiscal_period"].max()
    df = df[df["fiscal_period"] == latest]
    by_product = df[df["classification"] == "按产品分类"]
    if not by_product.empty:
        return by_product
    summary = df[df["classification"] == "合计"]
    if not summary.empty:
        return summary
    return df


def _auto_fill_segments(
    business_segments: pd.DataFrame | None,
    segment_pe_multiples: dict[str, float] | None,
    manual_segments: list[dict] | None,
) -> tuple[list[BusinessSegment], float | None]:
    """Build the segment list from data, with optional per-segment PE multipliers.

    If ``manual_segments`` is supplied, returns those instead — used for
    forward-looking thesis valuations (e.g. a new product whose future profit
    isn't in the historical breakdown yet).
    """
    if manual_segments:
        segs: list[BusinessSegment] = []
        for raw in manual_segments:
            pe = raw.get("pe_multiple")
            profit = raw.get("profit")
            val = float(pe) * float(profit) if pe and profit else None
            segs.append(BusinessSegment(
                name=str(raw.get("name", "未命名")),
                revenue=raw.get("revenue"),
                profit=profit,
                pe_multiple=pe,
                valuation=val,
                source="manual_override",
                note=str(raw.get("note", "")),
            ))
        total = sum((s.valuation or 0.0) for s in segs)
        return segs, (total if total else None)

    latest = _latest_segments(business_segments)
    if latest.empty:
        return [], None

    pe_map = segment_pe_multiples or {}
    segs = []
    total = 0.0
    has_any_pe = False
    for _, row in latest.iterrows():
        name = str(row.get("segment", "?"))
        profit = row.get("profit")
        pe = pe_map.get(name)
        val = float(pe) * float(profit) if (pe and pd.notna(profit)) else None
        if val is not None:
            total += val
            has_any_pe = True
        segs.append(BusinessSegment(
            name=name,
            revenue=row.get("revenue"),
            revenue_ratio=row.get("revenue_ratio"),
            profit=profit,
            profit_ratio=row.get("profit_ratio"),
            gross_margin=row.get("gross_margin"),
            pe_multiple=pe,
            valuation=val,
            source="auto",
        ))
    return segs, (total if has_any_pe else None)


def _auto_fill_sentiment(
    thesis: EquityThesis,
    northbound: pd.DataFrame | None,
    capital_flow: pd.DataFrame | None,
) -> None:
    """Fill Step 5 sentiment fields from northbound + capital-flow frames."""
    if northbound is not None and not northbound.empty:
        recent = northbound.sort_values("date").tail(30)
        net_30d = pd.to_numeric(recent.get("north_net_buy_cny"), errors="coerce").sum()
        if pd.notna(net_30d):
            thesis.north_30d_net_buy_cny = float(net_30d)
            thesis.auto_filled.append("north_30d_net_buy_cny")
        latest = northbound.sort_values("date").iloc[-1]
        latest_value = latest.get("north_value")
        if pd.notna(latest_value) and latest_value:
            thesis.north_30d_pct_of_value = float(net_30d) / float(latest_value) if pd.notna(net_30d) else None
    else:
        thesis.missing.append("northbound_holdings")

    if capital_flow is not None and not capital_flow.empty:
        latest = capital_flow.sort_values("date").iloc[-1]
        thesis.capital_flow_net_main = float(latest.get("net_main")) if pd.notna(latest.get("net_main")) else None
        thesis.capital_flow_net_total = float(latest.get("net_total")) if pd.notna(latest.get("net_total")) else None
        if thesis.capital_flow_net_main is not None:
            thesis.auto_filled.append("capital_flow_net_main")
    else:
        thesis.missing.append("capital_flow")


def build_thesis(
    item: pd.Series,
    history: pd.DataFrame,
    fund_data: dict[str, Any],
    *,
    business_segments: pd.DataFrame | None = None,
    earnings_express: pd.DataFrame | None = None,
    northbound: pd.DataFrame | None = None,
    capital_flow: pd.DataFrame | None = None,
    industry_peer_pe_median: float | None = None,
    as_of: date | None = None,
) -> EquityThesis:
    """Build a structured thesis from a watchlist row + the per-symbol slices.

    Manual fields are read from ``item["thesis"]`` (a dict if loaded from JSON).
    """
    symbol = str(item["symbol"])
    name = str(item.get("name", ""))
    thesis_input: dict = item.get("thesis") or {}
    if not isinstance(thesis_input, dict):
        thesis_input = {}

    t = EquityThesis(symbol=symbol, name=name, as_of=as_of or date.today())

    # ① Catalyst — manual primary, earnings_express announce_date as hint.
    t.catalyst_type = thesis_input.get("catalyst_type")
    t.catalyst_date = _to_date(thesis_input.get("catalyst_date"))
    t.catalyst_summary = str(thesis_input.get("catalyst_summary", ""))
    if t.catalyst_date is None and earnings_express is not None and not earnings_express.empty:
        latest = earnings_express.sort_values("announce_date", ascending=False).iloc[0]
        announce = latest.get("announce_date")
        if pd.notna(announce):
            t.catalyst_date = pd.Timestamp(announce).date()
            t.catalyst_type = t.catalyst_type or "earnings"
            t.auto_filled.append("catalyst_date")

    # ② Peer Anchoring
    if earnings_express is not None and not earnings_express.empty:
        industry = earnings_express.iloc[0].get("industry")
        if industry:
            t.peer_industry = str(industry)
            t.auto_filled.append("peer_industry")
    if industry_peer_pe_median is not None:
        t.peer_pe_median = float(industry_peer_pe_median)
        t.auto_filled.append("peer_pe_median")
    t.peer_bucket = str(thesis_input.get("peer_bucket", ""))
    t.peer_success = list(thesis_input.get("peer_success") or [])
    t.peer_failure = list(thesis_input.get("peer_failure") or [])

    # ③ Quality Differentiation (manual)
    for raw in thesis_input.get("quality_factors") or []:
        if isinstance(raw, str):
            t.quality_factors.append(QualityFactor(description=raw))
        elif isinstance(raw, dict):
            t.quality_factors.append(QualityFactor(
                description=str(raw.get("description", "")),
                evidence=str(raw.get("evidence", "")),
                direction=int(raw.get("direction", 1)),
            ))

    # ④ SOTP
    segs, sotp = _auto_fill_segments(
        business_segments,
        thesis_input.get("segment_pe_multiples"),
        thesis_input.get("manual_segments"),
    )
    t.segments = segs
    t.sotp_valuation = sotp
    mkt_cap = fund_data.get("market_cap")
    if mkt_cap is not None and pd.notna(mkt_cap):
        t.current_market_cap = float(mkt_cap)
        t.auto_filled.append("current_market_cap")
        if t.sotp_valuation:
            t.upside_pct = (t.sotp_valuation - t.current_market_cap) / t.current_market_cap

    # ⑤ Sentiment
    _auto_fill_sentiment(t, northbound, capital_flow)
    t.sentiment_notes = str(thesis_input.get("sentiment_notes", ""))

    # ⑥ Action
    sym_hist = history[history["symbol"] == symbol] if not history.empty else pd.DataFrame()
    if not sym_hist.empty:
        latest_close = pd.to_numeric(sym_hist.sort_values("date").iloc[-1]["close"], errors="coerce")
        if pd.notna(latest_close):
            t.current_price = float(latest_close)
            t.auto_filled.append("current_price")
    t.action = str(thesis_input.get("action", "watch"))
    t.thesis_notes = str(thesis_input.get("thesis_notes", ""))

    if not segs:
        t.missing.append("business_segments")
    if t.peer_pe_median is None:
        t.missing.append("peer_pe_median")
    if not t.quality_factors:
        t.missing.append("quality_factors")

    return t


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_thesis_markdown(t: EquityThesis) -> str:
    """Render a thesis to a Markdown report close to the source-article format."""
    lines: list[str] = []
    lines.append(f"# 投资论点:{t.name or t.symbol}  `{t.symbol}`")
    if t.as_of:
        lines.append(f"_as of {t.as_of.isoformat()}_")
    lines.append("")

    # Step 1
    lines.append("## ① 催化剂")
    if t.catalyst_type or t.catalyst_date or t.catalyst_summary:
        if t.catalyst_type:
            lines.append(f"- 类型:`{t.catalyst_type}`")
        if t.catalyst_date:
            lines.append(f"- 日期:{t.catalyst_date.isoformat()}")
        if t.catalyst_summary:
            lines.append(f"- 摘要:{t.catalyst_summary}")
    else:
        lines.append("_(无)_")
    lines.append("")

    # Step 2
    lines.append("## ② 同业锚定")
    if t.peer_industry:
        lines.append(f"- 行业:{t.peer_industry}")
    if t.peer_pe_median is not None:
        lines.append(f"- 行业 PE 中位:{t.peer_pe_median:.1f}")
    if t.peer_bucket:
        lines.append(f"- 锚定档次:{t.peer_bucket}")
    if t.peer_success:
        lines.append(f"- 成功对标:{', '.join(t.peer_success)}")
    if t.peer_failure:
        lines.append(f"- 失败对标:{', '.join(t.peer_failure)}")
    lines.append("")

    # Step 3
    lines.append("## ③ 质量加分项")
    if t.quality_factors:
        for i, qf in enumerate(t.quality_factors, 1):
            mark = "✅" if qf.direction > 0 else ("❌" if qf.direction < 0 else "—")
            lines.append(f"{i}. {mark} {qf.description}")
            if qf.evidence:
                lines.append(f"   _evidence: {qf.evidence}_")
    else:
        lines.append("_(待填)_")
    lines.append("")

    # Step 4
    lines.append("## ④ SOTP 分部估值")
    if t.segments:
        lines.append("")
        lines.append("| 分部 | 营收 | 利润 | 毛利率 | PE | 估值 |")
        lines.append("|---|---|---|---|---|---|")
        for s in t.segments:
            pe_str = f"{s.pe_multiple:.1f}" if s.pe_multiple is not None else "—"
            lines.append(
                f"| {s.name} "
                f"| {_yuan_to_yi(s.revenue)} "
                f"| {_yuan_to_yi(s.profit)} "
                f"| {_pct(s.gross_margin)} "
                f"| {pe_str} "
                f"| {_yuan_to_yi(s.valuation)} |"
            )
        lines.append("")
        if t.sotp_valuation is not None:
            lines.append(f"- **合计 SOTP 估值:{_yuan_to_yi(t.sotp_valuation)}**")
        if t.current_market_cap is not None:
            lines.append(f"- 当前市值:{_yuan_to_yi(t.current_market_cap)}")
        if t.upside_pct is not None:
            sign = "+" if t.upside_pct >= 0 else ""
            lines.append(f"- 上行空间:**{sign}{t.upside_pct * 100:.1f}%**")
    else:
        lines.append("_(无分部数据)_")
    lines.append("")

    # Step 5
    lines.append("## ⑤ 当前情绪")
    if t.north_30d_net_buy_cny is not None:
        sign = "+" if t.north_30d_net_buy_cny >= 0 else ""
        lines.append(f"- 北向 30 日净买入:{sign}{t.north_30d_net_buy_cny / 1e8:.2f}亿 CNY")
    if t.capital_flow_net_main is not None:
        sign = "+" if t.capital_flow_net_main >= 0 else ""
        lines.append(f"- 主力资金今日净流入(大+中):{sign}{t.capital_flow_net_main:.0f} 万")
    if t.capital_flow_net_total is not None:
        sign = "+" if t.capital_flow_net_total >= 0 else ""
        lines.append(f"- 全部资金今日净流入:{sign}{t.capital_flow_net_total:.0f} 万")
    if t.sentiment_notes:
        lines.append(f"- 备注:{t.sentiment_notes}")
    lines.append("")

    # Step 6
    lines.append("## ⑥ 行动建议")
    if t.current_price is not None:
        lines.append(f"- 当前价:{t.current_price:.2f}")
    lines.append(f"- 动作:**`{t.action}`**")
    if t.thesis_notes:
        lines.append("")
        lines.append(f"> {t.thesis_notes}")
    lines.append("")

    # Provenance
    if t.missing:
        lines.append(f"_⚠️ 数据缺失:{', '.join(t.missing)}_")

    return "\n".join(lines)
