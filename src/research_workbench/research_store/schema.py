"""Formal schema definitions for core research workbench tables.

Each schema is expressed as a list of ``ColumnDef`` dataclasses that serve as
the single source of truth for column names, types, and constraints across
Supabase/Postgres tables, CSV fallbacks, and DataFrame operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ColumnDef:
    name: str
    dtype: str  # pandas dtype string: "str", "float64", "datetime64[ns]", "int64"
    nullable: bool = True
    primary_key: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------
WATCHLIST_SCHEMA: list[ColumnDef] = [
    ColumnDef("symbol", "str", nullable=False, primary_key=True, description="股票代码"),
    ColumnDef("name", "str", nullable=False, description="股票名称"),
    ColumnDef("market", "str", nullable=False, description="市场，固定 CN"),
    ColumnDef("sector", "str", description="所属板块"),
    ColumnDef("status", "str", description="watch / near / triggered / invalidated"),
    ColumnDef("target_zone_low", "float64", description="目标买入区间下限"),
    ColumnDef("target_zone_high", "float64", description="目标买入区间上限"),
    ColumnDef("notes", "str", description="人工备注"),
]

# ---------------------------------------------------------------------------
# Equity Metrics (summary)
# ---------------------------------------------------------------------------
SUMMARY_SCHEMA: list[ColumnDef] = [
    ColumnDef("symbol", "str", nullable=False, primary_key=True, description="股票代码"),
    ColumnDef("name", "str", description="股票名称"),
    ColumnDef("market", "str", description="市场"),
    ColumnDef("start_date", "datetime64[ns]", description="数据起始日"),
    ColumnDef("end_date", "datetime64[ns]", description="数据截止日"),
    ColumnDef("value_score", "float64", description="综合价值得分"),
    ColumnDef("value_score_tier", "str", description="价值等级"),
    ColumnDef("entry_recommendation", "str", description="入场建议"),
    ColumnDef("score_hist_valuation", "float64", description="历史估值分"),
    ColumnDef("score_abs_valuation", "float64", description="绝对估值分"),
    ColumnDef("score_peer_valuation", "float64", description="可比估值分"),
    ColumnDef("score_peg", "float64", description="PEG 分"),
    ColumnDef("score_growth_quality", "float64", description="增长质量分"),
    ColumnDef("score_balance_sheet", "float64", description="资产负债表分"),
    ColumnDef("score_shareholder_return", "float64", description="股东回报分"),
    ColumnDef("next_refresh_date", "datetime64[ns]", description="下次刷新日"),
    ColumnDef("data_source", "str", description="数据来源标识"),
    ColumnDef("updated_at", "datetime64[ns]", description="最后更新时间"),
]

# ---------------------------------------------------------------------------
# Equity Metrics History (daily rows)
# ---------------------------------------------------------------------------
HISTORY_SCHEMA: list[ColumnDef] = [
    ColumnDef("symbol", "str", nullable=False, primary_key=True, description="股票代码"),
    ColumnDef("date", "datetime64[ns]", nullable=False, primary_key=True, description="交易日期"),
    ColumnDef("open", "float64", description="开盘价"),
    ColumnDef("high", "float64", description="最高价"),
    ColumnDef("low", "float64", description="最低价"),
    ColumnDef("close", "float64", nullable=False, description="收盘价"),
    ColumnDef("volume", "float64", description="成交量"),
    ColumnDef("ma50", "float64", description="50 日均线"),
    ColumnDef("ma200", "float64", description="200 日均线"),
    ColumnDef("rsi14", "float64", description="14 日 RSI"),
    ColumnDef("support_level_primary", "float64", description="主支撑位"),
    ColumnDef("support_level_secondary", "float64", description="次支撑位"),
    ColumnDef("volume_spike_ratio", "float64", description="成交量异动倍数"),
    ColumnDef("volume_ma20", "float64", description="20 日均量"),
    ColumnDef("close_percentile", "float64", description="历史分位"),
    ColumnDef("fib_38_2", "float64", description="Fibonacci 38.2%"),
    ColumnDef("fib_50", "float64", description="Fibonacci 50%"),
    ColumnDef("fib_61_8", "float64", description="Fibonacci 61.8%"),
    ColumnDef("data_source", "str", description="数据来源标识"),
]

# ---------------------------------------------------------------------------
# Signals (output of signal engine)
# ---------------------------------------------------------------------------
SIGNAL_SCHEMA: list[ColumnDef] = [
    ColumnDef("symbol", "str", nullable=False, primary_key=True, description="股票代码"),
    ColumnDef("as_of_date", "datetime64[ns]", nullable=False, primary_key=True, description="信号日期"),
    ColumnDef("signal_state", "str", nullable=False, description="watch / near_zone / triggered / invalidated"),
    ColumnDef("trigger_type", "str", description="pullback / breakout / none"),
    ColumnDef("trigger_score", "float64", description="触发得分"),
    ColumnDef("valuation_score", "float64", description="估值得分"),
    ColumnDef("quality_score", "float64", description="质量得分"),
    ColumnDef("event_risk_score", "float64", description="事件风险分"),
    ColumnDef("conviction_score", "float64", description="综合信心分"),
    ColumnDef("reasons", "str", description="信号原因（分号分隔）"),
    ColumnDef("invalidation_conditions", "str", description="失效条件"),
    ColumnDef("data_source", "str", description="数据来源标识"),
    ColumnDef("updated_at", "datetime64[ns]", description="最后更新时间"),
]

# ---------------------------------------------------------------------------
# Company Events
# ---------------------------------------------------------------------------
EVENT_SCHEMA: list[ColumnDef] = [
    ColumnDef("symbol", "str", nullable=False, primary_key=True, description="股票代码"),
    ColumnDef("event_date", "datetime64[ns]", nullable=False, primary_key=True, description="事件日期"),
    ColumnDef("event_type", "str", nullable=False, description="财报 / 业绩预告 / 回购 / 分红 / 管理层变化 / 重大公告"),
    ColumnDef("importance", "str", description="high / medium / low"),
    ColumnDef("impact", "str", description="positive / neutral / negative / mixed"),
    ColumnDef("summary", "str", description="事件摘要"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def schema_columns(schema: list[ColumnDef]) -> list[str]:
    """Return the ordered list of column names for a schema."""
    return [col.name for col in schema]


def primary_keys(schema: list[ColumnDef]) -> list[str]:
    """Return the primary key column names for a schema."""
    return [col.name for col in schema if col.primary_key]
