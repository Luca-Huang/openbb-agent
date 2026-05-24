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
    ColumnDef("ma20", "float64", description="20 日均线"),
    ColumnDef("highest_close_20d", "float64", description="20 日最高收盘价"),
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
    ColumnDef("risk_unit", "float64", description="1R 风险单位"),
    ColumnDef("take_profit_1", "float64", description="1R 第一止盈位"),
    ColumnDef("take_profit_2", "float64", description="2R 第二止盈位"),
    ColumnDef("trailing_stop", "float64", description="剩余仓位移动止盈位"),
    ColumnDef("reasons", "str", description="信号原因（分号分隔）"),
    ColumnDef("invalidation_conditions", "str", description="失效条件"),
    ColumnDef("exit_plan", "str", description="分批止盈与移动止盈计划"),
    ColumnDef("exit_plan_notes", "str", description="止盈计算说明"),
    ColumnDef("data_source", "str", description="数据来源标识"),
    ColumnDef("updated_at", "datetime64[ns]", description="最后更新时间"),
]

# ---------------------------------------------------------------------------
# Annual Financial Statements (CN A-shares via AKShare/AKTools)
# ---------------------------------------------------------------------------
# Columns mirror what AKShareCNProvider.fetch_annual_financials() returns.
# Primary key is (symbol, fiscal_period). All monetary fields are in the
# issuer's reporting currency (CNY for A-shares); ratios are dimensionless
# decimals (e.g. 0.10 = 10%).
FINANCIAL_SCHEMA: list[ColumnDef] = [
    ColumnDef("symbol", "str", nullable=False, primary_key=True, description="股票代码"),
    ColumnDef("fiscal_period", "datetime64[ns]", nullable=False, primary_key=True, description="财报报告期"),
    ColumnDef("fiscal_year", "Int64", description="财年(冗余,等于 fiscal_period.year)"),
    ColumnDef("report_type", "str", description="annual / q1 / h1 / q3 — 由 fiscal_period 月份派生"),
    # 利润表
    ColumnDef("revenue", "float64", description="营业总收入"),
    ColumnDef("revenue_main", "float64", description="营业收入(主营)"),
    ColumnDef("operating_cost", "float64", description="营业成本"),
    ColumnDef("operating_profit", "float64", description="营业利润"),
    ColumnDef("total_profit", "float64", description="利润总额"),
    ColumnDef("net_income", "float64", description="净利润(含少数股东损益)"),
    ColumnDef("net_income_parent", "float64", description="归属于母公司所有者的净利润(归母净利)"),
    ColumnDef("basic_eps", "float64", description="基本每股收益"),
    ColumnDef("diluted_eps", "float64", description="稀释每股收益"),
    # 资产负债表
    ColumnDef("total_assets", "float64", description="资产总计"),
    ColumnDef("total_liabilities", "float64", description="负债合计"),
    ColumnDef("equity_parent", "float64", description="归属于母公司股东权益合计"),
    ColumnDef("total_equity", "float64", description="所有者权益合计(含少数股东权益)"),
    ColumnDef("cash_and_equivalents", "float64", description="货币资金"),
    ColumnDef("short_term_debt", "float64", description="短期借款"),
    ColumnDef("long_term_debt", "float64", description="长期借款"),
    ColumnDef("bonds_payable", "float64", description="应付债券"),
    ColumnDef("inventory", "float64", description="存货"),
    # 现金流量表
    ColumnDef("operating_cash_flow_net", "float64", description="经营活动产生的现金流量净额"),
    ColumnDef("investing_cash_flow_net", "float64", description="投资活动产生的现金流量净额"),
    ColumnDef("financing_cash_flow_net", "float64", description="筹资活动产生的现金流量净额"),
    ColumnDef("capex", "float64", description="资本开支(购建固定/无形/长期资产支付的现金)"),
    # 派生比率
    ColumnDef("net_margin", "float64", description="净利率 = net_income_parent / revenue"),
    ColumnDef("asset_liability_ratio", "float64", description="资产负债率 = total_liabilities / total_assets"),
    ColumnDef("free_cash_flow", "float64", description="自由现金流 ≈ OCF - capex"),
    ColumnDef("roe_simple", "float64", description="净资产收益率(简化) = net_income_parent / equity_parent"),
    ColumnDef("data_source", "str", description="数据来源标识(akshare_sina 等)"),
]


# ---------------------------------------------------------------------------
# Earnings Express (业绩快报 — preliminary results, A-shares via AKShare)
# ---------------------------------------------------------------------------
# Preliminary report released 1-4 weeks ahead of the formal annual/quarterly
# filing. Same fields as FINANCIAL_SCHEMA where they overlap but pre-computed
# YoY/QoQ growth + industry classification baked in.
EARNINGS_EXPRESS_SCHEMA: list[ColumnDef] = [
    ColumnDef("symbol", "str", nullable=False, primary_key=True, description="股票代码"),
    ColumnDef("fiscal_period", "datetime64[ns]", nullable=False, primary_key=True, description="报告期"),
    ColumnDef("name", "str", description="股票简称"),
    ColumnDef("industry", "str", description="所处行业(eastmoney 分类)"),
    ColumnDef("eps", "float64", description="每股收益"),
    ColumnDef("bps", "float64", description="每股净资产"),
    ColumnDef("revenue", "float64", description="营业总收入"),
    ColumnDef("revenue_yoy_pct", "float64", description="营业总收入同比增长(百分比)"),
    ColumnDef("revenue_qoq_pct", "float64", description="营业总收入季度环比增长"),
    ColumnDef("net_income", "float64", description="净利润"),
    ColumnDef("net_income_yoy_pct", "float64", description="净利润同比增长(百分比)"),
    ColumnDef("net_income_qoq_pct", "float64", description="净利润季度环比增长"),
    ColumnDef("roe_pct", "float64", description="净资产收益率(百分比)"),
    ColumnDef("ocf_per_share", "float64", description="每股经营现金流量"),
    ColumnDef("gross_margin_pct", "float64", description="销售毛利率(百分比)"),
    ColumnDef("announce_date", "datetime64[ns]", description="最新公告日期"),
]


# ---------------------------------------------------------------------------
# Historical Valuation (daily PE/PB/PS series, A-shares via AKShare)
# ---------------------------------------------------------------------------
# Used by score_hist_valuation to compute current-vs-history percentile of
# fundamental multiples. Eastmoney 'stock value' typically covers ~2018-present.
VALUATION_HISTORY_SCHEMA: list[ColumnDef] = [
    ColumnDef("symbol", "str", nullable=False, primary_key=True, description="股票代码"),
    ColumnDef("date", "datetime64[ns]", nullable=False, primary_key=True, description="估值日期"),
    ColumnDef("close", "float64", description="当日收盘价"),
    ColumnDef("market_cap", "float64", description="总市值"),
    ColumnDef("pe_ttm", "float64", description="市盈率(TTM)"),
    ColumnDef("pe_static", "float64", description="市盈率(静态)"),
    ColumnDef("pb", "float64", description="市净率"),
    ColumnDef("peg", "float64", description="PEG"),
    ColumnDef("pcf", "float64", description="市现率"),
    ColumnDef("ps", "float64", description="市销率"),
    ColumnDef("data_source", "str", description="数据来源标识"),
]


# ---------------------------------------------------------------------------
# Company Events
# ---------------------------------------------------------------------------
EVENT_SCHEMA: list[ColumnDef] = [
    ColumnDef("symbol", "str", nullable=False, primary_key=True, description="股票代码"),
    ColumnDef("event_date", "datetime64[ns]", nullable=False, primary_key=True, description="事件日期"),
    ColumnDef("event_type", "str", nullable=False, primary_key=True,
              description="财报 / 业绩预告 / 业绩快报 / 分红 / 股东增减持 / 限售解禁 / 管理层变化 / 重大公告"),
    ColumnDef("importance", "str", description="high / medium / low"),
    ColumnDef("impact", "str", description="positive / neutral / negative / mixed"),
    ColumnDef("summary", "str", description="事件摘要"),
    ColumnDef("event_value", "float64", description="事件相关数值(净利预测/分红金额/增减持股数 等)"),
    ColumnDef("event_source", "str", description="数据来源(manual / akshare_em / akshare_ths / akshare_sina)"),
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
