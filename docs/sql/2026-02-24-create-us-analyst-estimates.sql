create table if not exists public.us_analyst_estimates (
    "symbol（股票代码）" text not null,
    "date（财年/报告期）" date not null,
    "revenueLow（营收下限）" double precision,
    "revenueHigh（营收上限）" double precision,
    "revenueAvg（营收均值）" double precision,
    "ebitdaLow（EBITDA下限）" double precision,
    "ebitdaHigh（EBITDA上限）" double precision,
    "ebitdaAvg（EBITDA均值）" double precision,
    "ebitLow（EBIT下限）" double precision,
    "ebitHigh（EBIT上限）" double precision,
    "ebitAvg（EBIT均值）" double precision,
    "netIncomeLow（净利润下限）" double precision,
    "netIncomeHigh（净利润上限）" double precision,
    "netIncomeAvg（净利润均值）" double precision,
    "sgaExpenseLow（销售与管理费用下限）" double precision,
    "sgaExpenseHigh（销售与管理费用上限）" double precision,
    "sgaExpenseAvg（销售与管理费用均值）" double precision,
    "epsAvg（每股收益均值）" double precision,
    "epsHigh（每股收益上限）" double precision,
    "epsLow（每股收益下限）" double precision,
    "numAnalystsRevenue（营收预测分析师数）" double precision,
    "numAnalystsEps（EPS预测分析师数）" double precision,
    "name（公司名称）" text,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    primary key ("symbol（股票代码）", "date（财年/报告期）")
);

create index if not exists idx_us_analyst_estimates_symbol
    on public.us_analyst_estimates ("symbol（股票代码）");

create index if not exists idx_us_analyst_estimates_date
    on public.us_analyst_estimates ("date（财年/报告期）" desc);
