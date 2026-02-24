create table if not exists public.equity_opportunity_radar (
    as_of_date date not null,
    priority_rank integer,
    symbol text not null,
    market text,
    trigger_type text,
    trigger_price double precision,
    stop_price double precision,
    opportunity_score double precision,
    risk_flags text,
    reason_1line text,
    value_score double precision,
    volume_spike_ratio double precision,
    drawdown_60d double precision,
    atr14 double precision,
    dollar_volume20 double precision,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    primary key (symbol, as_of_date)
);

create index if not exists idx_equity_opportunity_radar_asof
    on public.equity_opportunity_radar (as_of_date desc);

create index if not exists idx_equity_opportunity_radar_market
    on public.equity_opportunity_radar (market);

create index if not exists idx_equity_opportunity_radar_score
    on public.equity_opportunity_radar (opportunity_score desc);
