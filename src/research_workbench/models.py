from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class WatchlistItem:
    symbol: str
    name: str
    market: str = "CN"
    sector: str = ""
    status: str = "watch"
    target_zone_low: Optional[float] = None
    target_zone_high: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True)
class CompanyEvent:
    symbol: str
    event_date: date
    event_type: str
    importance: str
    impact: str
    summary: str


@dataclass(frozen=True)
class SignalSnapshot:
    symbol: str
    as_of_date: date
    signal_state: str
    trigger_type: str
    trigger_score: float
    valuation_score: float
    quality_score: float
    event_risk_score: float
    conviction_score: float
    risk_unit: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    trailing_stop: Optional[float] = None
    reasons: str = ""
    invalidation_conditions: str = ""
    exit_plan: str = ""
    exit_plan_notes: str = ""


@dataclass(frozen=True)
class ValidationSummary:
    trigger_type: str
    sample_count: int
    win_rate_5d: float
    win_rate_20d: float
    avg_return_5d: float
    avg_return_20d: float
