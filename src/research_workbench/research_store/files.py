from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from research_workbench.config import AppSettings



def load_summary(settings: AppSettings) -> pd.DataFrame:
    if not settings.summary_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(settings.summary_path)
    for col in ["start_date", "end_date", "next_refresh_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    if "name" not in df.columns:
        df["name"] = (
            df.get("name_cn", pd.Series(index=df.index, dtype=str))
            .fillna(df.get("name_en", pd.Series(index=df.index, dtype=str)))
            .fillna(df.get("symbol", pd.Series(index=df.index, dtype=str)))
        )
    return df


def load_history(settings: AppSettings) -> pd.DataFrame:
    if not settings.history_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(settings.history_path)
    if "as_of_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"as_of_date": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    if "support_level" not in df.columns and "support_level_primary" in df.columns:
        df["support_level"] = pd.to_numeric(df["support_level_primary"], errors="coerce")
    numeric_candidates = [
        "open",
        "high",
        "low",
        "close",
        "close_norm",
        "close_percentile",
        "support_level",
        "support_level_primary",
        "support_level_secondary",
        "ttm_eps",
        "pe",
        "ps_ratio",
        "ma50",
        "ma200",
        "ma20",
        "highest_close_20d",
        "rsi14",
        "fib_38_2",
        "fib_50",
        "fib_61_8",
        "volume",
        "volume_ma20",
        "volume_spike_ratio",
        "obv",
        "vpt",
        "vwap",
        "ad_line",
        "risk_unit",
        "take_profit_1",
        "take_profit_2",
        "trailing_stop",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_radar(settings: AppSettings) -> pd.DataFrame:
    if not settings.radar_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(settings.radar_path)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    for col in ["trigger_price", "stop_price", "risk_unit", "take_profit_1", "take_profit_2", "trailing_stop"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_manual_events(settings: AppSettings) -> pd.DataFrame:
    if not settings.manual_events_path.exists():
        return pd.DataFrame(
            columns=["symbol", "event_date", "event_type", "importance", "impact", "summary"]
        )
    df = pd.read_csv(settings.manual_events_path)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df
