from __future__ import annotations

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


_EVENT_BASE_COLS = [
    "symbol", "event_date", "event_type", "importance",
    "impact", "summary", "event_value", "event_source",
]


def load_manual_events(settings: AppSettings) -> pd.DataFrame:
    if not settings.manual_events_path.exists():
        return pd.DataFrame(columns=_EVENT_BASE_COLS)
    df = pd.read_csv(settings.manual_events_path)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def load_auto_events(settings: AppSettings) -> pd.DataFrame:
    """Load auto-sourced events (from AKShare) if the artifact exists."""
    if not getattr(settings, "auto_events_path", None) or not settings.auto_events_path.exists():
        return pd.DataFrame(columns=_EVENT_BASE_COLS)
    df = pd.read_csv(settings.auto_events_path)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    if "event_value" in df.columns:
        df["event_value"] = pd.to_numeric(df["event_value"], errors="coerce")
    return df


def load_events(settings: AppSettings) -> pd.DataFrame:
    """Union of manual + auto-sourced events, manual taking precedence on conflict."""
    manual = load_manual_events(settings)
    auto = load_auto_events(settings)
    if manual.empty:
        return auto
    if auto.empty:
        return manual
    # Manual rows shadow auto on (symbol, event_date, event_type) duplicates.
    key_cols = ["symbol", "event_date", "event_type"]
    auto_keys = auto[key_cols].apply(tuple, axis=1) if all(c in auto.columns for c in key_cols) else None
    manual_keys = manual[key_cols].apply(tuple, axis=1) if all(c in manual.columns for c in key_cols) else None
    if auto_keys is None or manual_keys is None:
        return pd.concat([manual, auto], ignore_index=True)
    auto = auto[~auto_keys.isin(set(manual_keys))]
    return pd.concat([manual, auto], ignore_index=True)


def load_financials(settings: AppSettings) -> pd.DataFrame:
    if not getattr(settings, "financials_path", None) or not settings.financials_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(settings.financials_path)
    if "fiscal_period" in df.columns:
        df["fiscal_period"] = pd.to_datetime(df["fiscal_period"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def load_valuation_history(settings: AppSettings) -> pd.DataFrame:
    if not getattr(settings, "valuation_history_path", None) or not settings.valuation_history_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(settings.valuation_history_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def save_history(df: pd.DataFrame, settings: AppSettings) -> None:
    settings.history_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.history_path, index=False)


def save_summary(df: pd.DataFrame, settings: AppSettings) -> None:
    settings.summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.summary_path, index=False)


def save_signals_snapshot(df: pd.DataFrame, settings: AppSettings) -> None:
    settings.signals_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.signals_snapshot_path, index=False)


def save_financials(df: pd.DataFrame, settings: AppSettings) -> None:
    settings.financials_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.financials_path, index=False)


def save_financials_quarterly(df: pd.DataFrame, settings: AppSettings) -> None:
    settings.financials_quarterly_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.financials_quarterly_path, index=False)


def save_valuation_history(df: pd.DataFrame, settings: AppSettings) -> None:
    settings.valuation_history_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.valuation_history_path, index=False)


def save_earnings_express(df: pd.DataFrame, settings: AppSettings) -> None:
    settings.earnings_express_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.earnings_express_path, index=False)


def save_northbound_holdings(df: pd.DataFrame, settings: AppSettings) -> None:
    settings.northbound_holdings_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.northbound_holdings_path, index=False)


def save_auto_events(df: pd.DataFrame, settings: AppSettings) -> None:
    settings.auto_events_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.auto_events_path, index=False)
