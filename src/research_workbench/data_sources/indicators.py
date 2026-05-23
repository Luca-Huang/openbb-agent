from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the indicator schema consumed by the analysis layer."""
    if df.empty:
        return df
    df = df.sort_values("date").copy()
    close = pd.to_numeric(df["close"], errors="coerce")

    df["ma20"] = close.rolling(window=20, min_periods=1).mean()
    df["ma50"] = close.rolling(window=50, min_periods=1).mean()
    df["ma200"] = close.rolling(window=200, min_periods=1).mean()
    df["rsi14"] = compute_rsi(close, period=14)

    rolling_min = close.rolling(window=20, min_periods=1).min()
    df["support_level_primary"] = rolling_min
    df["support_level_secondary"] = rolling_min * 1.1

    volume = pd.to_numeric(df.get("volume"), errors="coerce")
    df["volume_ma20"] = volume.rolling(window=20, min_periods=1).mean()
    df["volume_spike_ratio"] = volume / df["volume_ma20"]

    closes = np.sort(close.dropna().to_numpy())
    if len(closes) > 0:
        df["close_percentile"] = close.map(
            lambda x: np.searchsorted(closes, x, side="right") / len(closes)
            if pd.notna(x) else np.nan
        )
    else:
        df["close_percentile"] = np.nan

    first_valid = close.dropna()
    df["close_norm"] = close / first_valid.iloc[0] if not first_valid.empty else np.nan
    return df

