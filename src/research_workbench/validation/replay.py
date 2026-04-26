from __future__ import annotations

import numpy as np
import pandas as pd

from research_workbench.signal_engine.radar import add_radar_features, detect_trigger_type, DEFAULT_RADAR_CONFIG


def _max_drawdown_in_window(closes: pd.Series) -> float:
    """Return the maximum peak-to-trough drawdown within a price window.

    The result is expressed as a positive fraction (e.g. 0.12 means 12% drawdown).
    Returns ``np.nan`` when the window has fewer than 2 valid observations.
    """
    vals = pd.to_numeric(closes, errors="coerce").dropna()
    if len(vals) < 2:
        return np.nan
    running_max = vals.cummax()
    drawdowns = (running_max - vals) / running_max.replace(0, np.nan)
    return float(drawdowns.max())


def replay_trigger_history(
    history_df: pd.DataFrame,
    symbols: list[str] | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    cfg = cfg or DEFAULT_RADAR_CONFIG
    if history_df.empty:
        return pd.DataFrame()
    df = history_df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    if symbols:
        allowed = {symbol.upper() for symbol in symbols}
        df = df[df["symbol"].isin(allowed)]
    if df.empty:
        return pd.DataFrame()
    replays: list[dict] = []
    for symbol, group in df.sort_values("date").groupby("symbol"):
        enriched = add_radar_features(group, cfg).reset_index(drop=True)
        for idx, row in enriched.iterrows():
            trigger_type = detect_trigger_type(row, cfg.get("triggers", {}))
            if not trigger_type:
                continue
            current_close = pd.to_numeric(row.get("close"), errors="coerce")
            if pd.isna(current_close) or current_close == 0:
                continue
            record = {
                "symbol": symbol,
                "date": pd.to_datetime(row["date"]),
                "trigger_type": trigger_type,
                "close": current_close,
            }
            for horizon in (5, 10, 20, 60):
                end_idx = idx + horizon
                if end_idx >= len(enriched):
                    record[f"return_{horizon}d"] = np.nan
                    record[f"max_drawdown_{horizon}d"] = np.nan
                    continue
                future_close = pd.to_numeric(enriched.iloc[end_idx]["close"], errors="coerce")
                if pd.isna(future_close):
                    record[f"return_{horizon}d"] = np.nan
                else:
                    record[f"return_{horizon}d"] = (future_close - current_close) / current_close
                # max drawdown within the forward window (inclusive of trigger day)
                window_closes = enriched.iloc[idx : end_idx + 1]["close"]
                record[f"max_drawdown_{horizon}d"] = _max_drawdown_in_window(window_closes)
            replays.append(record)
    return pd.DataFrame(replays)


def build_validation_summary(replay_df: pd.DataFrame) -> pd.DataFrame:
    if replay_df.empty:
        return pd.DataFrame(
            columns=[
                "trigger_type",
                "sample_count",
                "win_rate_5d",
                "win_rate_20d",
                "avg_return_5d",
                "avg_return_20d",
                "max_drawdown_20d",
            ]
        )
    rows = []
    for trigger_type, group in replay_df.groupby("trigger_type"):
        ret_5d = pd.to_numeric(group["return_5d"], errors="coerce")
        ret_20d = pd.to_numeric(group["return_20d"], errors="coerce")
        mdd_20d = pd.to_numeric(group.get("max_drawdown_20d", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "trigger_type": trigger_type,
                "sample_count": int(len(group)),
                "win_rate_5d": float(ret_5d.gt(0).mean()) if ret_5d.notna().any() else np.nan,
                "win_rate_20d": float(ret_20d.gt(0).mean()) if ret_20d.notna().any() else np.nan,
                "avg_return_5d": float(ret_5d.mean()) if ret_5d.notna().any() else np.nan,
                "avg_return_20d": float(ret_20d.mean()) if ret_20d.notna().any() else np.nan,
                "max_drawdown_20d": float(mdd_20d.mean()) if mdd_20d.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("sample_count", ascending=False)


def build_symbol_validation(replay_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if replay_df.empty:
        return pd.DataFrame()
    subset = replay_df[replay_df["symbol"] == symbol.upper()].copy()
    if subset.empty:
        return subset
    return subset.sort_values("date", ascending=False)

