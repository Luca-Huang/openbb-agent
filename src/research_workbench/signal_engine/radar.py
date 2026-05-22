from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RADAR_CONFIG: dict[str, Any] = {
    "triggers": {
        "pullback": {
            "ma50_distance_max": 0.03,
            "volume_spike_min": 1.1,
            "require_support_above_primary": True,
        },
        "breakout": {
            "volume_spike_min": 1.5,
            "lookback_high_days": 20,
        },
    },
    "risk": {
        "max_drawdown_60d_penalty_start": 0.22,
    },
    "exits": {
        "take_profit_1_r": 1.0,
        "take_profit_2_r": 2.0,
        "take_profit_1_allocation": "30%-50%",
        "take_profit_2_allocation": "20%-30%",
        "trailing_atr_multiple": 2.0,
    },
}


def add_radar_features(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or DEFAULT_RADAR_CONFIG
    sort_cols = ["symbol", "date"] if "symbol" in df.columns else ["date"]
    enriched = df.sort_values(sort_cols).copy()
    lookback_high = int(cfg.get("triggers", {}).get("breakout", {}).get("lookback_high_days", 20))
    close = pd.to_numeric(enriched.get("close"), errors="coerce")
    high = pd.to_numeric(enriched.get("high", close), errors="coerce")
    low = pd.to_numeric(enriched.get("low", close), errors="coerce")
    volume = pd.to_numeric(enriched.get("volume"), errors="coerce")

    if "symbol" in enriched.columns:
        enriched["_close"] = close
        enriched["_high"] = high
        enriched["_volume"] = volume
        grouped = enriched.groupby("symbol", group_keys=False)
        enriched["high_20d"] = grouped["_high"].transform(
            lambda s: s.rolling(window=lookback_high, min_periods=1).max().shift(1)
        )
        enriched["ma20"] = grouped["_close"].transform(lambda s: s.rolling(window=20, min_periods=1).mean())
        enriched["highest_close_20d"] = grouped["_close"].transform(lambda s: s.rolling(window=20, min_periods=1).max())
        prev_close = grouped["_close"].shift(1)
    else:
        enriched["high_20d"] = high.rolling(window=lookback_high, min_periods=1).max().shift(1)
        enriched["ma20"] = close.rolling(window=20, min_periods=1).mean()
        enriched["highest_close_20d"] = close.rolling(window=20, min_periods=1).max()
        prev_close = close.shift(1)

    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    if "symbol" in enriched.columns:
        enriched["_tr"] = true_range
        enriched["_dollar_volume"] = close * volume
        grouped = enriched.groupby("symbol", group_keys=False)
        enriched["atr14"] = grouped["_tr"].transform(lambda s: s.rolling(window=14, min_periods=1).mean())
        rolling_peak = grouped["_close"].transform(lambda s: s.rolling(window=60, min_periods=1).max())
        enriched["dollar_volume20"] = grouped["_dollar_volume"].transform(
            lambda s: s.rolling(window=20, min_periods=1).mean()
        )
        enriched = enriched.drop(columns=["_close", "_high", "_volume", "_tr", "_dollar_volume"])
    else:
        enriched["atr14"] = true_range.rolling(window=14, min_periods=1).mean()
        rolling_peak = close.rolling(window=60, min_periods=1).max()
        enriched["dollar_volume20"] = (close * volume).rolling(window=20, min_periods=1).mean()
    enriched["drawdown_60d"] = (rolling_peak - close) / rolling_peak.replace(0, np.nan)
    return enriched


def detect_trigger_type(latest_row: pd.Series, trigger_cfg: dict[str, Any]) -> str | None:
    close = pd.to_numeric(latest_row.get("close"), errors="coerce")
    ma50 = pd.to_numeric(latest_row.get("ma50"), errors="coerce")
    high_20d = pd.to_numeric(latest_row.get("high_20d"), errors="coerce")
    volume_spike = pd.to_numeric(latest_row.get("volume_spike_ratio"), errors="coerce")
    support_primary = pd.to_numeric(
        latest_row.get("support_level_primary", latest_row.get("support_level")),
        errors="coerce",
    )

    pullback_cfg = trigger_cfg.get("pullback", {})
    breakout_cfg = trigger_cfg.get("breakout", {})
    pullback_distance_max = float(pullback_cfg.get("ma50_distance_max", 0.03))
    pullback_volume_min = float(pullback_cfg.get("volume_spike_min", 1.1))
    require_support = bool(pullback_cfg.get("require_support_above_primary", True))
    breakout_volume_min = float(breakout_cfg.get("volume_spike_min", 1.5))

    if (
        pd.notna(close)
        and pd.notna(high_20d)
        and close > high_20d
        and pd.notna(volume_spike)
        and volume_spike >= breakout_volume_min
    ):
        return "breakout"

    near_ma50 = (
        pd.notna(close)
        and pd.notna(ma50)
        and ma50 > 0
        and abs(close - ma50) / ma50 <= pullback_distance_max
    )
    support_ok = True
    if require_support:
        support_ok = pd.notna(support_primary) and pd.notna(close) and close >= support_primary
    if near_ma50 and pd.notna(volume_spike) and volume_spike >= pullback_volume_min and support_ok:
        return "pullback"

    return None


def compute_risk_penalty(drawdown_60d: float | None, atr14: float | None, cfg: dict[str, Any]) -> float:
    penalty = 0.0
    dd_start = float(cfg.get("max_drawdown_60d_penalty_start", 0.22))
    if drawdown_60d is not None and not pd.isna(drawdown_60d) and drawdown_60d > dd_start:
        penalty += min(20.0, (drawdown_60d - dd_start) * 100)
    if atr14 is not None and not pd.isna(atr14):
        if atr14 >= 12:
            penalty += 6.0
        elif atr14 >= 8:
            penalty += 3.0
    return penalty


def _score_valuation(summary_row: pd.Series) -> float:
    candidates = [
        summary_row.get("score_hist_valuation"),
        summary_row.get("score_abs_valuation"),
        summary_row.get("score_peer_valuation"),
    ]
    return float(sum(float(v) for v in candidates if pd.notna(v)))


def _score_quality(summary_row: pd.Series) -> float:
    candidates = [
        summary_row.get("score_peg"),
        summary_row.get("score_growth_quality"),
        summary_row.get("score_balance_sheet"),
        summary_row.get("score_shareholder_return"),
    ]
    return float(sum(float(v) for v in candidates if pd.notna(v)))


def _event_risk_score(events_df: pd.DataFrame, symbol: str, as_of_date: pd.Timestamp) -> tuple[float, str]:
    if events_df.empty:
        return 0.0, ""
    subset = events_df[events_df["symbol"] == symbol].copy()
    if subset.empty:
        return 0.0, ""
    subset["event_date"] = pd.to_datetime(subset["event_date"], errors="coerce")
    subset = subset[subset["event_date"].notna()]
    subset = subset[subset["event_date"] <= as_of_date]
    subset = subset[subset["event_date"] >= as_of_date - pd.Timedelta(days=45)]
    if subset.empty:
        return 0.0, ""
    risk = 0.0
    summaries: list[str] = []
    for _, row in subset.sort_values("event_date", ascending=False).head(3).iterrows():
        impact = str(row.get("impact", "")).lower()
        importance = str(row.get("importance", "")).lower()
        summary = str(row.get("summary", "")).strip()
        if impact in {"negative", "bearish", "risk"}:
            risk += 12.0 if importance == "high" else 6.0
        elif impact in {"mixed", "neutral"}:
            risk += 2.0
        if summary:
            summaries.append(summary)
    return min(25.0, risk), " | ".join(summaries)


def _target_zone_distance(close: float | None, target_low: float | None, target_high: float | None) -> float | None:
    if close is None or pd.isna(close) or target_low is None or pd.isna(target_low):
        return None
    target_high = target_high if target_high is not None and not pd.isna(target_high) else target_low
    if target_low <= close <= target_high:
        return 0.0
    if close < target_low:
        return (target_low - close) / target_low
    return (close - target_high) / target_high


def _signal_state(trigger_type: str | None, zone_distance: float | None, close: float | None, stop_price: float | None) -> str:
    if pd.notna(close) and pd.notna(stop_price) and close is not None and stop_price is not None and close < stop_price:
        return "invalidated"
    if trigger_type:
        return "triggered"
    if zone_distance is not None and zone_distance <= 0.08:
        return "near_zone"
    return "watch"


def _fmt_price(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}"


def compute_exit_plan(
    entry_price: float | None,
    stop_price: float | None,
    atr14: float | None = None,
    ma20: float | None = None,
    highest_close_20d: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reference take-profit plan from R multiples plus a trailing stop.

    The signal engine does not know the user's actual fill price or position age,
    so this uses the latest trigger price as the reference entry. ``atr14`` /
    ``ma20`` / ``highest_close_20d`` are optional; without them the trailing stop
    is left undefined (callers that only need TP levels can omit them).
    """
    cfg = cfg or {}
    tp1_r = float(cfg.get("take_profit_1_r", 1.0))
    tp2_r = float(cfg.get("take_profit_2_r", 2.0))
    tp1_alloc = str(cfg.get("take_profit_1_allocation", "30%-50%"))
    tp2_alloc = str(cfg.get("take_profit_2_allocation", "20%-30%"))
    atr_multiple = float(cfg.get("trailing_atr_multiple", 2.0))

    if (
        entry_price is None
        or stop_price is None
        or pd.isna(entry_price)
        or pd.isna(stop_price)
        or float(entry_price) <= float(stop_price)
    ):
        return {
            "risk_unit": np.nan,
            "take_profit_1": np.nan,
            "take_profit_2": np.nan,
            "trailing_stop": np.nan,
            "exit_plan": "No take-profit plan: entry must be above stop_price.",
            "exit_plan_notes": "Use position sizing first; skip the setup if the stop is not below entry.",
        }

    entry = float(entry_price)
    stop = float(stop_price)
    risk_unit = entry - stop
    take_profit_1 = entry + tp1_r * risk_unit
    take_profit_2 = entry + tp2_r * risk_unit

    trailing_candidates: list[float] = []
    if ma20 is not None and pd.notna(ma20) and float(ma20) <= entry:
        trailing_candidates.append(float(ma20))
    if (
        highest_close_20d is not None
        and pd.notna(highest_close_20d)
        and atr14 is not None
        and pd.notna(atr14)
    ):
        atr_trail = float(highest_close_20d) - atr_multiple * float(atr14)
        if atr_trail <= entry:
            trailing_candidates.append(atr_trail)
    trailing_stop = max(trailing_candidates) if trailing_candidates else np.nan

    exit_plan = (
        f"TP1 {_fmt_price(take_profit_1)} ({tp1_r:.1f}R, sell {tp1_alloc}); "
        f"TP2 {_fmt_price(take_profit_2)} ({tp2_r:.1f}R, sell {tp2_alloc}); "
        f"trail rest at {_fmt_price(trailing_stop)}."
    )
    exit_plan_notes = (
        f"R = entry - stop = {_fmt_price(risk_unit)}. "
        f"Trailing stop uses max(MA20, 20d highest close - {atr_multiple:.1f} * ATR14) when below entry."
    )
    return {
        "risk_unit": risk_unit,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "trailing_stop": trailing_stop,
        "exit_plan": exit_plan,
        "exit_plan_notes": exit_plan_notes,
    }


def build_current_signals(
    summary_df: pd.DataFrame,
    history_df: pd.DataFrame,
    watchlist_df: pd.DataFrame,
    events_df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = cfg or DEFAULT_RADAR_CONFIG
    if summary_df.empty or history_df.empty:
        return pd.DataFrame()

    latest = (
        history_df.sort_values("date")
        .dropna(subset=["symbol"])
        .groupby("symbol")
        .tail(1)
        .copy()
    )
    latest = add_radar_features(history_df, cfg).sort_values("date").groupby("symbol").tail(1)
    latest["symbol"] = latest["symbol"].astype(str).str.upper()

    summary = summary_df.copy()
    summary["symbol"] = summary["symbol"].astype(str).str.upper()
    merged = latest.merge(summary, on=["symbol"], how="inner", suffixes=("_hist", "_summary"))
    if not watchlist_df.empty:
        watch = watchlist_df.copy()
        watch["symbol"] = watch["symbol"].astype(str).str.upper()
        merged = merged.merge(watch, on="symbol", how="inner", suffixes=("", "_watch"))

    rows: list[dict[str, Any]] = []
    trigger_cfg = cfg.get("triggers", {})
    risk_cfg = cfg.get("risk", {})
    exit_cfg = cfg.get("exits", {})

    for _, row in merged.iterrows():
        close = pd.to_numeric(row.get("close"), errors="coerce")
        support_secondary = pd.to_numeric(row.get("support_level_secondary"), errors="coerce")
        ma200 = pd.to_numeric(row.get("ma200"), errors="coerce")
        stop_price = support_secondary if pd.notna(support_secondary) else (ma200 * 0.97 if pd.notna(ma200) else np.nan)
        trigger_type = detect_trigger_type(row, trigger_cfg)
        valuation_score = _score_valuation(row)
        quality_score = _score_quality(row)
        event_risk_score, event_summary = _event_risk_score(events_df, row["symbol"], pd.to_datetime(row["date"]))
        drawdown_60d = pd.to_numeric(row.get("drawdown_60d"), errors="coerce")
        atr14 = pd.to_numeric(row.get("atr14"), errors="coerce")
        ma20 = pd.to_numeric(row.get("ma20"), errors="coerce")
        highest_close_20d = pd.to_numeric(row.get("highest_close_20d"), errors="coerce")
        volume_spike = pd.to_numeric(row.get("volume_spike_ratio"), errors="coerce")
        value_score = pd.to_numeric(row.get("value_score"), errors="coerce")
        trigger_bonus = 8.0 if trigger_type == "breakout" else (6.0 if trigger_type == "pullback" else 0.0)
        volume_bonus = 0.0 if pd.isna(volume_spike) else max(0.0, min(12.0, (float(volume_spike) - 1.0) * 10))
        risk_penalty = compute_risk_penalty(drawdown_60d, atr14, risk_cfg)
        opportunity_score = max(
            0.0,
            min(100.0, float(value_score if pd.notna(value_score) else 0.0) + trigger_bonus + volume_bonus - risk_penalty),
        )
        zone_distance = _target_zone_distance(close, row.get("target_zone_low"), row.get("target_zone_high"))
        signal_state = _signal_state(trigger_type, zone_distance, close, stop_price)
        conviction_score = max(0.0, min(100.0, opportunity_score + valuation_score * 0.2 + quality_score * 0.2 - event_risk_score))
        reasons = [
            f"trigger={trigger_type or 'none'}",
            f"value_score={float(value_score):.1f}" if pd.notna(value_score) else "value_score=N/A",
            f"valuation_score={valuation_score:.1f}",
            f"quality_score={quality_score:.1f}",
        ]
        if pd.notna(volume_spike):
            reasons.append(f"volume_spike={float(volume_spike):.2f}x")
        if zone_distance is not None:
            reasons.append(f"zone_distance={zone_distance:.1%}")
        if event_summary:
            reasons.append(f"event={event_summary}")
        invalidation = []
        if pd.notna(stop_price):
            invalidation.append(f"close < {float(stop_price):.2f}")
        if pd.notna(ma200):
            invalidation.append(f"below_ma200={float(ma200):.2f}")
        exit_plan = compute_exit_plan(close, stop_price, atr14, ma20, highest_close_20d, exit_cfg)
        rows.append(
            {
                "symbol": row["symbol"],
                "name": row.get("name"),
                "market": row.get("market", "CN"),
                "sector": row.get("sector", ""),
                "status": row.get("status", "watch"),
                "date": pd.to_datetime(row["date"]),
                "signal_state": signal_state,
                "trigger_type": trigger_type,
                "trigger_score": round(opportunity_score, 2),
                "valuation_score": round(valuation_score, 2),
                "quality_score": round(quality_score, 2),
                "event_risk_score": round(event_risk_score, 2),
                "conviction_score": round(conviction_score, 2),
                "trigger_price": close,
                "stop_price": stop_price,
                "risk_unit": exit_plan["risk_unit"],
                "take_profit_1": exit_plan["take_profit_1"],
                "take_profit_2": exit_plan["take_profit_2"],
                "trailing_stop": exit_plan["trailing_stop"],
                "target_zone_low": row.get("target_zone_low"),
                "target_zone_high": row.get("target_zone_high"),
                "zone_distance": zone_distance,
                "value_score": value_score,
                "entry_recommendation": row.get("entry_recommendation"),
                "reasons": "; ".join(reasons),
                "invalidation_conditions": "; ".join(invalidation),
                "exit_plan": exit_plan["exit_plan"],
                "exit_plan_notes": exit_plan["exit_plan_notes"],
                "event_summary": event_summary,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["signal_state", "conviction_score"], ascending=[True, False])
