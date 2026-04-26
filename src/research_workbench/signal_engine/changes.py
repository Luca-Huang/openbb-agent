"""Detect daily status changes across watchlist symbols.

Compares the latest two snapshots of signal data to highlight:
- New entries into the observation zone
- Exits from the zone
- Trigger type changes
- Significant conviction score movements
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_daily_changes(
    current_signals: pd.DataFrame,
    previous_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of today's notable changes.

    Each row describes one change for a symbol with columns:
    ``symbol``, ``name``, ``change_type``, ``detail``, ``priority``.

    If *previous_signals* is ``None`` or empty, every triggered/near-zone
    signal is treated as a "new" event so the board is never blank.
    """
    changes: list[dict[str, Any]] = []

    if current_signals.empty:
        return pd.DataFrame(
            columns=["symbol", "name", "change_type", "detail", "priority"]
        )

    # Build lookup for previous state
    prev_map: dict[str, pd.Series] = {}
    if previous_signals is not None and not previous_signals.empty:
        for _, row in previous_signals.iterrows():
            sym = str(row.get("symbol", "")).upper()
            if sym:
                prev_map[sym] = row

    for _, cur in current_signals.iterrows():
        symbol = str(cur.get("symbol", "")).upper()
        name = cur.get("name", symbol)
        cur_state = str(cur.get("signal_state", "watch"))
        cur_trigger = cur.get("trigger_type")
        cur_conviction = pd.to_numeric(cur.get("conviction_score"), errors="coerce")

        prev = prev_map.get(symbol)

        if prev is None:
            # No previous record – report if currently interesting
            if cur_state in ("triggered", "near_zone"):
                changes.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "change_type": "新进入" if cur_state == "near_zone" else "新触发",
                        "detail": f"状态={cur_state}, 触发={cur_trigger or 'none'}, conviction={_fmt(cur_conviction)}",
                        "priority": 1 if cur_state == "triggered" else 2,
                    }
                )
            continue

        prev_state = str(prev.get("signal_state", "watch"))
        prev_trigger = prev.get("trigger_type")
        prev_conviction = pd.to_numeric(prev.get("conviction_score"), errors="coerce")

        # State transition
        if cur_state != prev_state:
            if cur_state == "triggered":
                changes.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "change_type": "新触发",
                        "detail": f"{prev_state} → {cur_state}，触发类型={cur_trigger or 'none'}",
                        "priority": 1,
                    }
                )
            elif cur_state == "near_zone" and prev_state == "watch":
                changes.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "change_type": "进入观察区间",
                        "detail": f"{prev_state} → {cur_state}",
                        "priority": 2,
                    }
                )
            elif cur_state == "invalidated":
                changes.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "change_type": "信号失效",
                        "detail": f"{prev_state} → invalidated",
                        "priority": 1,
                    }
                )
            elif cur_state == "watch" and prev_state in ("triggered", "near_zone"):
                changes.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "change_type": "脱离区间",
                        "detail": f"{prev_state} → watch",
                        "priority": 3,
                    }
                )

        # Trigger type change (within same state)
        if cur_state == prev_state and cur_trigger != prev_trigger and cur_trigger:
            changes.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "change_type": "触发类型变化",
                    "detail": f"{prev_trigger or 'none'} → {cur_trigger}",
                    "priority": 2,
                }
            )

        # Conviction score significant movement (>= 5 points)
        if pd.notna(cur_conviction) and pd.notna(prev_conviction):
            delta = cur_conviction - prev_conviction
            if abs(delta) >= 5:
                direction = "↑" if delta > 0 else "↓"
                changes.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "change_type": f"conviction {direction}",
                        "detail": f"{prev_conviction:.1f} → {cur_conviction:.1f} ({delta:+.1f})",
                        "priority": 2 if delta > 0 else 3,
                    }
                )

    out = pd.DataFrame(changes)
    if out.empty:
        return pd.DataFrame(
            columns=["symbol", "name", "change_type", "detail", "priority"]
        )
    return out.sort_values("priority").reset_index(drop=True)


def _fmt(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:.1f}"
