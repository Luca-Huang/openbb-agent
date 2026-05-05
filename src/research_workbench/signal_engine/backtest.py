"""Backtest helpers built on top of the production radar trigger logic.

Three things live here:

* :func:`compute_position_pnl` — accounting for an arbitrary BUY/SELL log.
  Returns both the moving-average view (true average cost, realized + unrealized)
  and the broker view (a.k.a. 摊薄成本: gross-buys minus gross-sells).
  Both views report the **same** total P&L; they just split it differently.

* :func:`decision_review` — for each historical trade, attach the indicator
  snapshot and the verdict produced by :func:`radar.detect_trigger_type` (the
  same rule the production radar uses).  Verdict is ACCEPT iff the production
  radar would have classified the bar as a `breakout` or `pullback`.

* :func:`simulate_strategy` — replay the strategy from a forced initial buy
  forward, using the same trigger rule for re-entries and an exit plan derived
  from :func:`radar.compute_exit_plan` (TP1/TP2/trailing stop / MA50 break).

The history dataframe is expected to follow the same column convention as
`fetch_cn_data.py` + `radar.add_radar_features`: ``date, close, high, low,
volume, ma20, ma50, ma200, rsi14, atr14, support_level_primary,
volume_spike_ratio, high_20d, highest_close_20d``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .radar import DEFAULT_RADAR_CONFIG, compute_exit_plan, detect_trigger_type


# ---------- portfolio P&L ----------

@dataclass
class PositionPnl:
    """Two equivalent views of a position's P&L.

    Both ``moving_avg.total_pnl`` and ``broker.total_pnl`` must agree by
    construction; they only differ in how realized vs floating is split.
    """
    holdings: float
    last_price: float

    # Moving-average view
    avg_cost: float            # weighted average of all currently-held shares
    realized_pnl: float        # sum over sells of (price - avg_at_sell) * shares
    unrealized_pnl: float      # (last_price - avg_cost) * holdings
    max_avg_cost_basis: float  # peak of (avg_cost * holdings) over the trade log

    # Broker view (摊薄)
    broker_cost: float         # gross_buys - gross_sells, divided by holdings
    floating_pnl: float        # holdings * last_price - net_invested
    floating_pnl_pct: float    # floating_pnl / net_invested
    net_invested: float        # gross_buys - gross_sells

    # Aggregate
    total_pnl: float           # realized + unrealized == floating_pnl
    return_on_max_capital: float  # total_pnl / max_avg_cost_basis


def compute_position_pnl(trades: Iterable[dict[str, Any]], last_price: float) -> PositionPnl:
    """Walk a BUY/SELL log and return both accounting views.

    Each trade is ``{"side": "BUY"|"SELL", "price": float, "shares": float}``.
    Trades must be in chronological order.

    The function is purely arithmetic and does not depend on any market data.
    """
    holdings = 0.0
    avg_cost_basis = 0.0  # moving-average cost basis = avg_cost * holdings
    realized_pnl = 0.0
    max_avg_cost_basis = 0.0

    gross_buys = 0.0
    gross_sells = 0.0

    for t in trades:
        side = t["side"]
        price = float(t["price"])
        shares = float(t["shares"])
        if side == "BUY":
            avg_cost_basis += price * shares
            holdings += shares
            gross_buys += price * shares
        elif side == "SELL":
            if holdings <= 0:
                raise ValueError("SELL with no holdings")
            avg = avg_cost_basis / holdings
            realized_pnl += (price - avg) * shares
            avg_cost_basis -= avg * shares
            holdings -= shares
            gross_sells += price * shares
        else:
            raise ValueError(f"Unknown side: {side}")
        max_avg_cost_basis = max(max_avg_cost_basis, avg_cost_basis)

    avg_cost = avg_cost_basis / holdings if holdings > 0 else 0.0
    unrealized_pnl = (last_price - avg_cost) * holdings if holdings > 0 else 0.0

    net_invested = gross_buys - gross_sells
    broker_cost = net_invested / holdings if holdings > 0 else 0.0
    floating_pnl = holdings * last_price - net_invested
    floating_pct = floating_pnl / net_invested if net_invested > 0 else 0.0

    total = realized_pnl + unrealized_pnl
    # Sanity check: the two views must agree on total P&L.
    if not math.isclose(total, floating_pnl, abs_tol=1e-6):
        raise AssertionError(
            f"P&L identity broken: moving_avg={total} broker={floating_pnl}"
        )

    return PositionPnl(
        holdings=holdings,
        last_price=last_price,
        avg_cost=avg_cost,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        max_avg_cost_basis=max_avg_cost_basis,
        broker_cost=broker_cost,
        floating_pnl=floating_pnl,
        floating_pnl_pct=floating_pct,
        net_invested=net_invested,
        total_pnl=total,
        return_on_max_capital=total / max_avg_cost_basis if max_avg_cost_basis > 0 else 0.0,
    )


# ---------- decision review (production-rule verdict) ----------

REQUIRED_COLUMNS = (
    "date", "close", "high", "low", "volume",
    "ma20", "ma50", "ma200",
    "atr14", "high_20d",
    "support_level_primary",
    "volume_spike_ratio",
)


def _row_for_date(history_df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    exact = history_df[history_df["date"] == ts]
    if not exact.empty:
        return exact.iloc[0]
    earlier = history_df[history_df["date"] <= ts]
    return earlier.iloc[-1] if not earlier.empty else None


def decision_review(
    history_df: pd.DataFrame,
    trades: Iterable[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Snap each user trade to its bar, attach indicators and verdict.

    ACCEPT iff :func:`detect_trigger_type` returns ``"breakout"`` or
    ``"pullback"`` for that bar.  SELL rows have an empty verdict; we still
    record them so the review CSV is a faithful trade log with context.
    """
    cfg = cfg or DEFAULT_RADAR_CONFIG
    trigger_cfg = cfg.get("triggers", {})
    missing = [c for c in REQUIRED_COLUMNS if c not in history_df.columns]
    if missing:
        raise KeyError(f"history_df missing columns: {missing}")

    df = history_df.sort_values("date").reset_index(drop=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    rows: list[dict[str, Any]] = []
    for t in trades:
        ts = pd.to_datetime(t["date"])
        row = _row_for_date(df, ts)
        if row is None:
            continue
        trigger = detect_trigger_type(row, trigger_cfg)
        verdict = ""
        if t["side"] == "BUY":
            verdict = "ACCEPT" if trigger in {"breakout", "pullback"} else "REJECT"
        rows.append({
            "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "side": t["side"],
            "price": float(t["price"]),
            "shares": float(t["shares"]),
            "close": float(row["close"]),
            "ma20": float(row["ma20"]) if pd.notna(row.get("ma20")) else np.nan,
            "ma50": float(row["ma50"]) if pd.notna(row.get("ma50")) else np.nan,
            "ma200": float(row["ma200"]) if pd.notna(row.get("ma200")) else np.nan,
            "atr14": float(row["atr14"]) if pd.notna(row.get("atr14")) else np.nan,
            "rsi14": float(row["rsi14"]) if pd.notna(row.get("rsi14")) else np.nan,
            "vol_spike": float(row["volume_spike_ratio"]) if pd.notna(row.get("volume_spike_ratio")) else np.nan,
            "dist_ma50": (float(row["close"]) - float(row["ma50"])) / float(row["ma50"])
                         if pd.notna(row.get("ma50")) and float(row["ma50"]) > 0 else np.nan,
            "trigger_type": trigger or "",
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


# ---------- strategy simulation ----------

@dataclass
class _OpenPosition:
    shares: float
    entry: float
    stop: float
    tp1: float
    tp2: float
    trail: float
    tp1_done: bool = False


def _stop_from_row(row: pd.Series, fallback_atr_mult: float = 2.0) -> float:
    """Initial stop = max(entry - 2*ATR, MA50). Mirrors radar's risk model."""
    entry = float(row["close"])
    candidates = [entry * 0.85]  # absolute floor
    if pd.notna(row.get("atr14")):
        candidates.append(entry - fallback_atr_mult * float(row["atr14"]))
    if pd.notna(row.get("ma50")):
        candidates.append(float(row["ma50"]))
    return max(candidates)


def simulate_strategy(
    history_df: pd.DataFrame,
    initial_buy: dict[str, Any],
    allow_reentry: bool,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replay strategy from ``initial_buy`` forward.

    ``initial_buy`` = ``{"date": "...", "price": float, "shares": float}``.
    Forced into the position at the user's price/size. After that, exits and
    re-entries follow the production rule (``detect_trigger_type``).

    Returns (log_df, summary).  Summary keys: ``pnl, max_capital, return``.
    """
    cfg = cfg or DEFAULT_RADAR_CONFIG
    trigger_cfg = cfg.get("triggers", {})
    exit_cfg = cfg.get("exits", {})
    atr_mult = float(exit_cfg.get("trailing_atr_multiple", 2.0))

    df = history_df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    start_ts = pd.to_datetime(initial_buy["date"])
    forward = df[df["date"] >= start_ts].reset_index(drop=True)
    if forward.empty:
        return pd.DataFrame(), {"pnl": 0.0, "max_capital": 0.0, "return": 0.0}

    log: list[dict[str, Any]] = []
    pos: _OpenPosition | None = None
    realized = 0.0
    max_capital = 0.0

    # ---- forced initial buy at user's price ----
    r0 = forward.iloc[0]
    entry = float(initial_buy["price"])
    stop = _stop_from_row(r0, atr_mult) if pd.notna(r0.get("ma50")) else entry * 0.85
    if stop >= entry:
        stop = entry * 0.92
    R = entry - stop
    pos = _OpenPosition(
        shares=float(initial_buy["shares"]),
        entry=entry,
        stop=stop,
        tp1=entry + R,
        tp2=entry + 2 * R,
        trail=stop,
    )
    capital = pos.shares * entry
    max_capital = max(max_capital, capital)
    log.append({
        "date": r0["date"].strftime("%Y-%m-%d"),
        "action": "FORCED_BUY", "shares": pos.shares, "price": entry,
        "note": "initial buy", "stop": stop, "tp1": pos.tp1, "tp2": pos.tp2, "trail": "",
    })

    for i in range(1, len(forward)):
        r = forward.iloc[i]
        # ---- position management ----
        if pos is not None:
            # Update trailing stop
            new_trail = pos.trail
            if pd.notna(r.get("atr14")):
                new_trail = max(new_trail, float(r["close"]) - 1.5 * float(r["atr14"]))
            if pd.notna(r.get("ma50")):
                new_trail = max(new_trail, float(r["ma50"]))
            pos.trail = new_trail

            # TP1: half off
            if not pos.tp1_done and float(r["high"]) >= pos.tp1:
                half = math.floor(pos.shares / 2)
                if half > 0:
                    realized += half * (pos.tp1 - pos.entry)
                    pos.shares -= half
                    pos.tp1_done = True
                    log.append({
                        "date": r["date"].strftime("%Y-%m-%d"),
                        "action": "SELL_TP1", "shares": half, "price": pos.tp1,
                        "note": "1R", "stop": "", "tp1": "", "tp2": "", "trail": pos.trail,
                    })

            # TP2: close all
            if pos and float(r["high"]) >= pos.tp2:
                realized += pos.shares * (pos.tp2 - pos.entry)
                log.append({
                    "date": r["date"].strftime("%Y-%m-%d"),
                    "action": "SELL_TP2", "shares": pos.shares, "price": pos.tp2,
                    "note": "2R", "stop": "", "tp1": "", "tp2": "", "trail": pos.trail,
                })
                pos = None
                continue

            # Trail / MA50 break
            exit_price: float | None = None
            close = float(r["close"])
            if close < pos.trail:
                exit_price = close
            elif pd.notna(r.get("ma50")) and close < float(r["ma50"]):
                exit_price = close
            if exit_price is not None:
                realized += pos.shares * (exit_price - pos.entry)
                log.append({
                    "date": r["date"].strftime("%Y-%m-%d"),
                    "action": "SELL_EXIT", "shares": pos.shares, "price": exit_price,
                    "note": "trail_or_ma50", "stop": "", "tp1": "", "tp2": "",
                    "trail": pos.trail,
                })
                pos = None

        # ---- re-entry ----
        if pos is None and allow_reentry:
            trigger = detect_trigger_type(r, trigger_cfg)
            if trigger in {"breakout", "pullback"}:
                entry = float(r["close"])
                stop = _stop_from_row(r, atr_mult)
                if stop >= entry:
                    continue
                R = entry - stop
                pos = _OpenPosition(
                    shares=float(initial_buy["shares"]),
                    entry=entry, stop=stop,
                    tp1=entry + R, tp2=entry + 2 * R, trail=stop,
                )
                capital = pos.shares * entry
                max_capital = max(max_capital, capital)
                log.append({
                    "date": r["date"].strftime("%Y-%m-%d"),
                    "action": "BUY_SIGNAL", "shares": pos.shares, "price": entry,
                    "note": trigger, "stop": stop, "tp1": pos.tp1, "tp2": pos.tp2, "trail": "",
                })

    # Mark-to-market remaining position
    last = forward.iloc[-1]
    if pos is not None:
        unrealized = pos.shares * (float(last["close"]) - pos.entry)
        log.append({
            "date": last["date"].strftime("%Y-%m-%d"),
            "action": "MARK_OR_END", "shares": pos.shares, "price": float(last["close"]),
            "note": "end", "stop": "", "tp1": "", "tp2": "", "trail": "",
        })
    else:
        unrealized = 0.0
    total = realized + unrealized
    summary = {
        "pnl": total,
        "max_capital": max_capital,
        "return": total / max_capital if max_capital > 0 else 0.0,
    }
    return pd.DataFrame(log), summary


__all__ = [
    "PositionPnl",
    "compute_position_pnl",
    "decision_review",
    "simulate_strategy",
    "REQUIRED_COLUMNS",
]
