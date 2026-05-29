"""Entry-timing / pricing backtest: same signals, different fill tactics.

Question this answers: given the production buy signal (detect_trigger_type),
does *when / at what price* you enter materially change the outcome? We hold the
signal set fixed and only vary the fill rule, so any difference is pure
execution alpha (or the cost of being patient).

Tactics compared (fill price for a signal decided on the close of day t):

  A 收盘    : buy at close[t]                       — baseline (current system)
  B 次日开盘: buy at open[t+1]                       — eats the overnight gap
  C 限价回踩: limit at ma20[t]; fill at limit if any  — left-side, patient
              low in the next WINDOW days touches it; else MISS
  D 不追高  : if a breakout is extended (close[t] >   — right-side but capped
              high_20d[t] + K*ATR14[t]), wait for a
              pullback to that threshold within WINDOW
              days; else buy close[t]. Pullback signals
              are never "chasing", so D == A for them.

For every filled entry we record the entry price vs tactic A (in %) and the
forward return at +5/+10/+20 trading days (close-to-close). Aggregates report
fill rate, avg entry discount, mean/median forward return and win rate — split
by trigger type, because the chase problem is breakout-specific.

Run: python scripts/entry_timing_backtest.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_workbench.data_sources.indicators import add_technical_indicators
from research_workbench.data_sources.longbridge import get_provider
from research_workbench.signal_engine.radar import (
    DEFAULT_RADAR_CONFIG,
    add_radar_features,
    detect_trigger_type,
)

# ---- knobs ----
WINDOW = 5          # trading days a limit / pullback order stays live
K_ATR = 1.0         # "extended" = close above breakout level + K_ATR * ATR14
HORIZONS = (5, 10, 20)
OUT = ROOT / "outputs" / "research_data"


def load_bars(symbol: str, market: str, start: date, end: date) -> pd.DataFrame:
    raw = get_provider(market).fetch_history(symbol, start, end)
    df = add_radar_features(add_technical_indicators(raw)).reset_index(drop=True)
    return df


def _fill_C(df: pd.DataFrame, t: int) -> float | None:
    """Limit at ma20[t]; fill if any low in (t, t+WINDOW] dips to it."""
    limit = df.at[t, "ma20"]
    if pd.isna(limit):
        return None
    for j in range(t + 1, min(t + 1 + WINDOW, len(df))):
        if float(df.at[j, "low"]) <= float(limit):
            return float(limit)
    return None


def _fill_D(df: pd.DataFrame, t: int, trigger: str) -> float | None:
    """Buy close[t] unless it's an extended breakout; then wait for pullback."""
    close = float(df.at[t, "close"])
    if trigger != "breakout":
        return close
    high20 = df.at[t, "high_20d"]
    atr = df.at[t, "atr14"]
    if pd.isna(high20) or pd.isna(atr):
        return close
    threshold = float(high20) + K_ATR * float(atr)
    if close <= threshold:
        return close  # breakout not over-extended — fine to take it
    for j in range(t + 1, min(t + 1 + WINDOW, len(df))):
        if float(df.at[j, "low"]) <= threshold:
            return threshold
    return None  # stayed extended the whole window — missed


def fwd_return(df: pd.DataFrame, t: int, entry: float, h: int) -> float | None:
    j = t + h
    if j >= len(df) or entry <= 0:
        return None
    return float(df.at[j, "close"]) / entry - 1.0


def run(symbol: str, market: str, start: date, end: date) -> pd.DataFrame:
    df = load_bars(symbol, market, start, end)
    trig_cfg = DEFAULT_RADAR_CONFIG.get("triggers", {})
    rows: list[dict] = []
    for t in range(len(df) - 1):  # need at least t+1 for tactic B
        trigger = detect_trigger_type(df.iloc[t], trig_cfg)
        if trigger not in {"breakout", "pullback"}:
            continue
        ma200 = df.at[t, "ma200"]
        regime = "uptrend" if pd.notna(ma200) and float(df.at[t, "close"]) >= float(ma200) else "downtrend"
        fills = {
            "A": float(df.at[t, "close"]),
            "B": float(df.at[t + 1, "open"]),
            "C": _fill_C(df, t),
            "D": _fill_D(df, t, trigger),
        }
        base = fills["A"]
        for tactic, entry in fills.items():
            rec = {
                "market": market,
                "symbol": symbol,
                "date": df.at[t, "date"].strftime("%Y-%m-%d"),
                "year": df.at[t, "date"].year,
                "regime": regime,
                "trigger": trigger,
                "tactic": tactic,
                "filled": entry is not None,
                "entry": entry,
                "entry_vs_A_pct": (entry / base - 1.0) * 100 if entry else None,
            }
            for h in HORIZONS:
                rec[f"ret_{h}d"] = fwd_return(df, t, entry, h) if entry else None
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, by_trigger: bool = False) -> pd.DataFrame:
    group_cols = (["trigger"] if by_trigger else []) + ["tactic"]
    out = []
    for keys, g in trades.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(g)
        filled = g[g["filled"]]
        rec = dict(zip(group_cols, keys))
        rec["signals"] = n
        rec["fill_rate"] = len(filled) / n if n else np.nan
        rec["avg_entry_vs_A_%"] = filled["entry_vs_A_pct"].mean()
        for h in HORIZONS:
            col = f"ret_{h}d"
            v = filled[col].dropna()
            rec[f"mean_{h}d_%"] = v.mean() * 100 if len(v) else np.nan
            if h == 20 and len(v):
                rec["win_20d_%"] = (v > 0).mean() * 100
        out.append(rec)
    return pd.DataFrame(out)


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:.2f}")


# Strategy is segmented by market — A-share (T+1, different microstructure) and
# HK behave differently, so conclusions and eventually entry rules are per-market.
MARKETS: dict[str, list[tuple[str, str]]] = {
    "HK": [
        ("700.HK", "腾讯"), ("9988.HK", "阿里巴巴"), ("9992.HK", "泡泡玛特"),
        ("1024.HK", "快手"), ("9999.HK", "网易"), ("9961.HK", "携程"),
        ("1810.HK", "小米"), ("3690.HK", "美团"),
    ],
}


def _summary_by(trades: pd.DataFrame, *extra: str) -> pd.DataFrame:
    out = []
    for keys, g in trades.groupby(list(extra) + ["tactic"]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(g)
        filled = g[g["filled"]]
        rec = dict(zip(list(extra) + ["tactic"], keys))
        rec["signals"] = n
        rec["fill_%"] = len(filled) / n * 100 if n else np.nan
        rec["entry_vs_A_%"] = filled["entry_vs_A_pct"].mean()
        v = filled["ret_20d"].dropna()
        rec["mean_20d_%"] = v.mean() * 100 if len(v) else np.nan
        rec["win_20d_%"] = (v > 0).mean() * 100 if len(v) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def report_market(market: str, trades: pd.DataFrame) -> None:
    n = len(trades) // 4
    mix = {k: v // 4 for k, v in trades["trigger"].value_counts().to_dict().items()}
    print(f"\n{'='*64}\n### 市场 {market}  —  {n} signals, mix={mix}  (WINDOW={WINDOW}d, K_ATR={K_ATR})\n{'='*64}")
    print("\n-- 总体 --")
    print(_fmt(_summary_by(trades)))
    print("\n-- 按行情阶段(信号日 close vs ma200) --")
    print(_fmt(_summary_by(trades, "regime")))
    print("\n-- 按年份 --")
    print(_fmt(_summary_by(trades, "year")))
    print("\n-- 行情阶段 × 触发类型 --")
    print(_fmt(_summary_by(trades, "regime", "trigger")))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    START, END = date(2022, 1, 1), date.today()

    all_parts = []
    for market, basket in MARKETS.items():
        print(f"拉取 {market} 篮子 {len(basket)} 只  {START} → {END} ...")
        parts = []
        for sym, name in basket:
            t = run(sym, market, START, END)
            parts.append(t)
            print(f"  {name} {sym}: {len(t)//4} signals")
        mkt_trades = pd.concat(parts, ignore_index=True)
        all_parts.append(mkt_trades)
        report_market(market, mkt_trades)

    trades = pd.concat(all_parts, ignore_index=True)
    trades.to_csv(OUT / "entry_timing_basket_trades.csv", index=False)
    print(f"\n明细 CSV: {OUT / 'entry_timing_basket_trades.csv'}")
