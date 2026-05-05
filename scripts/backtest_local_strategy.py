"""Run holding-level backtest using the production radar trigger rules.

Pulls A-share daily history via akshare, builds the indicator schema expected
by `signal_engine.backtest`, then prints a per-symbol report covering:

  - both P&L views (broker 摊薄 vs moving-average) — must agree on total
  - decision review for the user's actual trades
  - strategy simulation (initial-only / with-reentry)
  - today's verdict for the latest bar

This is the on-demand entry point. It writes the same CSV layout the prior
session produced under ``openbb_outputs/`` so existing notebooks/diffs still
work.
"""
from __future__ import annotations

import os
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"

import urllib.request
urllib.request.getproxies = lambda: {}

from pathlib import Path
import sys

import akshare as ak
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_workbench.signal_engine.backtest import (
    compute_position_pnl,
    decision_review,
    simulate_strategy,
)
from research_workbench.signal_engine.radar import (
    DEFAULT_RADAR_CONFIG,
    add_radar_features,
    detect_trigger_type,
)

OUT = ROOT / "openbb_outputs"


# ---------- data ----------

def fetch_daily(code: str, start: str, end: str) -> pd.DataFrame:
    raw = ak.stock_zh_a_hist(symbol=code, period="daily",
                             start_date=start, end_date=end, adjust="qfq")
    raw.columns = ["date", "code", "open", "close", "high", "low",
                   "volume", "amount", "amplitude", "pct_chg", "chg", "turnover"]
    raw["date"] = pd.to_datetime(raw["date"])
    for c in ("open", "close", "high", "low", "volume", "amount",
              "amplitude", "pct_chg", "chg", "turnover"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["symbol"] = code
    return raw.sort_values("date").reset_index(drop=True)


def compute_full_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the full indicator schema needed by signal_engine.backtest.

    Combines what `fetch_cn_data.py` would produce (ma50/ma200/rsi14/support/
    volume_spike_ratio) with what `radar.add_radar_features` adds
    (ma20/atr14/high_20d/highest_close_20d/drawdown_60d).
    """
    d = df.sort_values("date").copy()
    close = d["close"]

    # ma50 / ma200 / rsi14 (mirrors fetch_cn_data.py)
    d["ma50"] = close.rolling(50, min_periods=1).mean()
    d["ma200"] = close.rolling(200, min_periods=1).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - 100 / (1 + rs)

    # Support / volume — same definitions fetch_cn_data.py uses
    rolling_min = close.rolling(20, min_periods=1).min()
    d["support_level_primary"] = rolling_min
    d["support_level_secondary"] = rolling_min * 1.1
    d["volume_ma20"] = d["volume"].rolling(20, min_periods=1).mean()
    d["volume_spike_ratio"] = d["volume"] / d["volume_ma20"]

    # Now overlay radar's features (ma20 / atr14 / high_20d / drawdown_60d)
    d = add_radar_features(d)
    return d


# ---------- presentation ----------

def report(label: str, code: str, trades: list[dict], start: str, end: str):
    print(f"\n=== {label} ({code}) ===")
    raw = fetch_daily(code, start, end)
    d = compute_full_indicators(raw)
    last = d.iloc[-1]

    # P&L
    p = compute_position_pnl(trades, last_price=float(last["close"]))
    print(f"  last bar: {last['date'].strftime('%Y-%m-%d')}  close={last['close']:.2f}")
    print(f"  --- P&L ---")
    print(f"    total P&L: {p.total_pnl:+.0f}  (broker view {p.floating_pnl:+.0f}, mov-avg view {p.realized_pnl:+.0f}+{p.unrealized_pnl:+.0f})")
    print(f"    holdings: {p.holdings:.0f} shares")
    print(f"    broker (摊薄) cost: {p.broker_cost:.3f}  net invested: {p.net_invested:.0f}  floating: {p.floating_pnl_pct*100:+.2f}%")
    print(f"    moving-avg cost:    {p.avg_cost:.3f}  max cost basis: {p.max_avg_cost_basis:.0f}  ROI on max cap: {p.return_on_max_capital*100:+.2f}%")

    # Decision review (production rules)
    review = decision_review(d, trades)
    review.to_csv(OUT / f"{label}_decision_review.csv", index=False)
    accepts = review[(review["side"] == "BUY") & (review["verdict"] == "ACCEPT")]
    rejects = review[(review["side"] == "BUY") & (review["verdict"] == "REJECT")]
    print(f"  --- decisions (production trigger rule) ---")
    print(f"    BUY decisions: {len(accepts)} ACCEPT, {len(rejects)} REJECT")
    for _, r in accepts.iterrows():
        print(f"      ACCEPT {r['date']} @ {r['price']:.2f} ({r['trigger_type']})")

    # Strategy sims
    first_buy = next(t for t in trades if t["side"] == "BUY")
    initial = {"date": first_buy["date"], "price": first_buy["price"], "shares": first_buy["shares"]}
    log_init, st_init = simulate_strategy(d, initial, allow_reentry=False)
    log_re, st_re = simulate_strategy(d, initial, allow_reentry=True)
    log_init.to_csv(OUT / f"{label}_strategy_initial_only.csv", index=False)
    log_re.to_csv(OUT / f"{label}_strategy_with_reentry.csv", index=False)
    print(f"  --- strategy backtests (forced initial buy = first user BUY) ---")
    print(f"    initial-only:    pnl={st_init['pnl']:+.0f}  max_cap={st_init['max_capital']:.0f}  return={st_init['return']*100:+.2f}%")
    print(f"    with-reentry:    pnl={st_re['pnl']:+.0f}  max_cap={st_re['max_capital']:.0f}  return={st_re['return']*100:+.2f}%")

    # Today's verdict
    today_trigger = detect_trigger_type(last, DEFAULT_RADAR_CONFIG.get("triggers", {}))
    today_verdict = "ACCEPT" if today_trigger else "REJECT"
    print(f"  --- today ({last['date'].strftime('%Y-%m-%d')}) ---")
    print(f"    trigger: {today_trigger or 'none'}  →  verdict: {today_verdict}")
    if pd.notna(last.get("ma50")) and pd.notna(last.get("ma200")):
        print(f"    close={last['close']:.2f}  ma20={last['ma20']:.2f}  ma50={last['ma50']:.2f}  ma200={last['ma200']:.2f}")
        print(f"    rsi14={last.get('rsi14', float('nan')):.1f}  vol_spike={last.get('volume_spike_ratio', float('nan')):.2f}  support_primary={last.get('support_level_primary', float('nan')):.2f}")


HUATONG_TRADES = [
    {"date": "2025-09-09", "side": "BUY",  "price": 17.98, "shares": 400},
    {"date": "2025-09-11", "side": "BUY",  "price": 18.90, "shares": 400},
    {"date": "2025-09-24", "side": "SELL", "price": 21.15, "shares": 400},
    {"date": "2025-10-15", "side": "BUY",  "price": 18.89, "shares": 400},
    {"date": "2025-10-17", "side": "BUY",  "price": 18.25, "shares": 300},
    {"date": "2025-10-20", "side": "BUY",  "price": 17.34, "shares": 200},
    {"date": "2025-10-28", "side": "BUY",  "price": 17.83, "shares": 500},
    {"date": "2025-10-30", "side": "BUY",  "price": 17.31, "shares": 300},
    {"date": "2025-11-12", "side": "SELL", "price": 19.37, "shares": 700},
    {"date": "2025-11-17", "side": "BUY",  "price": 16.70, "shares": 500},
    {"date": "2025-11-17", "side": "BUY",  "price": 16.40, "shares": 600},
    {"date": "2025-12-15", "side": "BUY",  "price": 16.40, "shares": 500},
    {"date": "2026-01-05", "side": "SELL", "price": 18.10, "shares": 500},
    {"date": "2026-01-08", "side": "SELL", "price": 18.60, "shares": 500},
    {"date": "2026-01-30", "side": "SELL", "price": 20.00, "shares": 700},
    {"date": "2026-03-12", "side": "BUY",  "price": 16.89, "shares": 500},
]

PW_TRADES = [
    {"date": "2025-09-16", "side": "BUY",  "price": 18.90, "shares": 500},
    {"date": "2025-09-23", "side": "BUY",  "price": 18.70, "shares": 200},
    {"date": "2025-10-13", "side": "BUY",  "price": 17.16, "shares": 300},
    {"date": "2025-10-27", "side": "BUY",  "price": 16.88, "shares": 200},
    {"date": "2025-11-12", "side": "BUY",  "price": 15.90, "shares": 400},
    {"date": "2026-01-08", "side": "SELL", "price": 16.90, "shares": 600},
    {"date": "2026-02-06", "side": "SELL", "price": 19.57, "shares": 1000},
]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    report("Huatong",        "002602", HUATONG_TRADES, "20240101", "20260505")
    report("PerfectWorld_v2", "002624", PW_TRADES,     "20240101", "20260505")
