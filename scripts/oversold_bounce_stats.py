"""Measure base rate of bounces after oversold conditions.

For each symbol, find every bar where:
  - rsi14 < 30 (deeply oversold)
  - close < ma200 (in a bigger downtrend)
Then compute forward returns at +5 / +10 / +20 / +60 trading days.
"""
from __future__ import annotations

import os
for _k in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","all_proxy","ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
import urllib.request
urllib.request.getproxies = lambda: {}

from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_local_strategy import fetch_daily, compute_indicators

CASES = [
    ("002624", "PerfectWorld"),
    ("002602", "Huatong"),
    ("002241", "GoerTek"),
    ("002343", "CiwenMedia"),
]

def measure(code: str) -> pd.DataFrame:
    raw = fetch_daily(code, "20220101", "20260505")
    d = compute_indicators(raw)
    rows = []
    for i in range(len(d)):
        r = d.iloc[i]
        if pd.isna(r["rsi14"]) or pd.isna(r["ma200"]):
            continue
        if r["rsi14"] < 30 and r["close"] < r["ma200"]:
            for h in (5, 10, 20, 60):
                j = i + h
                if j < len(d):
                    fwd = (d.iloc[j]["close"] - r["close"]) / r["close"]
                    rows.append({"date": r["date"], "horizon": h, "fwd_ret": fwd})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    for code, name in CASES:
        df = measure(code)
        if df.empty:
            print(f"{name}: no oversold instances found")
            continue
        print(f"\n=== {name} ({code}) — RSI<30 & close<MA200 ===")
        n_signals = df["date"].nunique()
        print(f"  signals: {n_signals}")
        for h in (5, 10, 20, 60):
            sub = df[df["horizon"] == h]["fwd_ret"]
            if len(sub) == 0: continue
            win = (sub > 0).mean() * 100
            avg = sub.mean() * 100
            med = sub.median() * 100
            p25, p75 = sub.quantile(0.25)*100, sub.quantile(0.75)*100
            mn, mx = sub.min()*100, sub.max()*100
            print(f"  +{h:>2}d:  win%={win:5.1f}  mean={avg:+6.2f}%  median={med:+6.2f}%  "
                  f"P25={p25:+6.2f}%  P75={p75:+6.2f}%  min={mn:+6.2f}%  max={mx:+6.2f}%")
