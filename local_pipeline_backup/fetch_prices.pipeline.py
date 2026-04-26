#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch daily close prices via yfinance for META (US), 1810.HK (Xiaomi), 002624.SZ (Perfect World).
Outputs a per-ticker CSV and a merged CSV.
Data source: Yahoo Finance via yfinance (see docs: https://ranaroussi.github.io/yfinance/)
"""
import os, sys, json
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

HERE = os.path.abspath(os.path.dirname(__file__))

def load_cfg():
    with open(os.path.join(HERE, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    cfg = load_cfg()
    out_dir = os.path.join(HERE, cfg.get("out_dir", "outputs"))
    os.makedirs(out_dir, exist_ok=True)
    per_price_dir = os.path.join(out_dir, "prices")
    os.makedirs(per_price_dir, exist_ok=True)

    end = datetime.now(timezone.utc)
    lookback_years = cfg.get("price_lookback_years", 2)
    if lookback_years <= 0:
        print("[WARN] non-positive lookback configured, defaulting to 2 years")
        lookback_years = 2
    start = end - relativedelta(years=lookback_years)

    symbols = []
    for market in cfg.get("tickers", {}).values():
        if isinstance(market, dict):
            symbols.extend(market.keys())
    symbols = sorted(set(symbols))
    if not symbols:
        print("[WARN] no tickers configured; using defaults META/1810.HK/002624.SZ")
        symbols = ["META", "1810.HK", "002624.SZ"]

    frames = []
    for sym in symbols:
        print(f"[prices] downloading {sym} ...")
        df = yf.download(sym, start=start.date(), end=end.date(), progress=False)
        if df.empty:
            print(f"[WARN] no data for {sym}")
            continue
        out = df[["Close"]].rename(columns={"Close": "close"})
        out.index.name = "date"
        out["symbol"] = sym
        out.reset_index().to_csv(os.path.join(per_price_dir, f"{sym.replace('.', '_')}.csv"), index=False)
        frames.append(out.reset_index())

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(os.path.join(out_dir, "prices_merged.csv"), index=False)
        print(f"[OK] wrote {os.path.join(out_dir, 'prices_merged.csv')}")
    else:
        print("[WARN] no price frames to merge")

if __name__ == "__main__":
    main()
