#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify price closings (optional) and merge fundamentals for reporting.
- Loads outputs from fetch_prices.py and fetch_fundamentals.py
- Produces a compact merged CSV.
"""
import os, json
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))

def load_cfg():
    with open(os.path.join(HERE, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    cfg = load_cfg()
    out_dir = os.path.join(HERE, cfg.get("out_dir", "outputs"))
    price_csv = os.path.join(out_dir, "prices_merged.csv")
    fund_dir = os.path.join(out_dir, "fundamentals")
    combined_csv = os.path.join(out_dir, "report_last2y.csv")

    frames = []
    for name in ["META_quarterly_last8.csv", "1810_HK_recent.csv", "002624_SZ_quarterly_last8.csv"]:
        p = os.path.join(fund_dir, name)
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                print(f"[WARN] {name} is empty, skipping")
                continue
            df["source_file"] = name
            frames.append(df)
    if frames:
        fund = pd.concat(frames, ignore_index=True)
    else:
        fund = pd.DataFrame()

    fund.to_csv(combined_csv, index=False)
    print("[OK] wrote", combined_csv)

if __name__ == "__main__":
    main()
