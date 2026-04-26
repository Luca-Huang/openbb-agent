#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch last 8 quarters (~2y) fundamentals for:
- META (US): SEC EDGAR companyfacts (us-gaap)
- 1810.HK (Xiaomi): AKShare HK interfaces (fallback to HKEXnews links if API not available)
- 002624.SZ (Perfect World): AKShare (Sina/Eastmoney)

Outputs CSVs under outputs/fundamentals/*.csv

Sources:
- SEC EDGAR XBRL APIs (companyfacts): https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- company_tickers.json mapping: https://www.sec.gov/about/webmaster-frequently-asked-questions (company_tickers.json)
- AKShare docs & data dictionary: https://akshare.akfamily.xyz/data/index.html
- AKShare financial report (Sina) function history & changelog: https://akshare.akfamily.xyz/changelog.html
"""
import os
import json
from typing import Dict, Any, List
from urllib.parse import urlparse

import pandas as pd
import requests

HERE = os.path.abspath(os.path.dirname(__file__))


def load_cfg():
    with open(os.path.join(HERE, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------
# US: SEC companyfacts
# ------------------------
SEC_BASE = "https://data.sec.gov/api"


def sec_headers(ua: str, host: str) -> Dict[str, str]:
    return {
        "User-Agent": ua or "yourname youremail@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }


def sec_get(url: str, ua: str, timeout: int = 20) -> requests.Response:
    parsed = urlparse(url)
    headers = sec_headers(ua, parsed.netloc)
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def get_cik_from_sec(ticker: str, ua: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = sec_get(url, ua=ua, timeout=20).json()
    for _, row in data.items():
        if row.get("ticker", "").upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for {ticker}")


def sec_companyfacts(cik: str, ua: str) -> Dict[str, Any]:
    url = f"{SEC_BASE}/xbrl/companyfacts/CIK{cik}.json"
    return sec_get(url, ua=ua, timeout=20).json()


def extract_recent_quarters(series: List[Dict[str, Any]], n: int = 8, unit_preference: str = "USD") -> pd.DataFrame:
    df = pd.DataFrame(series)
    if df.empty:
        return pd.DataFrame(columns=["period_end", "value", "uom", "fy", "fp", "filed", "form"])
    df = df[df["fp"].isin(["Q1", "Q2", "Q3", "Q4"])]
    if "uom" in df.columns and unit_preference is not None:
        uom_series = df["uom"].astype(str)
        mask = uom_series == unit_preference
        if not mask.any():
            mask = uom_series.str.startswith(unit_preference)
        if mask.any():
            df = df[mask]
    df["period_end"] = pd.to_datetime(df["end"])
    df = df.sort_values("period_end").tail(n)
    cols = ["period_end", "val", "uom", "fy", "fp", "filed", "form"]
    available = [c for c in cols if c in df.columns]
    if "val" not in available:
        raise KeyError("SEC data missing 'val' field")
    out = df[available].rename(columns={"val": "value"})
    return out


def fetch_meta_sec_last8(ua: str, quarters: int) -> pd.DataFrame:
    cik = get_cik_from_sec("META", ua)
    facts = sec_companyfacts(cik, ua)
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    def pull(tag: str) -> List[Dict[str, Any]]:
        return us_gaap.get(tag, {}).get("units", {}).get("USD", [])

    eps_df = extract_recent_quarters(pull("EarningsPerShareDiluted"), quarters)
    rev_df = extract_recent_quarters(pull("RevenueFromContractWithCustomerExcludingAssessedTax"), quarters)
    ni_df = extract_recent_quarters(pull("NetIncomeLoss"), quarters)
    ocf_df = extract_recent_quarters(pull("NetCashProvidedByUsedInOperatingActivities"), quarters)
    out = (
        eps_df.merge(rev_df, on="period_end", how="outer", suffixes=("_eps", "_rev"))
        .merge(ni_df, on="period_end", how="outer")
        .merge(ocf_df, on="period_end", how="outer", suffixes=("_ni", "_ocf"))
    )
    out = out.rename(
        columns={
            "value_eps": "eps_diluted",
            "value_rev": "revenue",
            "value": "net_income",
            "value_ocf": "operating_cf",
        }
    )
    out["symbol"] = "META"
    return out.sort_values("period_end")


# ------------------------
# HK & CN via AKShare
# ------------------------
def try_import_ak():
    try:
        import akshare as ak

        return ak
    except Exception as exc:
        print("[WARN] AKShare import failed. Please: pip install akshare --upgrade")
        print("       underlying error:", exc)
        return None


def fetch_cn_a_three_statements_sz(stock_code_with_prefix: str, quarters: int) -> pd.DataFrame:
    ak = try_import_ak()
    if ak is None:
        return pd.DataFrame()
    try:
        prof = ak.stock_financial_report_sina(stock=stock_code_with_prefix, symbol="利润表")
    except Exception as exc:
        print("[WARN] stock_financial_report_sina failed:", exc)
        return pd.DataFrame()
    if prof is None or prof.empty:
        print("[WARN] empty dataframe returned for", stock_code_with_prefix)
        return pd.DataFrame()
    date_col = next((c for c in ["报告日期", "公布日期", "披露日期"] if c in prof.columns), None)
    if date_col is None:
        print("[WARN] 未找到报告日期列, 返回空结果")
        return pd.DataFrame()
    prof["period_end"] = pd.to_datetime(prof[date_col])
    keep = prof.sort_values("period_end").tail(quarters)
    cols_map = {
        "基本每股收益(元)": "eps_basic",
        "稀释每股收益(元)": "eps_diluted",
        "营业总收入(万元)": "revenue_10k_cny",
        "净利润(万元)": "net_income_10k_cny",
    }
    renamed = keep.rename(columns=cols_map)
    selected_cols = ["period_end"]
    for raw, alias in cols_map.items():
        if raw in keep.columns and alias not in selected_cols:
            selected_cols.append(alias)
    if len(selected_cols) == 1:
        print("[WARN] 财报表中缺失目标字段")
    out = renamed[selected_cols]
    out["symbol"] = "002624.SZ"
    return out


def fetch_hk_financials_ak(hk_code_no_leading: str, quarters: int) -> pd.DataFrame:
    ak = try_import_ak()
    if ak is None:
        return pd.DataFrame()
    funcs = [
        "stock_hk_financial_summary_em",
        "stock_hk_report_em",
    ]
    date_candidates = ["报告期", "截止日期", "date", "报告日期", "截止日"]
    metric_map = {
        "每股收益": "eps",
        "基本每股收益": "eps_basic",
        "稀释每股收益": "eps_diluted",
        "营业收入": "revenue",
        "收入": "revenue",
        "营收": "revenue",
        "净利润": "net_income",
    }
    for fn in funcs:
        if not hasattr(ak, fn):
            continue
        try:
            df = getattr(ak, fn)(symbol=hk_code_no_leading)
        except Exception as exc:
            print(f"[WARN] {fn} failed:", exc)
            continue
        if df is None or df.empty:
            continue
        date_col = next((c for c in date_candidates if c in df.columns), None)
        if date_col is None:
            print(f"[WARN] {fn} 未找到日期列")
            continue
        df = df.copy()
        df["period_end"] = pd.to_datetime(df[date_col])
        df = df.sort_values("period_end").tail(quarters)
        renamed = df.rename(columns=metric_map)
        selected_cols = ["period_end"]
        for raw, alias in metric_map.items():
            if raw in df.columns and alias not in selected_cols:
                selected_cols.append(alias)
        if len(selected_cols) == 1:
            print(f"[WARN] {fn} 缺少核心字段, 返回日期列表")
        out = renamed[selected_cols]
        out["symbol"] = "1810.HK"
        return out
    return pd.DataFrame()


def main():
    cfg = load_cfg()
    out_dir = os.path.join(HERE, cfg.get("out_dir", "outputs"))
    os.makedirs(out_dir, exist_ok=True)
    fund_dir = os.path.join(out_dir, "fundamentals")
    os.makedirs(fund_dir, exist_ok=True)

    quarters = int(cfg.get("fundamentals_quarters", 8))

    ua = cfg["tickers"]["US"]["META"].get("sec_user_agent") or "your name your@email"
    print("[US] Fetching META (SEC companyfacts) ...")
    us_df = fetch_meta_sec_last8(ua=ua, quarters=quarters)
    us_path = os.path.join(fund_dir, "META_quarterly_last8.csv")
    us_df.to_csv(us_path, index=False)
    print("[OK] wrote", us_path)

    print("[CN] Fetching 002624.SZ (AKShare/Sina) ...")
    cn_df = fetch_cn_a_three_statements_sz(cfg["tickers"]["CN"]["002624.SZ"]["ak_code"], quarters=quarters)
    cn_path = os.path.join(fund_dir, "002624_SZ_quarterly_last8.csv")
    cn_df.to_csv(cn_path, index=False)
    print("[OK] wrote", cn_path)

    print("[HK] Fetching 1810.HK (AKShare HK; fallback empty) ...")
    hk_df = fetch_hk_financials_ak(cfg["tickers"]["HK"]["1810.HK"]["hk_code"], quarters=quarters)
    hk_path = os.path.join(fund_dir, "1810_HK_recent.csv")
    hk_df.to_csv(hk_path, index=False)
    print("[OK] wrote", hk_path)
    if hk_df.empty:
        print("[NOTE] HK route via AKShare not available in your environment; consider parsing HKEXnews disclosures.")


if __name__ == "__main__":
    main()
