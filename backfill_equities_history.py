#!/usr/bin/env python3
"""
Backfill 3-year historical technical indicators for equities and upload to Supabase.
Steps:
1. Pull daily price data from yfinance.
2. Compute MA50/MA200, RSI, 90d change, recent supports, etc.
3. Save intermediate CSV locally for inspection.
4. Upload to Supabase `equity_metrics_history`, chunking with retry to avoid HTTP issues.
"""
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client

SUPABASE_URL = "https://wpyrevceqirzpwcpulqz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndweXJldmNlcWlyenB3Y3B1bHF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMzODUzOTEsImV4cCI6MjA3ODk2MTM5MX0.vY-lSpINIwDc80Caq7tX6iQ_zcBaKDflO5AfV79-tZA"

HISTORY_DAYS = 365 * 3
RECENT_DAYS = 3
OUTPUT_CSV = Path("openbb_outputs/equity_history_snapshot.csv")
BATCH_SIZE = 200
RETRY_LIMIT = 3

client = create_client(SUPABASE_URL, SUPABASE_KEY)

cfg_path = Path("equity_config.json")
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    STOCKS = [item["symbol"] for item in cfg.get("equities", [])]
else:
    STOCKS = ["MSFT", "AAPL"]


def chunked(rows: List[Dict[str, object]], size: int = BATCH_SIZE):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_supports(series: pd.Series, window: int = 7, top_n: int = 2) -> List[float]:
    supports: List[float] = []
    for idx in range(window, len(series) - window):
        val = series.iloc[idx]
        local = series.iloc[idx - window : idx + window + 1]
        if val == local.min():
            supports.append(float(val))
    return supports[-top_n:]


def process_symbol(symbol: str) -> List[Dict[str, object]]:
    print(f"[INFO] fetching {symbol}")
    try:
        data = yf.download(
            symbol,
            period="3y",
            interval="1d",
            progress=False,
            group_by="column",
        )
    except Exception as exc:
        print(f"[WARN] yfinance download failed for {symbol}: {exc}")
        return []
    if data.empty:
        print(f"[WARN] no data for {symbol}")
        return []
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.droplevel(1)
    data = data.dropna()
    data = data.tail(HISTORY_DAYS)
    if data.empty:
        return []

    close = data["Close"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi = compute_rsi(close)
    pct_90 = close.pct_change(90) * 100

    records: List[Dict[str, object]] = []
    for idx in range(len(data)):
        subset = data.iloc[: idx + 1]
        date = subset.index[-1]
        supports = detect_supports(subset["Close"])
        records.append(
            {
                "symbol": symbol,
                "as_of_date": date.date().isoformat(),
                "ma50": float(ma50.iloc[idx]) if not np.isnan(ma50.iloc[idx]) else None,
                "ma200": float(ma200.iloc[idx]) if not np.isnan(ma200.iloc[idx]) else None,
                "rsi14": float(rsi.iloc[idx]) if not np.isnan(rsi.iloc[idx]) else None,
                "pct_change": float(pct_90.iloc[idx]) if not np.isnan(pct_90.iloc[idx]) else None,
                "support_level_primary": supports[-1] if supports else None,
                "support_level_secondary": supports[-2] if len(supports) > 1 else None,
            }
        )
    return records


def save_csv(records: List[Dict[str, object]]):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[INFO] saved snapshot to {OUTPUT_CSV}")


def upload_records(records: List[Dict[str, object]]):
    # Delete recent range only
    recent_cutoff = (datetime.utcnow().date() - pd.Timedelta(days=RECENT_DAYS)).isoformat()
    print(f"[INFO] cleaning Supabase equity_metrics_history for as_of_date >= {recent_cutoff}")
    client.table("equity_metrics_history").delete().gte("as_of_date", recent_cutoff).execute()
    for batch in chunked(records):
        attempt = 0
        while True:
            try:
                client.table("equity_metrics_history").insert(batch).execute()
                break
            except Exception as exc:
                attempt += 1
                if attempt >= RETRY_LIMIT:
                    raise
                wait = 5 * attempt
                print(f"[WARN] insert failed ({exc}), retrying in {wait}s ...")
                time.sleep(wait)


def main():
    all_records: List[Dict[str, object]] = []
    for symbol in STOCKS:
        all_records.extend(process_symbol(symbol))
    if not all_records:
        print("[ERROR] no records generated")
        return
    print(f"[INFO] prepared {len(all_records)} rows")
    save_csv(all_records)
    upload_records(all_records)
    print("[OK] equity history backfill completed")


if __name__ == "__main__":
    main()
