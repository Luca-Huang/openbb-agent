import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from supabase import create_client

import fetch_crypto_supports as fcs

DEFAULT_SUPABASE_URL = "https://wpyrevceqirzpwcpulqz.supabase.co"
DEFAULT_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndweXJldmNlcWlyenB3Y3B1bHF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMzODUzOTEsImV4cCI6MjA3ODk2MTM5MX0.vY-lSpINIwDc80Caq7tX6iQ_zcBaKDflO5AfV79-tZA"

SUPABASE_URL = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL)
SUPABASE_KEY = os.getenv("SUPABASE_KEY", DEFAULT_SUPABASE_KEY)
LOOKBACK_DAYS = int(os.getenv("CRYPTO_LOOKBACK_DAYS", "365"))

client = create_client(SUPABASE_URL, SUPABASE_KEY)
config = fcs.load_config()
coingecko_env = os.getenv("COINGECKO_API_KEY")
if coingecko_env:
    config["coingecko_api_key"] = coingecko_env
volume_bins = config.get("volume_bins", 20)
swing_window = config.get("swing_window_days", 120)
vs_currency = config.get("vs_currency", "usd")

LLAMA_API = os.getenv("LLAMA_API_BASE", "https://api.llama.fi")


def chunked(items, size=200):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_chart_series(chain_name: Optional[str]) -> pd.Series:
    if not chain_name:
        return pd.Series(dtype=float)
    url = f"{LLAMA_API}/charts/{chain_name}"
    try:
        res = requests.get(url, timeout=40)
        res.raise_for_status()
        data = res.json()
        if not data:
            return pd.Series(dtype=float)
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df = df.set_index("date")
        return df["totalLiquidityUSD"].astype(float)
    except Exception as err:
        print(f"[WARN] fetch_chart_series failed for {chain_name}: {err}")
        return pd.Series(dtype=float)


def fetch_overview_series(kind: str, slug: Optional[str]) -> pd.Series:
    if not slug:
        return pd.Series(dtype=float)
    url = f"{LLAMA_API}/overview/{kind}/{slug}"
    params = {"excludeTotalDataChart": "false", "excludeProtocolChart": "true"}
    try:
        res = requests.get(url, params=params, timeout=40)
        res.raise_for_status()
        body = res.json()
        chart = body.get("totalDataChart", [])
        if not chart:
            return pd.Series(dtype=float)
        df = pd.DataFrame(chart, columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df = df.set_index("date")
        return df["value"].astype(float)
    except Exception as err:
        print(f"[WARN] fetch_overview_series failed for {kind} {slug}: {err}")
        return pd.Series(dtype=float)


CHAIN_META = fcs.CHAIN_METADATA if hasattr(fcs, "CHAIN_METADATA") else {
    "BTC": {"chart": "Bitcoin", "overview": "bitcoin"},
    "ETH": {"chart": "Ethereum", "overview": "ethereum"},
    "SOL": {"chart": "Solana", "overview": "solana"},
    "DOGE": {"chart": None, "overview": "dogechain"},
}


def align_series(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    if series.empty:
        return pd.Series(index=index, dtype=float)
    series = series.sort_index()
    reindexed = series.reindex(index, method="ffill")
    return reindexed


def compute_daily_metrics(symbol: str, df: pd.DataFrame, llama_map: Dict[str, pd.Series]) -> List[Dict[str, object]]:
    df = df.sort_index()
    price = df["price"]
    volumes = df["volume"]
    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()
    rsi_series = fcs.compute_rsi(price)
    pct_7d = price.pct_change(7) * 100
    pct_30d = price.pct_change(30) * 100
    dist_ma50 = (price - ma50) / ma50 * 100
    dist_ma200 = (price - ma200) / ma200 * 100
    volume_avg_30 = volumes.rolling(30).mean()

    llama_chain = align_series(llama_map.get("chain", pd.Series(dtype=float)), df.index)
    llama_dex = align_series(llama_map.get("dex", pd.Series(dtype=float)), df.index)
    llama_fees = align_series(llama_map.get("fees", pd.Series(dtype=float)), df.index)

    records = []
    for date in df.index:
        subset = df.loc[:date]
        price_today = price.loc[date]
        ma50_today = ma50.loc[date]
        ma200_today = ma200.loc[date]
        rsi_today = rsi_series.loc[date]
        pct7 = pct_7d.loc[date]
        pct30 = pct_30d.loc[date]
        dist50 = dist_ma50.loc[date]
        dist200 = dist_ma200.loc[date]
        vol_avg = volume_avg_30.loc[date]

        supports = fcs.detect_recent_supports(subset["price"], window=7, top_n=3)
        fibs, swing_low, swing_high = fcs.fibonacci_levels(subset.reset_index(), swing_window)
        hvn_nodes = fcs.compute_volume_nodes(subset.reset_index(), bins=volume_bins, top_n=3)

        record = {
            "symbol": symbol,
            "as_of_date": date.date().isoformat(),
            "last_price": float(price_today) if price_today == price_today else None,
            "ma50": float(ma50_today) if ma50_today == ma50_today else None,
            "ma200": float(ma200_today) if ma200_today == ma200_today else None,
            "ma_trend": "bullish" if ma200_today and price_today and price_today > ma200_today else "bearish",
            "rsi14": float(rsi_today) if rsi_today == rsi_today else None,
            "rsi_divergence": False,
            "recent_supports": supports,
            "support_strength": len(supports),
            "fib_38_2": fibs.get("38.2%"),
            "fib_50": fibs.get("50%"),
            "fib_61_8": fibs.get("61.8%"),
            "swing_low": swing_low,
            "swing_high": swing_high,
            "swing_range": (swing_high - swing_low) if swing_high and swing_low else None,
            "pct_change_7d": float(pct7) if pct7 == pct7 else None,
            "pct_change_30d": float(pct30) if pct30 == pct30 else None,
            "distance_ma50_pct": float(dist50) if dist50 == dist50 else None,
            "distance_ma200_pct": float(dist200) if dist200 == dist200 else None,
            "volume_avg_30d": float(vol_avg) if vol_avg == vol_avg else None,
            "llama_chain": llama_map.get("name"),
            "llama_chain_tvl": float(llama_chain.loc[date]) if date in llama_chain.index and llama_chain.loc[date] == llama_chain.loc[date] else None,
            "llama_dex_volume_24h": float(llama_dex.loc[date]) if date in llama_dex.index and llama_dex.loc[date] == llama_dex.loc[date] else None,
            "llama_fees_24h": float(llama_fees.loc[date]) if date in llama_fees.index and llama_fees.loc[date] == llama_fees.loc[date] else None,
            "llama_stablecoin_supply": None,
            "retrieved_at": datetime.combine(date.date(), datetime.min.time(), tzinfo=timezone.utc).isoformat(),
        }
        records.append(record)
    return records


def backfill_coin(coin_cfg: Dict[str, str]) -> List[Dict[str, object]]:
    coingecko_id = coin_cfg["id"]
    symbol = coin_cfg["symbol"]
    try:
        df, source = fcs.fetch_market_chart(
            coingecko_id,
            vs_currency,
            LOOKBACK_DAYS,
            api_key=config.get("coingecko_api_key"),
            binance_symbol=coin_cfg.get("binance_symbol"),
            yf_symbol=coin_cfg.get("yfinance_symbol"),
        )
    except Exception as err:
        print(f"[ERROR] failed to fetch prices for {symbol}: {err}")
        return []
    if df.empty:
        return []
    df = df.set_index("date").sort_index()
    meta = CHAIN_META.get(symbol.upper(), {})
    llama_map = {
        "chain": fetch_chart_series(meta.get("chart")),
        "dex": fetch_overview_series("dexs", meta.get("overview")),
        "fees": fetch_overview_series("fees", meta.get("overview")),
        "name": meta.get("chart") or meta.get("overview"),
    }
    return compute_daily_metrics(symbol.upper(), df, llama_map)


all_records: List[Dict[str, object]] = []
for coin in config.get("coins", []):
    symbol = coin["symbol"]
    print(f"[INFO] processing {symbol} ...")
    recs = backfill_coin(coin)
    all_records.extend(recs)

if not all_records:
    raise SystemExit("No records to upload")

print(f"[INFO] Prepared {len(all_records)} rows. Uploading ...")
client.table("crypto_supports_history").delete().neq("id", 0).execute()
for batch in chunked(all_records, 200):
    client.table("crypto_supports_history").insert(batch).execute()

print("[OK] Crypto history backfill completed")
