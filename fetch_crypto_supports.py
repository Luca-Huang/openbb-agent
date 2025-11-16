#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch crypto prices (CoinGecko), optional Glassnode on-chain metrics and derivatives sentiment
signals to implement the“三维度确认法” described by the user.

Outputs:
    openbb_outputs/crypto/<symbol>_support_map.json  - rich structured snapshot per资产
    openbb_outputs/crypto/crypto_support_dashboard.csv - 可直接在 Streamlit / Excel 中查看的表格
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import os

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "crypto_config.json"
OUTPUT_DIR = ROOT / "openbb_outputs" / "crypto"
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_KLINES_API = "https://api.binance.com/api/v3/klines"
FEAR_GREED_API = "https://api.alternative.me/fng/"
BINANCE_FUNDING_API = "https://fapi.binance.com/fapi/v1/fundingRate"
LLAMA_API = "https://api.llama.fi"
LLAMA_STABLE_API = "https://stablecoins.llama.fi"

CHAIN_METADATA = {
    "BTC": {"chart": "Bitcoin", "overview": "bitcoin", "stable_gecko": "bitcoin"},
    "ETH": {"chart": "Ethereum", "overview": "ethereum", "stable_gecko": "ethereum"},
    "SOL": {"chart": "Solana", "overview": "solana", "stable_gecko": "solana"},
    "DOGE": {"chart": None, "overview": "dogechain", "stable_gecko": "dogecoin"},
}


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def fetch_binance_klines(symbol: str, days: int) -> Optional[pd.DataFrame]:
    interval = "1d"
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days)
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": min(days + 10, 1000),
    }
    try:
        resp = requests.get(BINANCE_KLINES_API, params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] Binance klines failed for {symbol}: {exc}")
        return None
    rows = []
    for entry in resp.json():
        rows.append(
            {
                "date": pd.to_datetime(entry[0], unit="ms"),
                "price": float(entry[4]),  # 收盘价
                "volume": float(entry[5]),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def fetch_yfinance_history(symbol: str, days: int) -> Optional[pd.DataFrame]:
    if not symbol:
        return None
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=days + 5)).date()
    try:
        data = yf.download(symbol, start=start.isoformat(), end=None, interval="1d", progress=False)
    except Exception as exc:
        print(f"[WARN] yfinance download failed for {symbol}: {exc}")
        return None
    if data.empty:
        return None
    data = data.reset_index()
    data["date"] = pd.to_datetime(data["Date"])
    data = data.rename(columns={"Close": "price", "Volume": "volume"})
    return data[["date", "price", "volume"]]


def fetch_market_chart(
    coin_id: str,
    vs_currency: str,
    days: int,
    api_key: Optional[str] = None,
    binance_symbol: Optional[str] = None,
    yf_symbol: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    if binance_symbol:
        df = fetch_binance_klines(binance_symbol, days)
        if df is not None and not df.empty:
            return df, "binance"
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    headers = {"accept": "application/json"}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
        headers["x-cg-pro-api-key"] = api_key
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=40)
        resp.raise_for_status()
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            detail = exc.response.text[:200]
        print(f"[WARN] CoinGecko request failed for {coin_id}: {exc} {detail}")
    else:
        payload = resp.json()
        prices = pd.DataFrame(payload.get("prices", []), columns=["ts", "price"])
        volumes = pd.DataFrame(payload.get("total_volumes", []), columns=["ts", "volume"])
        if not prices.empty and not volumes.empty:
            df = prices.merge(volumes, on="ts")
            df["date"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.sort_values("date")
            return df[["date", "price", "volume"]], "coingecko"
    yf_symbol = yf_symbol or f"{coin_id.upper()}-USD"
    df = fetch_yfinance_history(yf_symbol, days)
    if df is not None and not df.empty:
        return df, "yfinance"
    raise RuntimeError(f"Unable to fetch market data for {coin_id}")


def fetch_llama_chain_chart(chain_name: str) -> Optional[float]:
    if not chain_name:
        return None
    try:
        resp = requests.get(f"{LLAMA_API}/charts/{chain_name}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[-1]["totalLiquidityUSD"])
    except Exception as exc:
        print(f"[WARN] Llama chart fetch failed for {chain_name}: {exc}")
    return None


def fetch_llama_overview(kind: str, slug: str) -> Optional[float]:
    if not slug:
        return None
    url = f"{LLAMA_API}/overview/{kind}/{slug}"
    params = {"excludeTotalDataChart": "false", "excludeProtocolChart": "true"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        chart = body.get("totalDataChart", [])
        if chart:
            return float(chart[-1][1])
    except Exception as exc:
        print(f"[WARN] Llama overview({kind}) failed for {slug}: {exc}")
    return None


def fetch_stablecoin_supply_map() -> Dict[str, float]:
    url = f"{LLAMA_STABLE_API}/stablecoinchains"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        supply_map = {}
        for entry in data:
            gecko_id = entry.get("gecko_id")
            circulating = entry.get("totalCirculatingUSD", {}).get("peggedUSD")
            if gecko_id and circulating is not None:
                supply_map[gecko_id.lower()] = float(circulating)
        return supply_map
    except Exception as exc:
        print("[WARN] stablecoin supply fetch failed:", exc)
        return {}


def compose_llama_metrics(symbol: str, stable_supply: Dict[str, float]) -> Dict[str, Optional[float]]:
    meta = CHAIN_METADATA.get(symbol.upper(), {})
    chart_name = meta.get("chart")
    overview_slug = meta.get("overview")
    stable_key = meta.get("stable_gecko")
    tvl = fetch_llama_chain_chart(chart_name) if chart_name else None
    dex_volume = fetch_llama_overview("dexs", overview_slug) if overview_slug else None
    fees = fetch_llama_overview("fees", overview_slug) if overview_slug else None
    stable_supply_value = stable_supply.get(stable_key.lower()) if stable_key else None
    return {
        "llama_chain": chart_name or overview_slug,
        "llama_chain_tvl": tvl,
        "llama_dex_volume_24h": dex_volume,
        "llama_fees_24h": fees,
        "llama_stablecoin_supply": stable_supply_value,
    }


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_recent_supports(series: pd.Series, window: int = 7, top_n: int = 3) -> List[Dict[str, Any]]:
    supports: List[Dict[str, Any]] = []
    for idx in range(window, len(series) - window):
        value = series.iloc[idx]
        local_window = series.iloc[idx - window : idx + window + 1]
        if value == local_window.min():
            supports.append({"date": series.index[idx].strftime("%Y-%m-%d"), "price": float(value)})
    # 只保留最新的 top_n 个支撑
    supports = sorted(supports, key=lambda x: x["date"], reverse=True)[:top_n]
    return supports


def compute_volume_nodes(df: pd.DataFrame, bins: int = 20, top_n: int = 3) -> List[Dict[str, Any]]:
    hist, edges = np.histogram(df["price"], bins=bins, weights=df["volume"])
    nodes: List[Dict[str, Any]] = []
    for i in range(len(hist)):
        nodes.append(
            {
                "price_low": float(edges[i]),
                "price_high": float(edges[i + 1]),
                "volume": float(hist[i]),
            }
        )
    nodes = sorted(nodes, key=lambda x: x["volume"], reverse=True)[:top_n]
    return nodes


def fibonacci_levels(df: pd.DataFrame, lookback_days: int) -> Tuple[Dict[str, float], float, float]:
    tail = df[df["date"] >= (df["date"].max() - pd.Timedelta(days=lookback_days))]
    if tail.empty:
        tail = df
    swing_high = float(tail["price"].max())
    swing_low = float(tail["price"].min())
    diff = swing_high - swing_low
    levels = {
        "0%": swing_low,
        "38.2%": swing_low + 0.382 * diff,
        "50%": swing_low + 0.5 * diff,
        "61.8%": swing_low + 0.618 * diff,
        "100%": swing_high,
    }
    return {k: float(v) for k, v in levels.items()}, swing_low, swing_high


def detect_bullish_divergence(price: pd.Series, rsi: pd.Series, lookback: int = 90) -> bool:
    price_tail = price.tail(lookback)
    if price_tail.empty:
        return False
    rsi_tail = rsi.reindex(price_tail.index)
    lowest = price_tail.nsmallest(2)
    if len(lowest) < 2:
        return False
    first_date, second_date = sorted(lowest.index)[-2:]
    if second_date <= first_date:
        return False
    price_first = price.loc[first_date]
    price_second = price.loc[second_date]
    rsi_first = rsi.loc[first_date]
    rsi_second = rsi.loc[second_date]
    if pd.isna(rsi_first) or pd.isna(rsi_second):
        return False
    return price_second < price_first and rsi_second > rsi_first


@dataclass
class GlassnodeClient:
    api_key: Optional[str]

    def _request(self, endpoint: str, asset: str, **params) -> Optional[List[Dict[str, Any]]]:
        if not self.api_key:
            return None
        base_url = f"https://api.glassnode.com/v1/{endpoint}"
        query = {"a": asset, "api_key": self.api_key}
        query.update(params)
        try:
            resp = requests.get(base_url, params=query, timeout=40)
            if resp.status_code == 401:
                print("[WARN] Glassnode API key invalid or missing permissions.")
                return None
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[WARN] Glassnode request failed ({endpoint}): {exc}")
            return None
        return resp.json()

    def latest_value(self, endpoint: str, asset: str, **params) -> Optional[float]:
        data = self._request(endpoint, asset, **params)
        if not data:
            return None
        last_point = data[-1]
        if isinstance(last_point, dict) and "v" in last_point:
            value = last_point["v"]
            if isinstance(value, list):
                # distribution数据 (e.g. UTXO)
                return None
            return float(value)
        return None

    def latest_series(self, endpoint: str, asset: str, **params) -> Optional[List[Dict[str, Any]]]:
        return self._request(endpoint, asset, **params)


def fetch_fear_greed() -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(FEAR_GREED_API, params={"limit": 1}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data")
        if data:
            return data[0]
        return None
    except requests.RequestException as exc:
        print("[WARN] Fear & Greed API failed:", exc)
        return None


def fetch_funding_rate(symbol: str) -> Optional[float]:
    if not symbol:
        return None
    try:
        resp = requests.get(BINANCE_FUNDING_API, params={"symbol": symbol.upper(), "limit": 1}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list) and payload:
            return float(payload[0]["fundingRate"])
    except requests.RequestException as exc:
        print(f"[WARN] Funding rate fetch failed for {symbol}: {exc}")
    return None


def format_supports(supports: List[Dict[str, Any]]) -> str:
    return ", ".join([f"{s['date']}@{s['price']:.2f}" for s in supports]) if supports else ""


def pct_change_percent(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return float("nan")
    value = series.pct_change(periods).iloc[-1]
    return float(value * 100) if pd.notna(value) else float("nan")


def percent_diff(value: float, reference: float) -> float:
    if reference in (0, None) or pd.isna(reference):
        return float("nan")
    return float((value - reference) / reference * 100)


def main() -> None:
    cfg = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vs_currency = cfg.get("vs_currency", "usd")
    lookback_days = cfg.get("lookback_days", 420)
    swing_window = cfg.get("swing_window_days", 120)
    volume_bins = cfg.get("volume_bins", 20)

    coingecko_api_key = cfg.get("coingecko_api_key") or os.environ.get("COINGECKO_API_KEY")
    if not coingecko_api_key:
        coingecko_api_key = "CG-DemoAPIKey"

    glassnode = GlassnodeClient(cfg.get("glassnode_api_key"))
    glassnode_metrics = cfg.get("glassnode_metrics", {})
    stablecoin_asset = cfg.get("stablecoin_asset", "USDT")
    stablecoin_balance_metric = cfg.get("stablecoin_exchange_balance_metric")

    stablecoin_exchange_balance = None
    if stablecoin_balance_metric:
        stablecoin_exchange_balance = glassnode.latest_value(
            stablecoin_balance_metric, stablecoin_asset, i="24h", s="30d"
        )

    fear_greed_snapshot = fetch_fear_greed() or {}
    stable_supply_map = fetch_stablecoin_supply_map()

    dashboard_rows: List[Dict[str, Any]] = []

    for coin in cfg.get("coins", []):
        try:
            df, price_source = fetch_market_chart(
                coin["id"],
                vs_currency,
                lookback_days,
                api_key=coingecko_api_key,
                binance_symbol=coin.get("binance_symbol"),
                yf_symbol=coin.get("yfinance_symbol"),
            )
        except Exception as exc:
            print(f"[ERROR] Failed to fetch prices for {coin['id']}: {exc}")
            continue
        df.set_index("date", inplace=True)
        closes = pd.Series(np.asarray(df["price"], dtype=float).reshape(-1), index=df.index)
        volumes = pd.Series(np.asarray(df["volume"], dtype=float).reshape(-1), index=df.index)

        rsi_series = compute_rsi(closes)
        latest_rsi = float(rsi_series.iloc[-1])
        supports = detect_recent_supports(closes, window=7, top_n=3)
        support_strength = len(supports)
        fibs, swing_low, swing_high = fibonacci_levels(df.reset_index(), swing_window)
        hvn_nodes = compute_volume_nodes(df.reset_index(), bins=volume_bins, top_n=3)

        ma50 = float(closes.rolling(window=50).mean().iloc[-1])
        ma200 = float(closes.rolling(window=200).mean().iloc[-1])
        last_price = float(closes.iloc[-1])
        divergence = bool(detect_bullish_divergence(closes, rsi_series, lookback=90))
        pct_change_7d = pct_change_percent(closes, 7)
        pct_change_30d = pct_change_percent(closes, 30)
        distance_ma50 = percent_diff(last_price, ma50)
        distance_ma200 = percent_diff(last_price, ma200)
        volume_avg_30d = float(volumes.tail(30).mean()) if not volumes.empty else float("nan")
        swing_range = swing_high - swing_low

        onchain_snapshot: Dict[str, Any] = {}
        for key, endpoint in glassnode_metrics.items():
            if key == "utxo_realized_price_distribution":
                dist = glassnode.latest_series(endpoint, coin.get("glassnode_asset", coin["symbol"]))
                onchain_snapshot[key] = dist[-1]["v"] if dist else None
            else:
                value = glassnode.latest_value(endpoint, coin.get("glassnode_asset", coin["symbol"]))
                onchain_snapshot[key] = value

        sentiment_snapshot: Dict[str, Any] = {
            "fear_greed_value": fear_greed_snapshot.get("value"),
            "fear_greed_text": fear_greed_snapshot.get("value_classification"),
            "funding_rate": fetch_funding_rate(coin.get("binance_symbol")),
            "stablecoin_exchange_balance": stablecoin_exchange_balance,
        }

        retrieved_ts = datetime.now(timezone.utc).isoformat()
        llama_metrics = compose_llama_metrics(coin["symbol"], stable_supply_map)

        per_coin = {
            "symbol": coin["symbol"],
            "name": coin.get("name_cn") or coin["symbol"],
            "last_price": last_price,
            "ma50": ma50,
            "ma200": ma200,
            "ma_trend": "bullish" if last_price > ma200 else "bearish",
            "retrieved_at": retrieved_ts,
            "price_source": price_source,
            "recent_supports": supports,
            "support_strength": support_strength,
            "fibonacci_levels": fibs,
            "swing_low": swing_low,
            "swing_high": swing_high,
            "swing_range": swing_range,
            "top_volume_nodes": hvn_nodes,
            "rsi14": latest_rsi,
            "rsi_bullish_divergence": divergence,
            "pct_change_7d": pct_change_7d,
            "pct_change_30d": pct_change_30d,
            "distance_ma50_pct": distance_ma50,
            "distance_ma200_pct": distance_ma200,
            "volume_avg_30d": volume_avg_30d,
            "onchain": onchain_snapshot,
            "sentiment": sentiment_snapshot,
            "llama": llama_metrics,
        }

        out_path = OUTPUT_DIR / f"{coin['symbol']}_support_map.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(per_coin, fh, ensure_ascii=False, indent=2)

        dashboard_rows.append(
            {
                "symbol": coin["symbol"],
                "name": per_coin["name"],
                "last_price": last_price,
                "ma50": ma50,
                "ma200": ma200,
                "ma_trend": per_coin["ma_trend"],
                "retrieved_at": retrieved_ts,
                "price_source": price_source,
                "recent_supports": format_supports(supports),
                "support_strength": support_strength,
                "fib_38_2": fibs["38.2%"],
                "fib_50": fibs["50%"],
                "fib_61_8": fibs["61.8%"],
                "swing_low": swing_low,
                "swing_high": swing_high,
                "swing_range": swing_range,
                "hvn_zones": json.dumps(hvn_nodes, ensure_ascii=False),
                "rsi14": latest_rsi,
                "rsi_divergence": divergence,
                "pct_change_7d": pct_change_7d,
                "pct_change_30d": pct_change_30d,
                "distance_ma50_pct": distance_ma50,
                "distance_ma200_pct": distance_ma200,
                "volume_avg_30d": volume_avg_30d,
                "llama_chain": llama_metrics.get("llama_chain"),
                "llama_chain_tvl": llama_metrics.get("llama_chain_tvl"),
                "llama_dex_volume_24h": llama_metrics.get("llama_dex_volume_24h"),
                "llama_fees_24h": llama_metrics.get("llama_fees_24h"),
                "llama_stablecoin_supply": llama_metrics.get("llama_stablecoin_supply"),
                "mvrv": onchain_snapshot.get("mvrv"),
                "lth_cost_basis": onchain_snapshot.get("lth_cost_basis"),
                "exchange_netflow": onchain_snapshot.get("exchange_netflow"),
                "asopr": onchain_snapshot.get("asopr"),
                "funding_rate": sentiment_snapshot.get("funding_rate"),
                "fear_greed": sentiment_snapshot.get("fear_greed_value"),
                "fear_greed_text": sentiment_snapshot.get("fear_greed_text"),
            }
        )

    if dashboard_rows:
        dashboard = pd.DataFrame(dashboard_rows)
        dashboard.to_csv(OUTPUT_DIR / "crypto_support_dashboard.csv", index=False)
        print(f"[OK] wrote {OUTPUT_DIR / 'crypto_support_dashboard.csv'}")
    else:
        print("[WARN] No crypto snapshots were produced.")


if __name__ == "__main__":
    main()
