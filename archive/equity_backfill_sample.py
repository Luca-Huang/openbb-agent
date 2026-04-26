import pandas as pd
import numpy as np
from datetime import datetime, timezone
import yfinance as yf
from supabase import create_client

SUPABASE_URL = "https://wpyrevceqirzpwcpulqz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndweXJldmNlcWlyenB3Y3B1bHF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMzODUzOTEsImV4cCI6MjA3ODk2MTM5MX0.vY-lSpINIwDc80Caq7tX6iQ_zcBaKDflO5AfV79-tZA"
HISTORY_DAYS = 365 * 3
RECENT_DAYS = 3
client = create_client(SUPABASE_URL, SUPABASE_KEY)

stocks = ["MSFT","AAPL","META","AMZN","GOOGL","NVDA","AMD","ADBE","CRM","NOW","AVGO","MRVL","ANET","SMCI","SNOW","MDB","TXN","TSM","ASML","SHOP","XOM","CVX","LLY","JNJ","COST","LVMUY","1810.HK","0700.HK","0763.HK","9992.HK","9988.HK","9888.HK","9999.HK","3690.HK","9961.HK","TCOM","RDDT","002624.SZ"]


def chunked(rows, size=200):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_supports(series: pd.Series, window=7, top_n=2):
    supports = []
    for idx in range(window, len(series) - window):
        value = series.iloc[idx]
        local = series.iloc[idx - window : idx + window + 1]
        if value == local.min():
            supports.append(float(value))
    return supports[-top_n:]


def fibonacci_levels(df: pd.DataFrame, lookback=120):
    tail = df.iloc[-lookback:] if len(df) > lookback else df
    swing_high = tail["Close"].max()
    swing_low = tail["Close"].min()
    diff = swing_high - swing_low
    if np.isnan(diff):
        return {"38": None, "50": None, "61": None, "low": None, "high": None}
    return {
        "38": swing_low + 0.382 * diff,
        "50": swing_low + 0.5 * diff,
        "61": swing_low + 0.618 * diff,
        "low": swing_low,
        "high": swing_high,
    }


def process_symbol(symbol: str):
    print(f"[INFO] {symbol}")
    data = yf.download(symbol, period="3y", interval="1d", progress=False)
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.droplevel(1)
    data = data.dropna()
    data = data.tail(HISTORY_DAYS)
    if data.empty:
        return []
    close = data["Close"]
    volume = data["Volume"]
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi = compute_rsi(close)
    pct_90d = close.pct_change(90) * 100

    records = []
    for idx in range(len(data)):
        subset = data.iloc[: idx + 1]
        date = subset.index[-1]
        supports = detect_supports(subset["Close"], window=7, top_n=2)
        fibs = fibonacci_levels(subset)
        rec = {
            "symbol": symbol,
            "as_of_date": date.date().isoformat(),
            "last_price": float(close.iloc[idx]),
            "ma50": float(ma50.iloc[idx]) if not np.isnan(ma50.iloc[idx]) else None,
            "ma200": float(ma200.iloc[idx]) if not np.isnan(ma200.iloc[idx]) else None,
            "ma_trend": "bullish" if ma200.iloc[idx] and not np.isnan(ma200.iloc[idx]) and close.iloc[idx] > ma200.iloc[idx] else "bearish",
            "rsi14": float(rsi.iloc[idx]) if not np.isnan(rsi.iloc[idx]) else None,
            "pct_change": float(pct_90d.iloc[idx]) if not np.isnan(pct_90d.iloc[idx]) else None,
            "support_level_primary": supports[-1] if supports else None,
            "support_level_secondary": supports[-2] if len(supports) > 1 else None,
            "fib_38_2": fibs["38"],
            "fib_50": fibs["50"],
            "fib_61_8": fibs["61"],
            "swing_low": fibs["low"],
            "swing_high": fibs["high"],
            "volume_avg_30d": float(volume.iloc[: idx + 1].rolling(30).mean().iloc[-1]) if idx + 1 >= 30 else None,
        }
        records.append(rec)
    return records


all_records = []
for symbol in STOCKS:
    all_records.extend(process_symbol(symbol))

print(f"Prepared {len(all_records)} rows")
