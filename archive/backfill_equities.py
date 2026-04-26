import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client

SUPABASE_URL = "https://wpyrevceqirzpwcpulqz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndweXJldmNlcWlyenB3Y3B1bHF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMzODUzOTEsImV4cCI6MjA3ODk2MTM5MX0.vY-lSpINIwDc80Caq7tX6iQ_zcBaKDflO5AfV79-tZA"
HISTORY_DAYS = 365 * 3

client = create_client(SUPABASE_URL, SUPABASE_KEY)
config_path = Path('equity_config.json')
if config_path.exists():
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    STOCKS = [item['symbol'] for item in cfg.get('equities', [])]
else:
    STOCKS = ['MSFT','AAPL']


def chunked(rows, size=200):
    for i in range(0, len(rows), size):
        yield rows[i:i+size]


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
        val = series.iloc[idx]
        local = series.iloc[idx-window:idx+window+1]
        if val == local.min():
            supports.append(float(val))
    return supports[-top_n:]


def process_symbol(symbol: str):
    print(f"[INFO] {symbol}")
    data = yf.download(symbol, period='3y', interval='1d', progress=False)
    if hasattr(data.columns, 'levels'):
        data.columns = data.columns.droplevel(1)
    data = data.dropna()
    data = data.tail(HISTORY_DAYS)
    if data.empty:
        return []

    close = data['Close']
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi = compute_rsi(close)
    pct_90 = close.pct_change(90) * 100

    records = []
    for idx in range(len(data)):
        subset = data.iloc[:idx+1]
        date = subset.index[-1]
        supports = detect_supports(subset['Close'])
        rec = {
            'symbol': symbol,
            'as_of_date': date.date().isoformat(),
            'ma50': float(ma50.iloc[idx]) if not np.isnan(ma50.iloc[idx]) else None,
            'ma200': float(ma200.iloc[idx]) if not np.isnan(ma200.iloc[idx]) else None,
            'rsi14': float(rsi.iloc[idx]) if not np.isnan(rsi.iloc[idx]) else None,
            'pct_change': float(pct_90.iloc[idx]) if not np.isnan(pct_90.iloc[idx]) else None,
            'support_level_primary': supports[-1] if supports else None,
            'support_level_secondary': supports[-2] if len(supports) > 1 else None,
        }
        records.append(rec)
    return records


client.table('equity_metrics_history').delete().neq('symbol','ZZZ').execute()
for symbol in STOCKS:
    recs = process_symbol(symbol)
    for batch in chunked(recs):
        client.table('equity_metrics_history').insert(batch).execute()

print('[OK] Completed equity history backfill')
