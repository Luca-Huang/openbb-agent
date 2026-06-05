import sys, json, pandas as pd, numpy as np
from datetime import date
from pathlib import Path
sys.path.insert(0, "src")
from research_workbench.config import default_settings
from research_workbench.outputs.files import load_history
from research_workbench.signal_engine.holdings_tracker import analyze_holding, load_holdings
from research_workbench.signal_engine.radar import add_radar_features

SETTINGS = default_settings(Path("."))
hist_full = load_history(SETTINGS)
positions = load_holdings(Path("research_inputs/holdings.json"))

def resolve(c):
    c = str(c).strip().upper()
    if "." in c: return c
    return f"{c}.SZ" if c.startswith(("000","001","002","003","300","301")) else f"{c}.SH"

def _build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    close = df["close"]
    high  = df.get("high",  close)
    low   = df.get("low",   close)
    vol   = df.get("volume", pd.Series(np.nan, index=df.index))

    df["ma20"]  = close.rolling(20, min_periods=1).mean()
    df["ma50"]  = close.rolling(50, min_periods=1).mean()
    df["ma200"] = close.rolling(200, min_periods=1).mean()
    df["highest_close_20d"] = close.rolling(20, min_periods=1).max()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    vol_ma20 = vol.rolling(20, min_periods=1).mean()
    df["volume_ma20"]       = vol_ma20
    df["volume_spike_ratio"] = vol / vol_ma20.replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=1).mean()

    df["support_level_primary"] = low.rolling(20, min_periods=1).min()
    df["close_percentile"]      = close.rank(pct=True)

    rolling_max = close.rolling(60, min_periods=1).max()
    df["drawdown_60d"] = (rolling_max - close) / rolling_max.replace(0, np.nan)
    return df

def _fetch_yfinance(symbol: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        raw = ticker.history(period="2y")
        if raw.empty:
            return pd.DataFrame()
        raw = raw.reset_index()
        raw.columns = [c.lower() for c in raw.columns]
        if "datetime" in raw.columns and "date" not in raw.columns:
            raw = raw.rename(columns={"datetime": "date"})
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
        raw["symbol"] = symbol
        df = raw[["symbol","date","open","high","low","close","volume"]].copy()
        df = df.dropna(subset=["close"])
        return _build_indicators(df)
    except Exception as e:
        print(f"[yfinance] {symbol} failed: {e}", flush=True)
        return pd.DataFrame()

out = []
for p in positions:
    sym = resolve(p.code)
    h = hist_full[hist_full["symbol"]==sym].sort_values("date").copy() if not hist_full.empty else pd.DataFrame()

    if len(h) < 50:
        print(f"[info] {sym}: CSV has {len(h)} rows, falling back to yfinance", flush=True)
        h = _fetch_yfinance(sym)
        if not h.empty:
            h["symbol"] = sym

    if "support_level" in h.columns and "support_level_primary" not in h.columns:
        h["support_level_primary"] = h["support_level"]

    if len(h) < 2:
        out.append({"code": p.code, "name": p.name, "error": f"history too short ({len(h)} rows)"})
        continue

    h_today     = add_radar_features(h)
    h_yesterday = add_radar_features(h.iloc[:-1])
    try:
        today     = analyze_holding(h_today,     p)
        yesterday = analyze_holding(h_yesterday, p)
        out.append({
            "code": today.code, "name": today.name, "last_date": today.last_date,
            "today_action":    today.action,    "today_priority": today.priority, "today_note": today.note,
            "yesterday_action": yesterday.action, "yesterday_note": yesterday.note,
            "last_close": float(today.last_close),
            "floating_pnl_pct": None if pd.isna(today.floating_pnl_pct) else float(today.floating_pnl_pct),
            "trigger": today.trigger_type, "verdict": today.verdict,
            "stop_price":  float(today.stop_price),
            "tp1": float(today.take_profit_1), "tp2": float(today.take_profit_2),
        })
    except Exception as e:
        out.append({"code": p.code, "name": p.name, "error": str(e)})

result = {"today": date.today().isoformat(), "holdings": out}
Path("analysis.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
