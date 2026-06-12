import sys, json, pandas as pd, numpy as np
from datetime import date
from pathlib import Path
sys.path.insert(0, "src")
from research_workbench.config import default_settings
from research_workbench.outputs.files import load_history
from research_workbench.signal_engine.holdings_tracker import analyze_holding, load_holdings
from research_workbench.signal_engine.radar import add_radar_features
from research_workbench.data_sources.indicators import add_technical_indicators

SETTINGS = default_settings(Path("."))
# openbb_outputs is the actual data directory (outputs/research_data is the config default but unused here)
_HISTORY_CSV = Path("openbb_outputs/three_month_close_history.csv")
positions = load_holdings(Path("research_inputs/holdings.json"))

def _resolve(c):
    c = str(c).strip().upper()
    if "." in c: return c
    return f"{c}.SZ" if c.startswith(("000","001","002","003","300","301")) else f"{c}.SH"

def _fetch_yf(sym: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        hist = yf.Ticker(sym).history(period="1y", auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        hist.columns = [c.lower() for c in hist.columns]
        if "date" in hist.columns:
            hist["date"] = pd.to_datetime(hist["date"], utc=True).dt.tz_localize(None)
        hist["symbol"] = sym.upper()
        cols = [c for c in ["symbol","date","open","high","low","close","volume"] if c in hist.columns]
        df = hist[cols].dropna(subset=["close"])
        return add_technical_indicators(df)
    except Exception as e:
        print(f"[analyze] yfinance error for {sym}: {e}")
        return pd.DataFrame()

if _HISTORY_CSV.exists():
    hist_full = pd.read_csv(_HISTORY_CSV)
    if "date" in hist_full.columns:
        hist_full["date"] = pd.to_datetime(hist_full["date"], errors="coerce")
    if "symbol" in hist_full.columns:
        hist_full["symbol"] = hist_full["symbol"].astype(str).str.upper()
    print(f"[analyze] loaded {len(hist_full)} rows from {_HISTORY_CSV}")
else:
    hist_full = load_history(SETTINGS)

wanted = {_resolve(p.code) for p in positions}
have = set(hist_full["symbol"].unique()) if not hist_full.empty else set()
missing = wanted - have
if missing:
    frames = [hist_full] if not hist_full.empty else []
    for sym in sorted(missing):
        print(f"[analyze] yfinance fetch: {sym}")
        df = _fetch_yf(sym)
        if df.empty:
            print(f"[analyze] WARNING: no data for {sym}")
        else:
            frames.append(df)
    hist_full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def resolve(c):
    return _resolve(c)

_DATA_ERR = "数据获取失败：Yahoo Finance / AKShare / Longbridge 均不在网络白名单，无法拉取行情"
out = []
for p in positions:
    sym = resolve(p.code)
    if hist_full.empty or "symbol" not in hist_full.columns:
        out.append({"code": p.code, "name": p.name, "error": _DATA_ERR})
        continue
    h = hist_full[hist_full["symbol"]==sym].sort_values("date").copy()
    if "support_level" in h.columns and "support_level_primary" not in h.columns:
        h["support_level_primary"] = h["support_level"]
    if len(h) < 2:
        out.append({"code": p.code, "name": p.name, "error": f"history too short ({len(h)} rows)"})
        continue
    h_today = add_radar_features(h)
    h_yesterday = add_radar_features(h.iloc[:-1])
    try:
        today = analyze_holding(h_today, p)
        yesterday = analyze_holding(h_yesterday, p)
        out.append({
            "code": today.code, "name": today.name, "last_date": today.last_date,
            "today_action": today.action, "today_priority": today.priority, "today_note": today.note,
            "yesterday_action": yesterday.action, "yesterday_note": yesterday.note,
            "last_close": float(today.last_close),
            "floating_pnl_pct": None if pd.isna(today.floating_pnl_pct) else float(today.floating_pnl_pct),
            "trigger": today.trigger_type, "verdict": today.verdict,
            "stop_price": float(today.stop_price),
            "tp1": float(today.take_profit_1), "tp2": float(today.take_profit_2),
        })
    except Exception as e:
        out.append({"code": p.code, "name": p.name, "error": str(e)})

result = {"today": date.today().isoformat(), "holdings": out}
Path("analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
