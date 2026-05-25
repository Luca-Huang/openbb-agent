import sys, json, pandas as pd
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

out = []
for p in positions:
    sym = resolve(p.code)
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
