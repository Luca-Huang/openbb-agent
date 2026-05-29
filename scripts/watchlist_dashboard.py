"""Build a standalone HTML dashboard for the multi-market watchlist.

Covers 港股 / A股 / 美股 in one tabbed view. For each name it pulls live valuation
(Longbridge) + the current technical signal (production radar logic), derives a
建仓/等回踩/观察/回踩-谨慎/不碰 verdict from the entry-timing rules validated in
scripts/entry_timing_backtest.py, and renders a single self-contained HTML file
(inline CSS+JS, no external deps).

Configuration (baskets / thesis / SOTP segments / FX rates) lives in
``research_inputs/dashboard_config.json`` — edit there, the script just renders.
Qualitative thesis is authored knowledge (verify independently); all numeric
fields are live from the data provider.

Run:  python3 scripts/watchlist_dashboard.py    ->  outputs/watchlist_dashboard.html
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_workbench.data_sources.indicators import add_technical_indicators
from research_workbench.data_sources.longbridge import LongbridgeCLIProvider, get_provider
from research_workbench.signal_engine.radar import (
    DEFAULT_RADAR_CONFIG,
    add_radar_features,
    detect_trigger_type,
)

log = logging.getLogger("watchlist_dashboard")

OUT = ROOT / "outputs" / "watchlist_dashboard.html"
CONFIG_PATH = ROOT / "research_inputs" / "dashboard_config.json"
K_ATR = 1.0          # 不追高: extended if close > 20d-high + K_ATR*ATR
LOOKBACK_YEARS = 4   # Longbridge daily k-line cap ≈ 1000 bars ≈ 4 yrs

VERDICT_META = {  # label -> (sort order, css class, 中文释义)
    "建仓":       (0, "buy",   "突破达标且未越追高线，右侧确认机会"),
    "等回踩":     (1, "wait",  "已突破但冲高过头，挂回踩单别追"),
    "观察":       (2, "watch", "接近突破、结构健康，等放量触发"),
    "回踩-谨慎":  (2, "watch", "回踩信号，历史上偏弱，谨慎"),
    "不碰":       (3, "avoid", "下降趋势无信号，接飞刀区"),
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the dashboard config. Fail fast — without it there's nothing to render."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fnum(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe(label: str, fn, default):
    """Run fn() and log if it raises — do NOT silently swallow errors."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — explicitly catching everything to keep going
        log.warning("%s failed: %s: %s", label, type(exc).__name__, exc)
        return default


# ---------------------------------------------------------------------------
# Data fetchers (use provider's public methods)
# ---------------------------------------------------------------------------

def _valuation_metrics(payload: dict) -> dict:
    """Distill Longbridge valuation payload into PE percentile + ai_summary."""
    pe = payload.get("history", {}).get("metrics", {}).get("pe", {})
    vals = [v for v in (_fnum(x.get("value")) for x in pe.get("list", [])) if v and v > 0]
    cur = vals[-1] if vals else None
    cheap_pct = (sum(1 for v in vals if v > cur) / len(vals) * 100) if (vals and cur is not None) else None
    summary = re.sub("<[^>]+>", "", payload.get("overview", {}).get("ai_summary", "")).strip()
    return {"pe_cheap_pct": cheap_pct, "val_summary": summary}


def _ratings_metrics(payload: dict) -> dict:
    """Distill Longbridge institution-rating payload."""
    ins = payload.get("instratings", {}) or {}
    ev = ins.get("evaluate", {}) or {}
    return {
        "target": _fnum(ins.get("target")),
        "target_ccy": ins.get("ccy_symbol", ""),
        "recommend": ins.get("recommend", ""),
        "rate_sb": ev.get("strong_buy"),
        "rate_buy": ev.get("buy"),
        "rate_hold": ev.get("hold"),
        "rate_sell": ev.get("sell"),
    }


def _annual_segments(payload: dict) -> list[tuple[str, float]]:
    """Latest full-year segments (name, revenue). Uses current segment names."""
    hist = payload.get("historical") or []
    if not hist:
        return []
    return [
        (b.get("name", ""), _fnum(b.get("value")))
        for b in hist[-1].get("business", [])
        if _fnum(b.get("value"))
    ]


# ---------------------------------------------------------------------------
# SOTP (sum-of-the-parts) valuation
# ---------------------------------------------------------------------------

def _match_seg(name: str, segments: list[dict]) -> dict | None:
    for s in segments:
        if any(kw.lower() in name.lower() for kw in s["match"]):
            return s
    return None


def _fx(rate_map: dict[str, float], from_ccy: str, to_ccy: str) -> float:
    """Conversion factor (units of `to_ccy` per 1 unit `from_ccy`). 1.0 if same."""
    if from_ccy == to_ccy:
        return 1.0
    key = f"{from_ccy}/{to_ccy}"
    rate = rate_map.get(key)
    if rate is None:
        log.warning("FX rate missing: %s — falling back to 1.0 (SOTP may be wrong)", key)
        return 1.0
    return float(rate)


def build_sotp(
    symbol: str,
    cfg: dict,
    provider: LongbridgeCLIProvider,
    mktcap: float | None,
    mktcap_ccy: str,
    fx_rates: dict[str, float],
) -> dict | None:
    """Compute SOTP target & implied upside vs market cap. Returns None when not configured."""
    sotp_cfg = cfg.get("segments", {}).get(symbol)
    if not sotp_cfg:
        return None
    seg_ccy = sotp_cfg.get("segment_currency", "CNY")
    payload = _safe(
        f"business-segments[{symbol}]",
        lambda: provider.fetch_business_segments(symbol, history=True, report="af"),
        {},
    ) or {}
    rows: list[dict] = []
    total_seg_ccy = 0.0
    for name, rev in _annual_segments(payload):
        match = _match_seg(name, sotp_cfg["segments"])
        if not match:
            continue  # skip 抵消/未分摊
        val = rev * match["ps"]
        total_seg_ccy += val
        rows.append({
            "label": match["label"],
            "rev_yi": rev / 1e8,
            "ps": match["ps"],
            "val_yi": val / 1e8,
            "reason": match["reason"],
        })
    if not rows:
        return None

    for ex in sotp_cfg.get("extra", []):
        val_yi = float(ex["value_yi"])
        total_seg_ccy += val_yi * 1e8
        rows.append({
            "label": ex["label"], "rev_yi": None, "ps": None,
            "val_yi": val_yi, "reason": ex["reason"],
        })

    target_seg_yi = total_seg_ccy / 1e8
    fx = _fx(fx_rates, seg_ccy, mktcap_ccy)
    target_mc_yi = target_seg_yi * fx
    mktcap_yi = mktcap / 1e8 if isinstance(mktcap, (int, float)) and pd.notna(mktcap) else None
    upside = (target_mc_yi / mktcap_yi - 1) * 100 if mktcap_yi else None
    for r in rows:
        r["share"] = (r["val_yi"] / target_seg_yi * 100) if target_seg_yi else None
    return {
        "rows": rows,
        "target_yi": target_mc_yi,
        "mktcap_yi": mktcap_yi,
        "upside": upside,
        "seg_ccy": seg_ccy,
        "mc_ccy": mktcap_ccy,
        "fx": fx,
    }


# ---------------------------------------------------------------------------
# Verdict logic (kept inline — small, single source of truth)
# ---------------------------------------------------------------------------

def compute_verdict(
    close: float, ma20: float, ma200: float, h20: float, atr: float,
    trigger: str | None,
) -> tuple[str, float]:
    """Return (verdict_label, stop_price). Centralised so test can pin it."""
    cap = h20 + K_ATR * atr
    if trigger == "breakout":
        return ("建仓" if close <= cap else "等回踩"), h20
    if trigger == "pullback":
        return "回踩-谨慎", min(ma20, h20 * 0.97)
    dist_high = (close / h20 - 1) * 100 if h20 else -100.0
    if close >= ma20 and dist_high >= -5.0:
        return "观察", ma20
    return "不碰", ma20


# ---------------------------------------------------------------------------
# Single-symbol analysis
# ---------------------------------------------------------------------------

def analyze(symbol: str, market: str, cfg: dict) -> dict:
    prov = get_provider(market)
    start = date.today() - timedelta(days=LOOKBACK_YEARS * 365)
    raw = prov.fetch_history(symbol, start, date.today())
    df = add_radar_features(add_technical_indicators(raw)).reset_index(drop=True)
    r = df.iloc[-1]
    c = float(r["close"])
    ma20 = float(r["ma20"]); ma50 = float(r["ma50"]); ma200 = float(r["ma200"])
    h20 = float(r["high_20d"]); atr = float(r["atr14"]); vs = float(r["volume_spike_ratio"])
    rsi = float(r["rsi14"]); dd = float(r["drawdown_60d"])
    chg1 = (c / float(df.iloc[-2]["close"]) - 1) * 100
    trig = detect_trigger_type(r, DEFAULT_RADAR_CONFIG["triggers"])
    up = c >= ma200
    dist_high = (c / h20 - 1) * 100 if h20 else float("nan")
    cap = h20 + K_ATR * atr

    verdict, stop = compute_verdict(c, ma20, ma200, h20, atr, trig)
    risk_pct = (c - stop) / c * 100 if stop and c > stop else None

    fund = _safe(f"fundamentals[{symbol}]", lambda: prov.fetch_fundamentals(symbol), {}) or {}
    mktcap = fund.get("market_cap")
    mc_ccy = (fund.get("currency") or market_to_ccy(market)).upper()

    val_payload = _safe(f"valuation[{symbol}]", lambda: prov.fetch_valuation(symbol), {}) or {}
    val = _valuation_metrics(val_payload)
    rate_payload = _safe(f"institution-rating[{symbol}]", lambda: prov.fetch_institution_rating(symbol), {}) or {}
    rate = _ratings_metrics(rate_payload)

    target_upside = (rate["target"] / c - 1) * 100 if rate.get("target") and c else None
    sotp = build_sotp(symbol, cfg, prov, mktcap, mc_ccy, cfg.get("fx_rates", {}))

    return {
        "symbol": symbol, "name": fund.get("name") or symbol, "market": market,
        "date": r["date"].strftime("%Y-%m-%d"),
        "close": c, "chg1": chg1, "verdict": verdict,
        "trigger": {"breakout": "突破", "pullback": "回踩"}.get(trig, "无"),
        "regime": "上升" if up else "下降",
        "rsi": rsi, "vol_spike": vs, "dist_high": dist_high,
        "ma20": ma20, "ma50": ma50, "ma200": ma200, "vs_ma200": (c / ma200 - 1) * 100,
        "drawdown": dd * 100, "stop": stop, "risk_pct": risk_pct, "cap": cap, "high20": h20,
        "pe": fund.get("end_pe"), "pb": fund.get("current_pb"),
        "eps_ttm": fund.get("eps_ttm"), "div_yield": fund.get("dividend_yield"),
        "mktcap": mktcap, "mktcap_ccy": mc_ccy, "turnover": fund.get("turnover_rate"),
        "pe_cheap_pct": val["pe_cheap_pct"], "val_summary": val["val_summary"],
        "target": rate.get("target"), "target_ccy": rate.get("target_ccy"),
        "target_upside": target_upside, "recommend": rate.get("recommend"),
        "rate_sb": rate.get("rate_sb"), "rate_buy": rate.get("rate_buy"),
        "rate_hold": rate.get("rate_hold"), "rate_sell": rate.get("rate_sell"),
        "sotp": sotp,
    }


def market_to_ccy(market: str) -> str:
    return {"HK": "HKD", "CN": "CNY", "US": "USD"}.get(market, "USD")


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _num(v, fmt="{:.2f}", dash="—"):
    return fmt.format(v) if isinstance(v, (int, float)) and pd.notna(v) else dash


def _cap_str(v):
    if not isinstance(v, (int, float)) or pd.isna(v):
        return "—"
    return f"{v/1e12:.2f} 万亿" if v >= 1e12 else f"{v/1e8:.0f} 亿"


def _metric(label, value):
    return f'<div class="m"><span class="ml">{label}</span><span class="mv">{value}</span></div>'


def _valh(a: dict) -> str:
    """Valuation-health block: PE percentile bar + analyst target & ratings."""
    parts = []
    cp = a.get("pe_cheap_pct")
    if isinstance(cp, (int, float)) and pd.notna(cp):
        w = max(0, min(100, cp))
        tip = escape(a.get("val_summary") or "")
        parts.append(
            f'<div class="vh"><span class="vhl">估值便宜分位</span>'
            f'<div class="bar" title="{tip}"><div class="bf" style="width:{w:.0f}%"></div></div>'
            f'<span class="vhv">{cp:.0f}%</span></div>'
        )
    tgt = a.get("target")
    if isinstance(tgt, (int, float)) and pd.notna(tgt):
        up = a.get("target_upside")
        up_cls = "up" if (up or 0) >= 0 else "down"
        up_s = f'<em class="{up_cls}">{up:+.1f}%</em>' if isinstance(up, (int, float)) and pd.notna(up) else ""
        dist = "/".join(
            f"{lab}{a[k]}" for lab, k in
            (("强买", "rate_sb"), ("买", "rate_buy"), ("持", "rate_hold"), ("卖", "rate_sell"))
            if isinstance(a.get(k), (int, float))
        )
        parts.append(
            f'<div class="vh"><span class="vhl">机构目标</span>'
            f'<span class="vhv2">{a.get("target_ccy","")}{tgt:.0f} {up_s}'
            + (f' · {dist}' if dist else "") + '</span></div>'
        )
    return f'<section class="valh">{"".join(parts)}</section>' if parts else ""


def _sotp(a: dict) -> str:
    s = a.get("sotp")
    if not s or not s.get("rows"):
        return ""
    body = []
    for r in s["rows"]:
        tip = escape(r.get("reason") or "")
        rev = f'{r["rev_yi"]:.0f}亿' if r.get("rev_yi") else "—"
        # `is not None` instead of truthy so PS=0 (explicit "discarded") renders as ×0.0
        ps = f'×{r["ps"]:.1f}' if r.get("ps") is not None else "—"
        share = f'{r["share"]:.0f}%' if r.get("share") is not None else ""
        body.append(
            f'<tr title="{tip}"><td class="sl">{escape(r["label"])}</td>'
            f'<td>{rev}</td><td class="sp">{ps}</td>'
            f'<td class="sv">{r["val_yi"]:.0f}亿</td><td class="ss">{share}</td></tr>'
        )
    up = s.get("upside")
    tgt = s.get("target_yi"); mc = s.get("mktcap_yi")

    def _wan(v):
        return f"{v/1e4:.2f}万亿" if v and v >= 1e4 else (f"{v:.0f}亿" if v else "—")

    if isinstance(up, (int, float)) and pd.notna(up):
        up_cls = "buy" if up >= 0 else "avoid"
        badge = f'<span class="badge {up_cls}">隐含 {up:+.0f}%</span>'
        w = max(0, min(100, 50 + up / 2))  # center 50%, ±100% spans the bar
        bar = f'<div class="bar sotpbar"><div class="bf {up_cls}" style="width:{w:.0f}%"></div></div>'
    else:
        badge, bar = "", ""
    fx_note = (f"段{s['seg_ccy']}×{s['fx']:.3f}→{s['mc_ccy']}" if s.get("seg_ccy") != s.get("mc_ccy")
               else f"同币种 {s['seg_ccy']}")
    return (
        '<section class="sotp"><div class="sh">分部估值 (SOTP) '
        f'<span class="sh-x">营收×PS · {fx_note} · vs 现价市值</span></div>'
        f'<table class="st"><tbody>{"".join(body)}</tbody></table>'
        f'<div class="sf"><span>目标 <b>{_wan(tgt)}</b> · 现价 {_wan(mc)}</span>{badge}</div>{bar}</section>'
    )


def card(a: dict, thesis: dict) -> str:
    order, cls, vexpl = VERDICT_META[a["verdict"]]
    th = thesis.get(a["symbol"], {})
    chg1 = a["chg1"]; chg_cls = "up" if chg1 >= 0 else "down"
    fundamentals = "".join([
        _metric("PE", _num(a["pe"])), _metric("PB", _num(a["pb"])),
        _metric("EPS(TTM)", _num(a["eps_ttm"])),
        _metric("股息率", _num(a["div_yield"]*100, "{:.2f}%") if isinstance(a["div_yield"], (int, float)) and pd.notna(a["div_yield"]) else "—"),
        _metric("市值", _cap_str(a["mktcap"])),
        _metric("换手率", _num(a["turnover"], "{:.2f}%")),
    ])
    signal = "".join([
        _metric("触发", a["trigger"]), _metric("阶段", a["regime"]),
        _metric("RSI", _num(a["rsi"], "{:.0f}")), _metric("量比", _num(a["vol_spike"], "{:.2f}x")),
        _metric("距20日高", _num(a["dist_high"], "{:+.1f}%")),
        _metric("vs MA200", _num(a["vs_ma200"], "{:+.1f}%")),
        _metric("60日回撤", _num(a["drawdown"], "{:.1f}%")),
        _metric("止损位", f'{_num(a["stop"])}' + (f' (风险{_num(a["risk_pct"],"{:.1f}%")})' if a["risk_pct"] else "")),
    ])
    thesis_html = "".join(
        f'<div class="t"><span class="tk">{k}</span><span class="tv">{escape(th.get(k,"—"))}</span></div>'
        for k in ("业务", "多头", "空头", "估值")
    )
    return f"""
    <article class="card {cls}" data-order="{order}">
      <header>
        <div class="hl">
          <span class="nm">{escape(a['name'])}</span>
          <span class="code">{a['symbol']}</span>
        </div>
        <div class="hr">
          <span class="px">{_num(a['close'])}</span>
          <span class="chg {chg_cls}">{_num(chg1,'{:+.2f}%')}</span>
          <span class="badge {cls}">{a['verdict']}</span>
        </div>
      </header>
      <p class="vexpl">{vexpl}</p>
      <section class="grid">{fundamentals}</section>
      {_valh(a)}
      {_sotp(a)}
      <section class="grid sig">{signal}</section>
      <section class="thesis">{thesis_html}</section>
    </article>
    """


def _panel(market: str, rows: list[dict], active: bool, thesis: dict) -> str:
    rows = sorted(rows, key=lambda a: (VERDICT_META[a["verdict"]][0], -a["close"]))
    on = " on" if active else ""
    if not rows:
        return (f'<section class="panel{on}" data-m="{market}">'
                f'<div class="empty">暂无标的 — 在 dashboard_config.json 的 baskets.{market} 里添加</div></section>')
    panel_date = max(a["date"] for a in rows)
    n_buy = sum(1 for a in rows if a["verdict"] == "建仓")
    header = (f'<div class="phead">本市场数据日期 {panel_date} · {len(rows)} 只'
              f' · 可建仓 <b>{n_buy}</b> 只</div>')
    cards = "\n".join(card(a, thesis) for a in rows)
    return (f'<section class="panel{on}" data-m="{market}">{header}'
            f'<div class="wrap">{cards}</div></section>')


def render(by_market: dict[str, list[dict]], cfg: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_rows = [a for rs in by_market.values() for a in rs]
    total = len(all_rows)
    n_buy = sum(1 for a in all_rows if a["verdict"] == "建仓")
    market_tabs = cfg.get("market_tabs", [["HK", "港股"], ["CN", "A股"], ["US", "美股"]])
    active_market = next((m for m, _ in market_tabs if by_market.get(m)), market_tabs[0][0])

    tab_parts = []
    for m, label in market_tabs:
        rs = by_market.get(m, [])
        nb = sum(1 for a in rs if a["verdict"] == "建仓")
        dot = f'<b class="dot">{nb}</b>' if nb else ""
        on = " on" if m == active_market else ""
        tab_parts.append(
            f'<button class="tab{on}" data-m="{m}" onclick="showTab(\'{m}\')">'
            f'{label}<i>{len(rs)}</i>{dot}</button>'
        )
    tabs = "".join(tab_parts)
    thesis = cfg.get("thesis", {})
    panels = "".join(_panel(m, by_market.get(m, []), m == active_market, thesis) for m, _ in market_tabs)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>自选监控 · 基本面 & 信号</title>
<style>
  :root {{ --bg:#0f1115; --card:#181b22; --line:#262b35; --txt:#e6e8ec; --dim:#8b93a1;
           --buy:#1f9d57; --wait:#c98a1a; --watch:#3b7dd8; --avoid:#6b7280; --up:#e5534b; --down:#1f9d57; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--txt);
          font:14px/1.55 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }}
  .top {{ position:sticky; top:0; background:linear-gradient(180deg,#0f1115ee,#0f1115cc);
          backdrop-filter:blur(6px); border-bottom:1px solid var(--line); padding:14px 22px; }}
  .top h1 {{ margin:0; font-size:17px; font-weight:600; }}
  .top .sub {{ color:var(--dim); font-size:12px; margin-top:3px; }}
  .top .sub b {{ color:var(--buy); }}
  .tabs {{ display:flex; gap:6px; margin-top:11px; }}
  .tab {{ appearance:none; cursor:pointer; font:inherit; color:var(--dim);
          background:transparent; border:1px solid var(--line); border-radius:8px;
          padding:6px 14px; display:flex; align-items:center; gap:6px; transition:.12s; }}
  .tab:hover {{ color:var(--txt); border-color:#3a4150; }}
  .tab.on {{ color:var(--txt); background:var(--card); border-color:#3a4150; font-weight:600; }}
  .tab i {{ font-style:normal; font-size:11px; color:var(--dim); }}
  .tab .dot {{ font-style:normal; font-size:10px; font-weight:700; color:#fff;
               background:var(--buy); border-radius:10px; padding:0 6px; line-height:16px; }}
  .panel {{ display:none; }}
  .panel.on {{ display:block; }}
  .phead {{ color:var(--dim); font-size:12px; padding:14px 22px 0; max-width:1500px; }}
  .phead b {{ color:var(--buy); }}
  .empty {{ color:var(--dim); padding:40px 22px; font-size:13px; }}
  .wrap {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr));
           gap:14px; padding:14px 22px 40px; max-width:1500px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--avoid);
           border-radius:10px; padding:14px 16px; }}
  .card.buy {{ border-left-color:var(--buy); }}
  .card.wait {{ border-left-color:var(--wait); }}
  .card.watch {{ border-left-color:var(--watch); }}
  header {{ display:flex; justify-content:space-between; align-items:baseline; }}
  .nm {{ font-size:16px; font-weight:600; }}
  .code {{ color:var(--dim); font-size:12px; margin-left:7px; }}
  .hr {{ display:flex; align-items:center; gap:9px; }}
  .px {{ font-size:16px; font-weight:600; font-variant-numeric:tabular-nums; }}
  .chg {{ font-size:12px; font-variant-numeric:tabular-nums; }}
  .chg.up {{ color:var(--up); }} .chg.down {{ color:var(--down); }}
  .badge {{ font-size:12px; font-weight:600; padding:2px 9px; border-radius:20px; color:#fff; }}
  .badge.buy {{ background:var(--buy); }} .badge.wait {{ background:var(--wait); }}
  .badge.watch {{ background:var(--watch); }} .badge.avoid {{ background:var(--avoid); }}
  .vexpl {{ color:var(--dim); font-size:12px; margin:7px 0 11px; }}
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:7px 14px; margin-bottom:10px; }}
  .grid.sig {{ grid-template-columns:repeat(4,1fr); padding-top:10px; border-top:1px dashed var(--line); }}
  .m {{ display:flex; flex-direction:column; }}
  .ml {{ color:var(--dim); font-size:11px; }}
  .mv {{ font-variant-numeric:tabular-nums; font-weight:500; }}
  .valh {{ border-top:1px dashed var(--line); padding-top:9px; margin-bottom:2px; }}
  .vh {{ display:flex; align-items:center; gap:9px; margin-bottom:6px; }}
  .vhl {{ flex:0 0 70px; color:var(--dim); font-size:11px; }}
  .bar {{ flex:1; height:7px; background:#232833; border-radius:4px; overflow:hidden; }}
  .bf {{ height:100%; background:var(--watch); border-radius:4px; }}
  .bf.buy {{ background:var(--buy); }} .bf.avoid {{ background:var(--up); }}
  .vhv {{ flex:0 0 38px; text-align:right; font-size:12px; font-variant-numeric:tabular-nums; }}
  .vhv2 {{ flex:1; font-size:12px; font-variant-numeric:tabular-nums; }}
  .vhv2 em {{ font-style:normal; margin-left:3px; }}
  .vhv2 em.up {{ color:var(--up); }} .vhv2 em.down {{ color:var(--down); }}
  .sotp {{ border-top:1px dashed var(--line); padding-top:9px; margin-bottom:10px; }}
  .sh {{ font-size:12px; font-weight:600; margin-bottom:5px; }}
  .sh-x {{ color:var(--dim); font-weight:400; font-size:10.5px; margin-left:4px; }}
  .st {{ width:100%; border-collapse:collapse; font-size:12px; }}
  .st td {{ padding:2px 0; font-variant-numeric:tabular-nums; }}
  .st .sl {{ color:#c8cdd6; }} .st td {{ text-align:right; color:var(--dim); }}
  .st .sl {{ text-align:left; }} .st .sp {{ color:var(--watch); }}
  .st .sv {{ color:var(--txt); font-weight:500; }} .st .ss {{ width:34px; }}
  .sf {{ display:flex; justify-content:space-between; align-items:center; font-size:12px;
         margin-top:6px; padding-top:6px; border-top:1px solid var(--line); }}
  .sf b {{ color:var(--txt); }}
  .sotpbar {{ margin-top:6px; position:relative; }}
  .sotpbar::after {{ content:""; position:absolute; left:50%; top:0; bottom:0; width:1px; background:#3a4150; }}
  .thesis {{ border-top:1px solid var(--line); padding-top:10px; }}
  .t {{ display:flex; gap:8px; margin-bottom:5px; }}
  .tk {{ flex:0 0 34px; color:var(--dim); font-size:12px; }}
  .tv {{ flex:1; font-size:12.5px; color:#c8cdd6; }}
  footer {{ color:var(--dim); font-size:11px; padding:0 22px 30px; max-width:1500px; }}
</style></head>
<body>
  <div class="top">
    <h1>自选监控 · 基本面 &amp; 信号</h1>
    <div class="sub">生成于 {ts} · 共 {total} 只 · 全市场可建仓 <b>{n_buy}</b> 只
      &nbsp;|&nbsp; 入场规则：突破未越追高线=建仓，缩量阴跌=不碰</div>
    <div class="tabs">{tabs}</div>
  </div>
  <main>{panels}</main>
  <footer>说明：PE/PB/股息率/市值等为数据源(Longbridge)实时取数；触发/阶段/止损为生产 radar 逻辑计算；
    基本面分析为定性整理，仅供参考、请独立核实。结论基于已回测的入场规则，不构成投资建议。</footer>
  <script>
  function showTab(m){{
    document.querySelectorAll('.tab').forEach(function(t){{ t.classList.toggle('on', t.dataset.m===m); }});
    document.querySelectorAll('.panel').forEach(function(p){{ p.classList.toggle('on', p.dataset.m===m); }});
  }}
  </script>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    by_market: dict[str, list[dict]] = {}
    for market, label in cfg.get("market_tabs", []):
        basket = cfg.get("baskets", {}).get(market, [])
        if not basket:
            continue
        print(f"== {label} ({market}) ==")
        rows: list[dict] = []
        for sym, name in basket:
            try:
                rows.append(analyze(sym, market, cfg))
                print(f"  ✓ {name} {sym}: {rows[-1]['verdict']}")
            except Exception as e:
                print(f"  ✗ {name} {sym}: {type(e).__name__}: {str(e)[:80]}")
        by_market[market] = rows
    OUT.write_text(render(by_market, cfg), encoding="utf-8")
    print(f"\n已生成: {OUT}")


if __name__ == "__main__":
    main()
