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


def _akshare_shareholder_changes(symbol: str) -> list[dict]:
    """Pull recent shareholder buy/sell records for an A-share.

    Returns up to 5 records within last 180 days, sorted newest-first.
    Each record: {date, shareholder, change_text, shares, avg_price, is_buy}.
    """
    try:
        from research_workbench.data_sources.akshare_cn import get_provider as cn_get_provider
        df = _safe(f"shareholder[{symbol}]",
                   lambda: cn_get_provider("CN").fetch_shareholder_changes(symbol),
                   pd.DataFrame())
    except Exception:
        return []
    if df is None or df.empty:
        return []
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
    if "announce_date" in df.columns:
        df = df[df["announce_date"] >= cutoff]
    if df.empty:
        return []
    out: list[dict] = []
    for _, row in df.head(5).iterrows():
        text = str(row.get("change_text", ""))
        out.append({
            "date": row["announce_date"].strftime("%Y-%m-%d") if pd.notna(row.get("announce_date")) else "—",
            "shareholder": str(row.get("shareholder", ""))[:24],
            "change_text": text,
            "shares": row.get("change_shares"),
            "avg_price": row.get("avg_price"),
            "is_buy": "增" in text,
        })
    return out


def load_holdings() -> dict[str, dict]:
    """Load current holdings keyed by bare code (e.g. '002602' → row).

    Reads analysis.json (output of analyze.py). Returns {} if missing/malformed
    so dashboard never fails when holdings file isn't there.
    """
    path = ROOT / "analysis.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(h.get("code", "")).strip(): h for h in data.get("holdings", []) if h.get("code")}
    except Exception as exc:
        log.warning("load_holdings failed: %s", exc)
        return {}


def _akshare_consensus_for_a_share(symbol: str, current_price: float, current_pe: float | None) -> dict:
    """Synthesize an 'implied target' consensus for A-shares from 东财 reports.

    Returns the same shape as `_ratings_metrics()` so callers don't branch.
    'target' here = mean(2026 EPS forecast) × current PE — explicitly an
    implied target, not a sell-side stated target price. The
    `is_implied=True` flag is rendered in the tooltip.
    """
    empty = {"target": None, "target_low": None, "target_high": None,
             "target_ccy": "¥", "recommend": "",
             "rate_sb": None, "rate_buy": None, "rate_hold": None, "rate_sell": None,
             "is_implied": True}
    try:
        from research_workbench.data_sources.akshare_cn import get_provider as cn_get_provider
        ak_prov = cn_get_provider("CN")
        df = _safe(f"research_reports[{symbol}]", lambda: ak_prov.fetch_research_reports(symbol),
                   pd.DataFrame())
    except Exception:
        return empty
    if df is None or df.empty or current_pe is None or current_price is None:
        return empty
    # Only reports from last 12 months — older forecasts are stale
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
    df = df[df["announce_date"] >= cutoff] if "announce_date" in df.columns else df
    if df.empty or "eps_2026" not in df.columns:
        return empty
    eps_series = df["eps_2026"].dropna()
    if len(eps_series) < 3:
        return empty
    eps_low, eps_mean, eps_high = eps_series.min(), eps_series.mean(), eps_series.max()
    # 'Implied target' = forecast EPS × reasonable PE multiple.
    # Cap PE at 25 to avoid the early-growth trap: e.g. 完美世界 has TTM PE 51
    # (low EPS base before 异环), so forecast EPS × current PE → +200% nonsense.
    # When EPS actually grows, PE compresses; capping at 25 reflects that.
    PE_CAP = 25.0
    pe_used = min(current_pe, PE_CAP)
    tgt_low = eps_low * pe_used
    tgt_high = eps_high * pe_used
    tgt_mean = eps_mean * pe_used
    # 东财评级 distribution
    ratings = df["rating"].dropna() if "rating" in df.columns else pd.Series(dtype=str)
    sb = int((ratings == "强烈推荐").sum()) if not ratings.empty else 0
    buy = int(((ratings == "买入") | (ratings == "推荐")).sum())
    hold = int(((ratings == "增持") | (ratings == "中性")).sum())
    sell = int(((ratings == "减持") | (ratings == "卖出")).sum())
    return {
        "target": float(tgt_mean),
        "target_low": float(tgt_low),
        "target_high": float(tgt_high),
        "target_ccy": "¥",
        "recommend": "buy" if buy + sb > hold + sell else "hold",
        "rate_sb": sb, "rate_buy": buy, "rate_hold": hold, "rate_sell": sell,
        "is_implied": True,
    }


def _ratings_metrics(payload: dict) -> dict:
    """Distill Longbridge institution-rating payload."""
    ins = payload.get("instratings", {}) or {}
    ev = ins.get("evaluate", {}) or {}
    # `analyst.target` carries lowest/highest from the underlying analyst pool —
    # essential for showing dispersion rather than a single misleading mean.
    an_tgt = (payload.get("analyst", {}) or {}).get("target", {}) or {}
    return {
        "target": _fnum(ins.get("target")),
        "target_low": _fnum(an_tgt.get("lowest_price")),
        "target_high": _fnum(an_tgt.get("highest_price")),
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


def _sotp_rows_from_manual(manual_segments: list[dict]) -> tuple[list[dict], float]:
    """Build SOTP rows from user-authored 'profit × PE' segments.

    Use this path when Longbridge has no segment data for a stock (most A-shares)
    or when you want a forward-looking thesis (e.g. estimated profit from an
    unreleased product). Each manual segment supplies its own profit + PE; the
    'PS' column in the rendered table is left blank since there is no revenue.
    """
    rows: list[dict] = []
    total = 0.0
    for ms in manual_segments:
        profit_yi = float(ms.get("profit_yi", 0))
        pe = float(ms.get("pe", 0))
        val_yi = profit_yi * pe
        total += val_yi * 1e8
        rows.append({
            "label": str(ms.get("label", "未命名")),
            "rev_yi": None,  # manual is profit-driven, no revenue column
            "ps": None,      # no PS — show 利润×PE in reason tooltip instead
            "val_yi": val_yi,
            "reason": f"利润{profit_yi:.1f}亿×PE{pe:.0f} — {ms.get('reason','')}",
        })
    return rows, total


def build_sotp(
    symbol: str,
    cfg: dict,
    provider: LongbridgeCLIProvider,
    mktcap: float | None,
    mktcap_ccy: str,
    fx_rates: dict[str, float],
) -> dict | None:
    """Compute SOTP target & implied upside vs market cap. Returns None when not configured.

    Supports two configuration shapes for `cfg["segments"][symbol]`:

    1. Auto (revenue × PS): config has `segments` list; each item matches a
       Longbridge `business-segments` row by keyword and multiplies revenue by PS.
       Falls back gracefully if Longbridge returns nothing.

    2. Manual (profit × PE): config has `manual_segments` list; profit & PE come
       straight from the JSON. No Longbridge call. Use for A-shares without
       segment data, or for forward-looking thesis valuations.

    `extra` lines (non-revenue value like net cash / investment NAV) work for both.
    """
    sotp_cfg = cfg.get("segments", {}).get(symbol)
    if not sotp_cfg:
        return None
    seg_ccy = sotp_cfg.get("segment_currency", "CNY")
    rows: list[dict] = []
    total_seg_value = 0.0  # raw value in segment_currency

    if "manual_segments" in sotp_cfg:
        rows, total_seg_value = _sotp_rows_from_manual(sotp_cfg["manual_segments"])
    elif "segments" in sotp_cfg:
        payload = _safe(
            f"business-segments[{symbol}]",
            lambda: provider.fetch_business_segments(symbol, history=True, report="af"),
            {},
        ) or {}
        for name, rev in _annual_segments(payload):
            match = _match_seg(name, sotp_cfg["segments"])
            if not match:
                continue  # skip 抵消/未分摊
            val = rev * match["ps"]
            total_seg_value += val
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
        total_seg_value += val_yi * 1e8
        rows.append({
            "label": ex["label"], "rev_yi": None, "ps": None,
            "val_yi": val_yi, "reason": ex["reason"],
        })

    target_seg_yi = total_seg_value / 1e8
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
        "mode": "manual" if "manual_segments" in sotp_cfg else "auto",
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

    # A-share fallback: Longbridge has no analyst rating data for A-shares.
    # Pull 东财 research reports via AKShare and synthesize an implied-target
    # consensus from EPS forecasts × current PE. Marked `is_implied=True` so
    # tooltip can disclose the methodology difference.
    shareholder_changes: list[dict] = []
    if market == "CN" and not rate.get("target"):
        rate = _akshare_consensus_for_a_share(symbol, c, fund.get("end_pe"))
    # Shareholder buy/sell signal — only for A-shares, latest 3 within 180 days.
    if market == "CN":
        shareholder_changes = _akshare_shareholder_changes(symbol)

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
        "target_low": rate.get("target_low"), "target_high": rate.get("target_high"),
        "target_upside": target_upside, "recommend": rate.get("recommend"),
        "is_implied_target": rate.get("is_implied", False),
        "rate_sb": rate.get("rate_sb"), "rate_buy": rate.get("rate_buy"),
        "rate_hold": rate.get("rate_hold"), "rate_sell": rate.get("rate_sell"),
        "sotp": sotp,
        "shareholder_changes": shareholder_changes,
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


GLOSSARY = {
    # 基本面
    "PE": "市盈率 = 股价 ÷ 每股盈利。花多少钱买1块钱年利润。10-25 算合理，<10 偏便宜，>30 偏贵（不同行业差异大）。",
    "PB": "市净率 = 股价 ÷ 每股净资产。<1 破净，1-3 中等，>5 偏高。回购大户(如苹果)会因为净资产被压缩导致 PB 虚高，不能单看。",
    "EPS(TTM)": "每股盈利(最近12个月)。是 PE 计算的分母；TTM 比单年报更新更及时。",
    "股息率": "年股息 ÷ 当前股价。如果只靠分红每年能拿回多少%。3-5% 算稳健红利股。",
    "市值": "总股本 × 当前股价。公司在市场上整体值多少钱。",
    "换手率": "当日成交股数 ÷ 流通股本。流动性指标：<0.5% 冷清，1-3% 活跃，>5% 极活跃(情绪过热)。",
    # 估值健康
    "估值便宜分位": "当前 PE 在过去 3-5 年的位置。100%=比过去任何时候都便宜，0%=比过去都贵。配合行业排名看。",
    "机构目标": "横条 = 卖方分析师目标价的真实区间(最低-最高)，●=现价位置，细线=均值位置。区间宽=分歧大；现价贴下沿=市场对乐观共识打折(往往是买点)；贴上沿=已经透支预期。'+N%' 仅是相对均值，单独看会误导。",
    # SOTP
    "分部估值 (SOTP)": "Sum-Of-The-Parts：把公司各业务分别估值再加总，避免把不同业务硬套同一个 PE。适合多业务结构(腾讯/阿里/小米等)。",
    # 技术信号
    "触发": "技术信号系统当前给出的入场信号：突破/回踩/无。'突破' = 价格创新高且放量。",
    "阶段": "趋势阶段：上升(MA20>MA50>MA200) / 下降 / 震荡。",
    "RSI": "相对强弱指数。>70 超买(可能短期回调)，<30 超卖(可能反弹)，30-70 中性。",
    "量比": "当日成交量 ÷ 过去 20 日平均。>1.5 放量(信号变强)，<0.5 缩量(信号变弱)。",
    "距20日高": "当前价相对过去 20 日最高价的距离。-5% 内属于'高位附近'。",
    "vs MA200": "当前价相对 200 日均线。>0 长期趋势向上，<0 趋势向下，<-10% 跌得比较深。",
    "60日回撤": "过去 60 天从最高点回落的最大幅度。",
    "止损位": "技术系统建议的止损价格 + 当前到止损的风险%。建议风险 < 5%。",
}


def _gl(label: str) -> str:
    """Wrap label with tooltip if glossary defines it."""
    tip = GLOSSARY.get(label)
    if not tip:
        return escape(label)
    return f'<span class="g" data-tip="{escape(tip)}">{escape(label)}</span>'


def _metric(label, value):
    return f'<div class="m"><span class="ml">{_gl(label)}</span><span class="mv">{value}</span></div>'


def _valh(a: dict) -> str:
    """Valuation-health block: PE percentile bar + analyst target & ratings."""
    parts = []
    cp = a.get("pe_cheap_pct")
    if isinstance(cp, (int, float)) and pd.notna(cp):
        w = max(0, min(100, cp))
        tip = escape(a.get("val_summary") or "")
        parts.append(
            f'<div class="vh"><span class="vhl">{_gl("估值便宜分位")}</span>'
            f'<div class="bar" title="{tip}"><div class="bf" style="width:{w:.0f}%"></div></div>'
            f'<span class="vhv">{cp:.0f}%</span></div>'
        )
    tgt = a.get("target")
    if isinstance(tgt, (int, float)) and pd.notna(tgt):
        lo = a.get("target_low"); hi = a.get("target_high"); cur = a.get("close")
        ccy = a.get("target_ccy", "")
        up = a.get("target_upside")
        up_cls = "up" if (up or 0) >= 0 else "down"
        up_s = (f'<em class="{up_cls}">{up:+.0f}%</em>'
                if isinstance(up, (int, float)) and pd.notna(up) else "")
        # rating distribution chips
        dist = "/".join(
            f"{lab}{a[k]}" for lab, k in
            (("强买", "rate_sb"), ("买", "rate_buy"), ("持", "rate_hold"), ("卖", "rate_sell"))
            if isinstance(a.get(k), (int, float))
        )
        # If we have a real range, render the dispersion bar; else fall back to text-only.
        if (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and hi > lo
                and isinstance(cur, (int, float)) and pd.notna(cur)):
            span = hi - lo
            pos = max(0, min(100, (cur - lo) / span * 100))
            mean_pos = max(0, min(100, (tgt - lo) / span * 100))
            # qualitative tag for where current price sits
            if pos < 15:
                tag, tag_cls = "贴下沿", "buy"   # market discounts analysts → contrarian buy
            elif pos < 40:
                tag, tag_cls = "偏下", "buy"
            elif pos < 60:
                tag, tag_cls = "中段", "watch"
            elif pos < 85:
                tag, tag_cls = "偏上", "wait"
            else:
                tag, tag_cls = "贴上沿", "avoid"
            n_cov = (a.get('rate_sb') or 0)+(a.get('rate_buy') or 0)+(a.get('rate_hold') or 0)+(a.get('rate_sell') or 0)
            if a.get("is_implied_target"):
                tip = (f"⚠️ 隐含目标(非真目标价)：东财研报无目标价字段，用最近 12 个月研报的"
                       f"2026 EPS 预测 × min(当前PE, 25) 反推。区间 {ccy}{lo:.2f} - {ccy}{hi:.2f}"
                       f"，均 {ccy}{tgt:.2f}（{n_cov} 家覆盖）。现价 {ccy}{cur:.2f} 位置 {pos:.0f}%。"
                       f"PE 上限 25 防早期成长股目标价虚高(完美世界 PE 51 案例)。")
            else:
                tip = (f"分析师目标价区间 {ccy}{lo:.0f} - {ccy}{hi:.0f}（覆盖 {n_cov} 家）。"
                       f"现价 {ccy}{cur:.0f} 处于区间 {pos:.0f}% 位（{tag}）。"
                       f"均值 {ccy}{tgt:.0f}。位置越靠下沿 = 市场对乐观共识打折。")
            label = "机构目标" + (' <span class="impl-tag">隐含</span>' if a.get("is_implied_target") else "")
            parts.append(
                f'<div class="vh"><span class="vhl">{_gl("机构目标") if not a.get("is_implied_target") else label}</span>'
                f'<div class="rangebar" title="{escape(tip)}">'
                f'  <div class="rb-mean" style="left:{mean_pos:.1f}%" title="均值位置"></div>'
                f'  <div class="rb-cur" style="left:{pos:.1f}%"></div>'
                f'</div>'
                f'<span class="vhv-rb {tag_cls}">{pos:.0f}%·{tag}</span></div>'
                f'<div class="vh-meta">'
                f'<span class="rb-lo">{ccy}{lo:.0f}</span> – '
                f'<b>现价 {ccy}{cur:.0f}</b> – '
                f'均 {ccy}{tgt:.0f} {up_s} – '
                f'<span class="rb-hi">{ccy}{hi:.0f}</span>'
                + (f' · {dist}' if dist else "") +
                '</div>'
            )
        else:
            # fallback when range missing (e.g., A 股 small caps)
            parts.append(
                f'<div class="vh"><span class="vhl">{_gl("机构目标")}</span>'
                f'<span class="vhv2">{ccy}{tgt:.0f} {up_s}'
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
    mode = "手填利润×PE" if s.get("mode") == "manual" else "营收×PS"
    return (
        f'<section class="sotp"><div class="sh">{_gl("分部估值 (SOTP)")} '
        f'<span class="sh-x">{mode} · {fx_note} · vs 现价市值</span></div>'
        f'<table class="st"><tbody>{"".join(body)}</tbody></table>'
        f'<div class="sf"><span>目标 <b>{_wan(tgt)}</b> · 现价 {_wan(mc)}</span>{badge}</div>{bar}</section>'
    )


def _thesis_html(th: dict) -> str:
    """Render thesis section.

    New schema (preferred): th has `narrative` (list of paragraphs) + optional
    `non_consensus`, `catalysts` (list of {date,event,watch}), `sell_triggers`
    ({"前瞻": [...], "滞后": [...]}).

    Legacy schema (fallback): th has 业务/多头/空头/估值 as single-sentence fields.
    """
    narrative = th.get("narrative")
    if not narrative:
        # legacy fallback
        return "".join(
            f'<div class="t"><span class="tk">{escape(k)}</span>'
            f'<span class="tv">{escape(th.get(k,"—"))}</span></div>'
            for k in ("业务", "多头", "空头", "估值")
        )

    paras = "".join(f"<p>{escape(p)}</p>" for p in narrative)
    parts = [f'<div class="narr">{paras}</div>']

    nc = th.get("non_consensus")
    if nc:
        parts.append(
            f'<div class="nc"><div class="bh">非共识观点</div>'
            f'<div class="bb">{escape(nc)}</div></div>'
        )

    cats = th.get("catalysts") or []
    if cats:
        # Tag expired entries — render greyed out and demoted, so stale config
        # (e.g. 完美世界 异环 2026-04-23) doesn't show as upcoming.
        today = pd.Timestamp.now().normalize()
        def _is_past(c):
            d = str(c.get("date", ""))
            # only fully numeric YYYY-MM-DD entries are reliably parseable;
            # 'YYYY-MM' or '2026-Q3' etc. → treat as not-past (forward-looking)
            try:
                return pd.Timestamp(d) < today
            except (ValueError, TypeError):
                return False
        items = "".join(
            f'<li class="{"past" if _is_past(c) else ""}"><b>{escape(str(c.get("date","?")))}</b> '
            f'{("✓ 已发生 · " if _is_past(c) else "")}'
            f'{escape(str(c.get("event","")))} '
            f'<span class="dim">— 看 {escape(str(c.get("watch","")))}</span></li>'
            for c in sorted(cats, key=lambda x: (_is_past(x), str(x.get("date", ""))))
        )
        parts.append(
            f'<details class="cat" open><summary class="bh">近期催化日历</summary>'
            f'<ul>{items}</ul></details>'
        )

    triggers = th.get("sell_triggers") or {}
    if triggers:
        cols = []
        for k in ("前瞻", "滞后"):
            lst = triggers.get(k) or []
            if not lst:
                continue
            items = "".join(f"<li>{escape(s)}</li>" for s in lst)
            cols.append(f'<div class="ec"><b>{k}</b><ul>{items}</ul></div>')
        if cols:
            parts.append(
                f'<details class="exit"><summary class="bh">卖出触发条件</summary>'
                f'<div class="eg">{"".join(cols)}</div></details>'
            )

    return "".join(parts)


def _shareholder_html(a: dict) -> str:
    """Render recent (180d) shareholder buy/sell changes. CN only — empty if none."""
    changes = a.get("shareholder_changes") or []
    if not changes:
        return ""
    rows = []
    n_buy = sum(1 for c in changes if c.get("is_buy"))
    n_sell = len(changes) - n_buy
    for c in changes:
        sh = c.get("shares")
        sh_s = ""
        if isinstance(sh, (int, float)) and not pd.isna(sh):
            sh_abs = abs(sh)
            sh_s = f"{sh_abs/1e4:.0f}万" if sh_abs >= 1e4 else f"{sh_abs:.0f}"
        ap = c.get("avg_price")
        ap_s = f" @¥{ap:.2f}" if isinstance(ap, (int, float)) and pd.notna(ap) else ""
        sign_cls = "buy" if c.get("is_buy") else "sell"
        sign_sym = "↑" if c.get("is_buy") else "↓"
        rows.append(
            f'<tr><td class="sh-d">{escape(c["date"])}</td>'
            f'<td class="sh-n">{escape(c["shareholder"])}</td>'
            f'<td class="sh-c {sign_cls}">{sign_sym} {sh_s}{ap_s}</td></tr>'
        )
    summary = f'<span class="sh-buy">{n_buy} 增</span> / <span class="sh-sell">{n_sell} 减</span>'
    return (
        f'<details class="shareholders"><summary class="bh">'
        f'最近 180 天股东变动 <span class="sh-x">({len(changes)} 条 · {summary})</span>'
        f'</summary><table class="sh-tbl"><tbody>{"".join(rows)}</tbody></table></details>'
    )


def _hold_badge(a: dict, holdings: dict | None) -> str:
    """Show 🔹 持仓 +N% chip when this symbol is in current holdings."""
    if not holdings:
        return ""
    code = a["symbol"].split(".")[0]
    h = holdings.get(code)
    if not h:
        return ""
    pnl = h.get("floating_pnl_pct")
    pnl_s = ""
    if isinstance(pnl, (int, float)):
        pnl_pct = pnl * 100
        cls = "up" if pnl_pct >= 0 else "down"
        pnl_s = f' <em class="{cls}">{pnl_pct:+.1f}%</em>'
    return f'<span class="hold-tag" title="持仓中（来自 analysis.json）">🔹 持仓{pnl_s}</span>'


def _verdict_label(verdict: str, symbol: str, holdings: dict | None) -> str:
    """If user already holds this AND dashboard says 建仓, surface '补仓机会'."""
    if holdings and verdict == "建仓":
        code = symbol.split(".")[0]
        if code in holdings:
            return "补仓机会"
    return verdict


def card(a: dict, thesis: dict, holdings: dict | None = None) -> str:
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
    thesis_html = _thesis_html(th)
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
          {_hold_badge(a, holdings)}
          <span class="badge {cls}">{_verdict_label(a['verdict'], a['symbol'], holdings)}</span>
          <button class="exp-btn" onclick="toggleCard(this)" title="展开/收起详情">▾</button>
        </div>
      </header>
      <p class="vexpl">{vexpl} <span class="data-date" title="价格/技术信号数据截止日">· 数据 {a.get('date','—')}</span></p>
      <section class="grid">{fundamentals}</section>
      {_valh(a)}
      {_sotp(a)}
      {_shareholder_html(a)}
      <div class="card-deep">
        <section class="grid sig">{signal}</section>
        <section class="thesis">{thesis_html}</section>
      </div>
    </article>
    """


def _panel(market: str, rows: list[dict], active: bool, thesis: dict, holdings: dict | None = None) -> str:
    rows = sorted(rows, key=lambda a: (VERDICT_META[a["verdict"]][0], -a["close"]))
    on = " on" if active else ""
    if not rows:
        return (f'<section class="panel{on}" data-m="{market}">'
                f'<div class="empty">暂无标的 — 在 dashboard_config.json 的 baskets.{market} 里添加</div></section>')
    panel_date = max(a["date"] for a in rows)
    n_buy = sum(1 for a in rows if a["verdict"] == "建仓")
    header = (f'<div class="phead">本市场数据日期 {panel_date} · {len(rows)} 只'
              f' · 可建仓 <b>{n_buy}</b> 只</div>')
    cards = "\n".join(card(a, thesis, holdings) for a in rows)
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
    holdings = cfg.get("_holdings")
    panels = "".join(_panel(m, by_market.get(m, []), m == active_market, thesis, holdings) for m, _ in market_tabs)
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
  .data-date {{ color:#5a6273; font-size:11px; font-variant-numeric:tabular-nums;
                cursor:help; margin-left:4px; }}
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
  /* analyst-target range bar: shows dispersion (low..high) with current price marker */
  .rangebar {{ flex:1; position:relative; height:9px; background:linear-gradient(90deg,
              rgba(31,157,87,0.18), rgba(59,125,216,0.22), rgba(229,83,75,0.18));
              border-radius:5px; border:1px solid var(--line); }}
  .rb-cur {{ position:absolute; top:-3px; bottom:-3px; width:3px;
             background:#ffd54a; border-radius:2px;
             box-shadow:0 0 4px rgba(255,213,74,.7); }}
  .rb-mean {{ position:absolute; top:0; bottom:0; width:1px; background:var(--dim); opacity:.6; }}
  .vhv-rb {{ flex:0 0 auto; font-size:11.5px; font-variant-numeric:tabular-nums;
             padding:1px 7px; border-radius:10px; color:#fff; font-weight:600; }}
  .vhv-rb.buy {{ background:var(--buy); }} .vhv-rb.wait {{ background:var(--wait); }}
  .vhv-rb.watch {{ background:var(--watch); }} .vhv-rb.avoid {{ background:var(--avoid); }}
  .vh-meta {{ font-size:11px; color:var(--dim); padding:3px 0 0 78px;
              font-variant-numeric:tabular-nums; }}
  .vh-meta b {{ color:var(--txt); }}
  .vh-meta .rb-lo {{ color:var(--buy); }}
  .vh-meta .rb-hi {{ color:var(--up); }}
  .impl-tag {{ display:inline-block; font-size:9px; padding:1px 4px; border-radius:3px;
               background:var(--wait); color:#fff; vertical-align:middle; margin-left:3px; }}
  .hold-tag {{ display:inline-flex; align-items:center; font-size:11px;
               padding:1px 7px; border-radius:10px; background:rgba(31,157,87,0.15);
               border:1px solid var(--buy); color:#7be1a8; font-weight:500;
               white-space:nowrap; }}
  .hold-tag em {{ font-style:normal; margin-left:4px; font-variant-numeric:tabular-nums; }}
  .hold-tag em.up {{ color:#e5534b; }} .hold-tag em.down {{ color:#7be1a8; }}
  /* shareholder changes (CN only) */
  .shareholders {{ margin:9px 0 2px; border-top:1px dashed var(--line); padding-top:9px; }}
  .shareholders summary {{ cursor:pointer; list-style:none; }}
  .shareholders summary::-webkit-details-marker {{ display:none; }}
  .shareholders summary::before {{ content:"▸ "; color:var(--dim); font-size:10px; }}
  .shareholders[open] summary::before {{ content:"▾ "; }}
  .sh-x {{ color:var(--dim); font-weight:400; font-size:10.5px; margin-left:4px; }}
  .sh-buy {{ color:#e5534b; }} .sh-sell {{ color:#7be1a8; }}
  .sh-tbl {{ width:100%; margin-top:5px; font-size:11.5px; border-collapse:collapse;
             font-variant-numeric:tabular-nums; }}
  .sh-tbl td {{ padding:2px 4px; color:#c8cdd6; }}
  .sh-tbl .sh-d {{ color:var(--dim); width:80px; }}
  .sh-tbl .sh-n {{ }}
  .sh-tbl .sh-c {{ text-align:right; font-weight:500; }}
  .sh-tbl .sh-c.buy {{ color:#e5534b; }} .sh-tbl .sh-c.sell {{ color:#7be1a8; }}
  /* collapsible card: default collapse signal+thesis; click ▾ to expand */
  .card-deep {{ display:none; }}
  .card.open .card-deep {{ display:block; }}
  .exp-btn {{ appearance:none; cursor:pointer; background:transparent; color:var(--dim);
              border:1px solid var(--line); border-radius:5px; font-size:11px;
              padding:1px 6px; margin-left:6px; transition:transform .15s; }}
  .exp-btn:hover {{ color:var(--txt); border-color:#3a4150; }}
  .card.open .exp-btn {{ transform:rotate(180deg); color:var(--txt); }}
  .exp-all {{ font-size:11px; color:var(--dim); cursor:pointer; margin-left:8px;
              padding:2px 8px; border:1px solid var(--line); border-radius:5px;
              background:transparent; user-select:none; }}
  .exp-all:hover {{ color:var(--txt); }}
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
  /* narrative-style thesis (new schema) */
  .narr p {{ margin:0 0 9px; font-size:12.5px; line-height:1.7; color:#d4d8e0; text-indent:2em; }}
  .narr p:last-child {{ margin-bottom:0; }}
  .nc {{ margin-top:10px; padding:9px 11px; background:rgba(59,125,216,0.08);
         border-left:3px solid var(--watch); border-radius:4px; }}
  .nc .bb {{ font-size:12px; line-height:1.65; color:#d4d8e0; margin-top:4px; }}
  .cat, .exit {{ margin-top:10px; }}
  .cat summary, .exit summary {{ cursor:pointer; list-style:none; user-select:none; }}
  .cat summary::-webkit-details-marker, .exit summary::-webkit-details-marker {{ display:none; }}
  .cat summary::before, .exit summary::before {{ content:"▸ "; color:var(--dim); font-size:10px; }}
  .cat[open] summary::before, .exit[open] summary::before {{ content:"▾ "; }}
  .cat ul {{ margin:6px 0 0; padding-left:18px; font-size:12px; line-height:1.65; color:#c8cdd6; }}
  .cat li {{ margin-bottom:3px; }}
  .cat li b {{ color:var(--watch); font-weight:600; }}
  .cat .dim {{ color:var(--dim); }}
  .cat li.past {{ opacity:.5; }}
  .cat li.past b {{ color:var(--dim); text-decoration:line-through; }}
  .eg {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:6px; }}
  .ec b {{ display:block; font-size:11.5px; color:var(--wait); margin-bottom:4px; }}
  .ec ul {{ margin:0; padding-left:16px; font-size:11.5px; line-height:1.55; color:#c8cdd6; }}
  .ec li {{ margin-bottom:3px; }}
  .bh {{ font-size:11.5px; font-weight:600; color:var(--txt); letter-spacing:.3px; }}
  /* glossary tooltip (hover over PE/PB/RSI/etc to see plain-language explanation) */
  .g {{ position:relative; cursor:help; border-bottom:1px dotted #4a5160; }}
  .g:hover::after {{
    content:attr(data-tip);
    position:absolute; left:50%; bottom:calc(100% + 6px); transform:translateX(-50%);
    background:#1c2029; border:1px solid #3a4150; border-radius:6px;
    padding:8px 11px; width:260px; font-size:11.5px; font-weight:400;
    line-height:1.55; color:#e6e8ec; letter-spacing:0;
    white-space:normal; text-align:left; z-index:200;
    box-shadow:0 4px 18px rgba(0,0,0,.5);
  }}
  .g:hover::before {{
    content:""; position:absolute; left:50%; bottom:calc(100% + 1px);
    transform:translateX(-50%); width:0; height:0;
    border:5px solid transparent; border-top-color:#3a4150; z-index:201;
  }}
  footer {{ color:var(--dim); font-size:11px; padding:0 22px 30px; max-width:1500px; }}
</style></head>
<body>
  <div class="top">
    <h1>自选监控 · 基本面 &amp; 信号</h1>
    <div class="sub">生成于 {ts} · 共 {total} 只 · 全市场可建仓 <b>{n_buy}</b> 只
      &nbsp;|&nbsp; 入场规则：突破未越追高线=建仓，缩量阴跌=不碰</div>
    <div class="tabs">{tabs}<button class="exp-all" onclick="toggleAllCards()">展开全部 ▾</button></div>
  </div>
  <main>{panels}</main>
  <footer>说明：PE/PB/股息率/市值等为数据源(Longbridge)实时取数；触发/阶段/止损为生产 radar 逻辑计算；
    基本面分析为定性整理，仅供参考、请独立核实。结论基于已回测的入场规则，不构成投资建议。</footer>
  <script>
  function showTab(m){{
    document.querySelectorAll('.tab').forEach(function(t){{ t.classList.toggle('on', t.dataset.m===m); }});
    document.querySelectorAll('.panel').forEach(function(p){{ p.classList.toggle('on', p.dataset.m===m); }});
  }}
  function toggleCard(btn){{
    btn.closest('.card').classList.toggle('open');
  }}
  function toggleAllCards(){{
    var cards = document.querySelectorAll('.panel.on .card');
    var anyClosed = Array.from(cards).some(function(c){{ return !c.classList.contains('open'); }});
    cards.forEach(function(c){{ c.classList.toggle('open', anyClosed); }});
    var btn = document.querySelector('.exp-all');
    if (btn) btn.textContent = anyClosed ? '收起全部 ▴' : '展开全部 ▾';
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
    cfg["_holdings"] = load_holdings()
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
