"""Build a standalone HTML dashboard for the HK watchlist.

For each name it pulls live valuation (Longbridge) + the current technical
signal (production radar logic), derives a 建仓/等回踩/观察/不碰 verdict from the
entry-timing rules validated in scripts/entry_timing_backtest.py, and renders a
single self-contained HTML file (inline CSS, no external deps).

Qualitative 基本面分析 is authored here (knowledge-based — verify independently);
all numeric fields are live from the data provider.

Run:  python scripts/watchlist_dashboard.py    ->  outputs/watchlist_dashboard.html
"""
from __future__ import annotations

import re
import sys
from datetime import date, datetime
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_workbench.data_sources.indicators import add_technical_indicators
from research_workbench.data_sources.longbridge import _run_longbridge, get_provider
from research_workbench.signal_engine.radar import (
    DEFAULT_RADAR_CONFIG,
    add_radar_features,
    detect_trigger_type,
)

OUT = ROOT / "outputs" / "watchlist_dashboard.html"
K_ATR = 1.0  # 不追高: extended if close > 20d-high + K_ATR*ATR
CNY_PER_HKD = 0.92  # one constant; segment revenue reports CNY, mktcap trades HKD

# Per-market baskets. Symbol format follows Longbridge: <CODE>.<MARKET>.
BASKETS: dict[str, list[tuple[str, str]]] = {
    "HK": [
        ("700.HK", "腾讯"), ("9988.HK", "阿里巴巴"), ("9992.HK", "泡泡玛特"),
        ("1024.HK", "快手"), ("9999.HK", "网易"), ("9961.HK", "携程"),
        ("1810.HK", "小米"), ("3690.HK", "美团"),
    ],
    "CN": [
        ("002624.SZ", "完美世界"), ("002241.SZ", "歌尔股份"), ("002602.SZ", "世纪华通"),
    ],
    "US": [
        ("RDDT.US", "Reddit"), ("META.US", "Meta"), ("GOOGL.US", "谷歌"),
        ("AAPL.US", "苹果"), ("AMZN.US", "亚马逊"), ("MSFT.US", "微软"),
    ],
}
MARKET_TABS = [("HK", "港股"), ("CN", "A股"), ("US", "美股")]  # display order

# Qualitative thesis — authored, not live. Keep concise & balanced.
THESIS: dict[str, dict[str, str]] = {
    "700.HK": {
        "业务": "微信社交生态为核心，叠加游戏、广告、金融科技与云。现金流与用户黏性行业最强。",
        "多头": "游戏常青+新游储备、视频号广告变现仍在爬坡、持续大额回购注销。",
        "空头": "游戏版号与监管、宏观消费疲软压制广告与支付，体量大增速放缓。",
        "估值": "PE 处历史偏低区间，回购提供安全垫；属“稳健但弹性有限”的底仓型。",
    },
    "9988.HK": {
        "业务": "淘天电商基本盘 + 阿里云 + 国际电商 + 菜鸟物流。",
        "多头": "云收入受 AI 拉动重回双位数增长、电商份额企稳、回购与分拆释放价值。",
        "空头": "拼多多/抖音电商分流、消费力弱、组织调整执行风险。",
        "估值": "电商按现金牛、云按 AI 期权定价；低 PE 反映增长担忧，反转看云。",
    },
    "9992.HK": {
        "业务": "IP 潮玩龙头，Labubu/THE MONSTERS 等自有 IP，毛利极高、出海加速。",
        "多头": "Labubu 全球破圈、海外门店与线上高速放量、单 IP 货币化能力强。",
        "空头": "IP 有生命周期、潮流情绪驱动、高估值对增长容错低、二级市场炒作风险。",
        "估值": "成长股估值，PB 高企；定价已计入高增长，预期差在“IP 能否持续造血”。",
    },
    "1024.HK": {
        "业务": "短视频+直播，直播电商 GMV 与广告为主要变现，海外(Kwai)仍在投入。",
        "多头": "电商货币化率提升、经营利润转正、海外减亏。",
        "空头": "抖音正面竞争、国内用户增长见顶、海外盈利路径长。",
        "估值": "利润刚转正，按 GMV/利润切换定价；弹性来自变现率，风险在竞争。",
    },
    "9999.HK": {
        "业务": "自研游戏为主(端游+手游)，叠加有道、云音乐；游戏利润率行业领先。",
        "多头": "强自研管线、老游戏长线运营、出海与高分红。",
        "空头": "暴雪合作等外部变数、新游成败不确定、买量成本。",
        "估值": "现金充裕+高分红，PE 不贵；属攻守均衡，催化看新游流水。",
    },
    "9961.HK": {
        "业务": "在线旅游 OTA 龙头，国内+出境+国际 Trip.com 三轮驱动。",
        "多头": "出境游持续修复、国际化变现、行业集中度高。",
        "空头": "宏观旅游消费波动、汇率、平台竞争与佣金压力。",
        "估值": "复苏+国际化双逻辑，估值随消费预期波动；周期与成长兼具。",
    },
    "1810.HK": {
        "业务": "手机+AIoT+互联网服务，新增汽车(SU7/YU7)；“人车家”生态闭环。",
        "多头": "汽车交付放量+口碑、高端化提升 ASP、生态协同与海外。",
        "空头": "汽车盈利与产能爬坡、手机红海竞争、汽车重资产拖累利润。",
        "估值": "估值含汽车成长期权，弹性最大也最贵；定价跟随汽车交付节奏。",
    },
    "3690.HK": {
        "业务": "本地生活龙头：外卖+到店酒旅+即时零售(美团闪购)+优选。",
        "多头": "外卖高频流量入口、即时零售长坡厚雪、到店壁垒。",
        "空头": "抖音到店竞争、外卖补贴战压制利润、新业务持续亏损。",
        "估值": "核心本地商业现金牛 vs 新业务投入；利润率受竞争与补贴节奏主导。",
    },
    # ---- A股 ----
    "002624.SZ": {
        "业务": "游戏(MMO老游+新游《异环》)为主，叠加影视；端游手游双线。",
        "多头": "《异环》开放世界新品周期、老游戏长线现金流、研发底蕴。",
        "空头": "新游成败不确定、版号与买量成本、影视拖累、行业竞争激烈。",
        "估值": "成败系于新游流水兑现；分部看老游戏给低 PE、新游给期权估值。",
    },
    "002241.SZ": {
        "业务": "消费电子精密制造：声学(扬声器/麦克风)+VR/AR 整机代工+精密零组件。",
        "多头": "Meta/苹果 VR/AR 供应链卡位、AI 硬件与可穿戴放量、份额提升。",
        "空头": "大客户集中(砍单风险)、消费电子周期、代工毛利薄、地缘扰动。",
        "估值": "随大客户新品周期与 VR/AR 渗透率波动；弹性来自整机代工放量。",
    },
    "002602.SZ": {
        "业务": "游戏(页游/手游，盛趣)+数据中心(IDC)双主业。",
        "多头": "存量游戏稳定现金流、数据中心受 AI 算力需求拉动、出海。",
        "空头": "老游戏自然衰减、新游乏力、IDC 资本开支重、商誉风险。",
        "估值": "游戏现金牛支撑底部、IDC 给算力期权；定价取决于新游与算力兑现。",
    },
    # ---- 美股 ----
    "RDDT.US": {
        "业务": "全球最大兴趣社区平台，变现靠广告 + 数据授权(供 AI 训练)。",
        "多头": "广告变现率仍低、海外与本地化空间、独家语料的数据授权溢价。",
        "空头": "社区调性与商业化的张力、流量依赖搜索、盈利刚起步估值高。",
        "估值": "高成长高估值，定价已计入广告加速；预期差在变现率与数据收入持续性。",
    },
    "META.US": {
        "业务": "广告双寡头(FB/IG/WhatsApp)，叠加 AI 与 Reality Labs(VR/AR)。",
        "多头": "AI 驱动广告 ROI 与时长提升、开源 Llama 生态、回购与降本增效。",
        "空头": "Reality Labs 持续巨亏、资本开支(AI 算力)激增、监管与隐私政策。",
        "估值": "核心广告现金牛支撑，PE 合理；弹性看 AI 变现 vs RL 烧钱节奏。",
    },
    "GOOGL.US": {
        "业务": "搜索广告基本盘 + YouTube + 谷歌云 + Android/Chrome 生态。",
        "多头": "云加速盈利、Gemini 全栈 AI、自研 TPU 成本优势、分发渠道垄断。",
        "空头": "AI 问答冲击搜索入口、反垄断拆分风险、云竞争、资本开支。",
        "估值": "大盘股里相对不贵；核心争议是 AI 对搜索是颠覆还是增量。",
    },
    "AAPL.US": {
        "业务": "iPhone 硬件为核心 + 高毛利服务(App Store/订阅)生态闭环。",
        "多头": "服务收入高增长高毛利、换机周期与高端化、生态黏性与回购。",
        "空头": "iPhone 增长见顶、AI(Apple Intelligence)落后、中国份额与监管。",
        "估值": "估值偏贵且增长平、靠服务与回购支撑；AI 叙事弱于同行。",
    },
    "AMZN.US": {
        "业务": "电商零售 + AWS 云(利润主来源) + 高增长广告业务。",
        "多头": "AWS 重回加速、广告高毛利放量、零售利润率改善、物流杠杆。",
        "空头": "AWS 面对 Azure/GCP 竞争、零售周期、AI 资本开支、监管。",
        "估值": "利润弹性来自 AWS 与广告；按利润率改善路径定价，云增速是关键变量。",
    },
    "MSFT.US": {
        "业务": "Azure 云 + Office/Copilot 生产力 + Windows + 游戏(动视暴雪)。",
        "多头": "AI(Copilot+OpenAI)全面变现、Azure 份额提升、订阅高黏性高毛利。",
        "空头": "AI 资本开支拖累自由现金流、OpenAI 关系不确定、估值偏高。",
        "估值": "AI 龙头溢价，PE 高于历史；定价已计入 Copilot 变现，看兑现速度。",
    },
}

VERDICT_META = {  # label -> (order, css-class, 中文释义)
    "建仓":   (0, "buy",   "突破达标且未越追高线，右侧确认机会"),
    "等回踩": (1, "wait",  "已突破但冲高过头，挂回踩单别追"),
    "观察":   (2, "watch", "接近突破、结构健康，等放量触发"),
    "回踩-谨慎": (2, "watch", "回踩信号，历史上偏弱，谨慎"),
    "不碰":   (3, "avoid", "下降趋势无信号，接飞刀区"),
}

# 分部估值 (SOTP) — only meaningful for multi-business conglomerates. Each segment:
# 营收(自动·CNY) × ps(手填倍数) = 估值. PS ≈ 段净利率 × 合理PE. `extra` lines are
# off-revenue value (投资组合/净现金, 亿CNY) that pure-revenue SOTP misses. Edit freely.
SEGMENTS: dict[str, dict] = {
    "700.HK": {  # 腾讯 (报 CNY)
        "segments": [
            {"match": ["增值服务"], "label": "增值服务(游戏+社交)", "ps": 7.0, "reason": "净利率~38%×PE18 — 游戏现金牛+视频号弹性，确定性高"},
            {"match": ["营销服务", "网络广告"], "label": "营销服务(广告)", "ps": 7.0, "reason": "净利率~35%×PE20 — 视频号广告高增长、高利润"},
            {"match": ["金融科技"], "label": "金融科技及企业服务", "ps": 3.5, "reason": "净利率~18%×PE20 — 支付薄利、云刚转盈"},
            {"match": ["其他"], "label": "其他", "ps": 1.0, "reason": "杂项，保守给1×"},
        ],
        "extra": [{"label": "投资组合(NAV×0.65)", "value_cny_yi": 5800, "reason": "上市+非上市股权按净值打折，营收口径不含，不加严重低估"}],
    },
    "9988.HK": {  # 阿里 (报 CNY)
        "segments": [
            {"match": ["China E-commerce", "中国电商", "淘天"], "label": "淘天(中国电商)", "ps": 3.0, "reason": "净利率~30%×PE10 — 慢增长现金牛"},
            {"match": ["云"], "label": "云智能", "ps": 4.5, "reason": "AI重估，直接对标全球云给PS"},
            {"match": ["AIDC", "国际"], "label": "AIDC国际", "ps": 1.0, "reason": "高增长但亏/薄利"},
            {"match": ["所有其他", "其他"], "label": "所有其他", "ps": 0.8, "reason": "菜鸟/本地/大文娱，薄利或亏"},
        ],
        "extra": [{"label": "净现金+投资(蚂蚁等)", "value_cny_yi": 4500, "reason": "巨额净现金+联营投资按折扣计入"}],
    },
    "3690.HK": {  # 美团 (报 CNY)
        "segments": [
            {"match": ["核心本地"], "label": "核心本地商业", "ps": 2.0, "reason": "经营利润率~19%×被竞争压制的PE~10-11 — 抖音到店+补贴战压制估值，校准到现价市值+机构目标"},
            {"match": ["新举措", "新业务"], "label": "新举措", "ps": 0.8, "reason": "仍亏但收窄，Keeta海外期权，按营收小倍数"},
        ],
    },
    "1810.HK": {  # 小米 (报 CNY)
        "segments": [
            {"match": ["智能手机", "手机"], "label": "智能手机", "ps": 0.8, "reason": "净利率~5%×PE15 — 低毛利硬件"},
            {"match": ["IoT"], "label": "IoT与生活消费", "ps": 1.0, "reason": "净利率~6%×PE15 — 略好的硬件"},
            {"match": ["互联网服务"], "label": "互联网服务", "ps": 5.5, "reason": "净利率~50%×PE18 — 广告/游戏，真利润中心"},
            {"match": ["EV", "汽车"], "label": "Smart EV+AI", "ps": 3.5, "reason": "⚠争议最大：当前亏，市场按销量爬坡+特斯拉对标给高估值，取3.5贴近市场(可3-4浮动)"},
            {"match": ["其他"], "label": "其他", "ps": 1.0, "reason": "杂项"},
        ],
    },
}


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _valuation(symbol: str) -> dict:
    """PE percentile (cheaper-than-X%-of-history) + ready-made 中文结论."""
    d = _safe(lambda: _run_longbridge(["valuation", symbol]), {}) or {}
    pe = d.get("history", {}).get("metrics", {}).get("pe", {})
    vals = [f for f in (_fnum(x.get("value")) for x in pe.get("list", [])) if f and f > 0]
    cur = vals[-1] if vals else None
    cheap_pct = (sum(1 for v in vals if v > cur) / len(vals) * 100) if (vals and cur) else None
    summary = re.sub("<[^>]+>", "", d.get("overview", {}).get("ai_summary", "")).strip()
    return {"pe_cheap_pct": cheap_pct, "val_summary": summary}


def _ratings(symbol: str) -> dict:
    """Consensus target price + analyst buy/hold/sell distribution."""
    d = _safe(lambda: _run_longbridge(["institution-rating", symbol]), {}) or {}
    ins = d.get("instratings", {})
    ev = ins.get("evaluate", {}) or {}
    return {
        "target": _fnum(ins.get("target")), "ccy": ins.get("ccy_symbol", ""),
        "recommend": ins.get("recommend", ""),
        "sb": ev.get("strong_buy"), "buy": ev.get("buy"),
        "hold": ev.get("hold"), "sell": ev.get("sell"),
    }


def _seg_annual(symbol: str) -> list[tuple[str, float]]:
    """Latest full-year segment revenue (CNY), current segment names."""
    d = _safe(lambda: _run_longbridge(["business-segments", symbol, "--history", "--report", "af"]), {}) or {}
    hist = d.get("historical") or []
    if not hist:
        return []
    return [(b.get("name", ""), _fnum(b.get("value")))
            for b in hist[-1].get("business", []) if _fnum(b.get("value"))]


def _match_seg(name: str, segments: list[dict]) -> dict | None:
    for s in segments:
        if any(kw.lower() in name.lower() for kw in s["match"]):
            return s
    return None


def build_sotp(symbol: str, mktcap_hkd: float | None, close: float) -> dict | None:
    cfg = SEGMENTS.get(symbol)
    if not cfg:
        return None
    rows: list[dict] = []
    total_cny = 0.0
    for name, rev in _seg_annual(symbol):
        m = _match_seg(name, cfg["segments"])
        if not m:
            continue  # skip 抵消/未分摊
        val = rev * m["ps"]
        total_cny += val
        rows.append({"label": m["label"], "rev_yi": rev / 1e8, "ps": m["ps"],
                     "val_yi": val / 1e8, "reason": m["reason"]})
    if not rows:
        return None
    for ex in cfg.get("extra", []):
        total_cny += ex["value_cny_yi"] * 1e8
        rows.append({"label": ex["label"], "rev_yi": None, "ps": None,
                     "val_yi": ex["value_cny_yi"], "reason": ex["reason"]})
    target_cny_yi = total_cny / 1e8
    target_hkd_yi = target_cny_yi / CNY_PER_HKD
    mktcap_hkd_yi = mktcap_hkd / 1e8 if isinstance(mktcap_hkd, (int, float)) and pd.notna(mktcap_hkd) else None
    upside = (target_hkd_yi / mktcap_hkd_yi - 1) * 100 if mktcap_hkd_yi else None
    for r in rows:
        r["share"] = r["val_yi"] / target_cny_yi * 100 if target_cny_yi else None
    return {"rows": rows, "target_hkd_yi": target_hkd_yi,
            "mktcap_hkd_yi": mktcap_hkd_yi, "upside": upside}


def analyze(symbol: str, market: str = "HK") -> dict:
    prov = get_provider(market)
    raw = prov.fetch_history(symbol, date(2022, 1, 1), date.today())
    df = add_radar_features(add_technical_indicators(raw)).reset_index(drop=True)
    r = df.iloc[-1]
    c = float(r["close"]); ma20 = float(r["ma20"]); ma50 = float(r["ma50"]); ma200 = float(r["ma200"])
    h20 = float(r["high_20d"]); atr = float(r["atr14"]); vs = float(r["volume_spike_ratio"])
    rsi = float(r["rsi14"]); dd = float(r["drawdown_60d"])
    prev = float(df.iloc[-2]["close"])
    chg1 = (c / prev - 1) * 100
    chg5 = (c / float(df.iloc[-6]["close"]) - 1) * 100 if len(df) >= 6 else float("nan")
    trig = detect_trigger_type(r, DEFAULT_RADAR_CONFIG["triggers"])
    up = c >= ma200
    dist_high = (c / h20 - 1) * 100
    cap = h20 + K_ATR * atr

    if trig == "breakout":
        verdict = "建仓" if c <= cap else "等回踩"
        stop = h20
    elif trig == "pullback":
        verdict = "回踩-谨慎"
        stop = min(ma20, h20 * 0.97)
    elif (c >= ma20) and (dist_high >= -5.0):
        verdict = "观察"
        stop = ma20
    else:
        verdict = "不碰"
        stop = ma20
    risk_pct = (c - stop) / c * 100 if stop and c > stop else None

    fund = prov.fetch_fundamentals(symbol)
    mktcap = fund.get("market_cap")
    val = _valuation(symbol)
    rate = _ratings(symbol)
    target = rate.get("target")
    target_upside = (target / c - 1) * 100 if target and c else None
    return {
        "symbol": symbol, "name": fund.get("name") or symbol, "market": market,
        "date": r["date"].strftime("%Y-%m-%d"),
        "close": c, "chg1": chg1, "chg5": chg5, "verdict": verdict,
        "trigger": {"breakout": "突破", "pullback": "回踩"}.get(trig, "无"),
        "regime": "上升" if up else "下降",
        "rsi": rsi, "vol_spike": vs, "dist_high": dist_high,
        "ma20": ma20, "ma50": ma50, "ma200": ma200, "vs_ma200": (c / ma200 - 1) * 100,
        "drawdown": dd * 100, "stop": stop, "risk_pct": risk_pct, "cap": cap, "high20": h20,
        "pe": fund.get("end_pe"), "pb": fund.get("current_pb"),
        "eps_ttm": fund.get("eps_ttm"), "div_yield": fund.get("dividend_yield"),
        "mktcap": mktcap, "turnover": fund.get("turnover_rate"),
        "pe_cheap_pct": val.get("pe_cheap_pct"), "val_summary": val.get("val_summary"),
        "target": target, "target_ccy": rate.get("ccy"), "target_upside": target_upside,
        "recommend": rate.get("recommend"),
        "rate_sb": rate.get("sb"), "rate_buy": rate.get("buy"),
        "rate_hold": rate.get("hold"), "rate_sell": rate.get("sell"),
        "sotp": build_sotp(symbol, mktcap, c),
    }


# ---------- HTML rendering ----------

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
        ps = f'×{r["ps"]:.1f}' if r.get("ps") else "—"
        share = f'{r["share"]:.0f}%' if r.get("share") is not None else ""
        body.append(
            f'<tr title="{tip}"><td class="sl">{escape(r["label"])}</td>'
            f'<td>{rev}</td><td class="sp">{ps}</td>'
            f'<td class="sv">{r["val_yi"]:.0f}亿</td><td class="ss">{share}</td></tr>'
        )
    up = s.get("upside")
    tgt = s.get("target_hkd_yi"); mc = s.get("mktcap_hkd_yi")

    def _wan(v):
        return f"{v/1e4:.2f}万亿" if v and v >= 1e4 else (f"{v:.0f}亿" if v else "—")

    if isinstance(up, (int, float)) and pd.notna(up):
        up_cls = "buy" if up >= 0 else "avoid"
        badge = f'<span class="badge {up_cls}">隐含 {up:+.0f}%</span>'
        w = max(0, min(100, 50 + up / 2))  # center 50%, ±100% spans the bar
        bar = f'<div class="bar sotpbar"><div class="bf {up_cls}" style="width:{w:.0f}%"></div></div>'
    else:
        badge, bar = "", ""
    return (
        '<section class="sotp"><div class="sh">分部估值 (SOTP) '
        '<span class="sh-x">营收×PS加总 ÷ 0.92汇率 vs 现价市值</span></div>'
        f'<table class="st"><tbody>{"".join(body)}</tbody></table>'
        f'<div class="sf"><span>目标 <b>{_wan(tgt)}</b> · 现价 {_wan(mc)}</span>{badge}</div>{bar}</section>'
    )


def card(a: dict) -> str:
    order, cls, vexpl = VERDICT_META[a["verdict"]]
    th = THESIS.get(a["symbol"], {})
    chg1 = a["chg1"]; chg_cls = "up" if chg1 >= 0 else "down"
    fundamentals = "".join([
        _metric("PE", _num(a["pe"])), _metric("PB", _num(a["pb"])),
        _metric("EPS(TTM)", _num(a["eps_ttm"])),
        _metric("股息率", _num(a["div_yield"]*100, "{:.2f}%") if isinstance(a["div_yield"], (int,float)) and pd.notna(a["div_yield"]) else "—"),
        _metric("市值", _cap_str(a["mktcap"])), _metric("换手率", _num(a["turnover"], "{:.2f}%")),
    ])
    signal = "".join([
        _metric("触发", a["trigger"]), _metric("阶段", a["regime"]),
        _metric("RSI", _num(a["rsi"], "{:.0f}")), _metric("量比", _num(a["vol_spike"], "{:.2f}x")),
        _metric("距20日高", _num(a["dist_high"], "{:+.1f}%")),
        _metric("vs MA200", _num(a["vs_ma200"], "{:+.1f}%")),
        _metric("60日回撤", _num(a["drawdown"], "{:.1f}%")),
        _metric("止损位", f'{_num(a["stop"])}' + (f' (风险{_num(a["risk_pct"],"{:.1f}%")})' if a["risk_pct"] else "")),
    ])
    thesis = "".join(
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
      <section class="thesis">{thesis}</section>
    </article>
    """


def _panel(market: str, rows: list[dict], active: bool) -> str:
    rows = sorted(rows, key=lambda a: (VERDICT_META[a["verdict"]][0], -a["close"]))
    on = " on" if active else ""
    if not rows:
        return (f'<section class="panel{on}" data-m="{market}">'
                f'<div class="empty">暂无标的 — 在 BASKETS["{market}"] 里添加</div></section>')
    cards = "\n".join(card(a) for a in rows)
    return f'<section class="panel{on}" data-m="{market}"><div class="wrap">{cards}</div></section>'


def render(by_market: dict[str, list[dict]]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_rows = [a for rs in by_market.values() for a in rs]
    total = len(all_rows)
    n_buy = sum(1 for a in all_rows if a["verdict"] == "建仓")
    data_date = max((a["date"] for a in all_rows), default="—")
    # first tab with data is active by default
    active_market = next((m for m, _ in MARKET_TABS if by_market.get(m)), MARKET_TABS[0][0])

    tab_parts = []
    for m, label in MARKET_TABS:
        rs = by_market.get(m, [])
        nb = sum(1 for a in rs if a["verdict"] == "建仓")
        dot = f'<b class="dot">{nb}</b>' if nb else ""
        on = " on" if m == active_market else ""
        tab_parts.append(
            f'<button class="tab{on}" data-m="{m}" onclick="showTab(\'{m}\')">'
            f'{label}<i>{len(rs)}</i>{dot}</button>'
        )
    tabs = "".join(tab_parts)
    panels = "".join(_panel(m, by_market.get(m, []), m == active_market) for m, _ in MARKET_TABS)
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
  .empty {{ color:var(--dim); padding:40px 22px; font-size:13px; }}
  .wrap {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr));
           gap:14px; padding:18px 22px 40px; max-width:1500px; }}
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
    <div class="sub">数据日期 {data_date} · 生成于 {ts} · 共 {total} 只 · 全市场可建仓 <b>{n_buy}</b> 只
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


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    by_market: dict[str, list[dict]] = {}
    for market, label in MARKET_TABS:
        basket = BASKETS.get(market, [])
        if not basket:
            continue
        print(f"== {label} ({market}) ==")
        rows: list[dict] = []
        for sym, name in basket:
            try:
                rows.append(analyze(sym, market))
                print(f"  ✓ {name} {sym}: {rows[-1]['verdict']}")
            except Exception as e:
                print(f"  ✗ {name} {sym}: {type(e).__name__}: {str(e)[:80]}")
        by_market[market] = rows
    OUT.write_text(render(by_market), encoding="utf-8")
    print(f"\n已生成: {OUT}")
