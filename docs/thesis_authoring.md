# Equity Thesis Authoring Guide

The `analysis.equity_thesis` module produces a structured 6-step investment
thesis per symbol — catalyst, peer anchoring, quality, SOTP valuation,
sentiment, action — and renders it to Markdown. Data-driven fields auto-fill
from the existing pipeline; the rest are filled by adding a `thesis` block to
each watchlist entry.

The framework abstracts the analysis style of catalyst-anchored, peer-
bucketed analyst notes. Each step has a clear source of truth:

| Step | Auto-filled from | Manual override |
|---|---|---|
| ① Catalyst | `earnings_express.announce_date` | `catalyst_type`, `catalyst_date`, `catalyst_summary` |
| ② Peer anchoring | `earnings_express.industry`, `industry_peer_pe_median` | `peer_bucket`, `peer_success`, `peer_failure` |
| ③ Quality | — | `quality_factors[]` |
| ④ SOTP | `business_segments` (latest annual, 按产品分类) | `segment_pe_multiples` per segment, or `manual_segments[]` for forward-looking |
| ⑤ Sentiment | `northbound_holdings` (30d sum), `capital_flow_history` (today) | `sentiment_notes` |
| ⑥ Action | `history` (latest close), `summary.market_cap` (upside calc) | `action`, `thesis_notes` |

## Watchlist Schema (per-item)

Add an optional `thesis` object to each entry in `research_inputs/watchlist_cn.json`:

```json
{
  "symbol": "002624.SZ",
  "name": "完美世界",
  "market": "CN",
  "sector": "传媒游戏",
  "status": "watch",
  "thesis": {
    "catalyst_type": "product_launch",
    "catalyst_date": "2026-04-23",
    "catalyst_summary": "新游戏《异环》正式上线",
    "peer_bucket": "50亿俱乐部",
    "peer_success": ["原神", "绝区零", "崩坏铁道", "火影忍者"],
    "peer_failure": ["王者荣耀世界(同标签下失败案例)"],
    "quality_factors": [
      {"description": "内容扎实/团队向心力强,大世界填充度高",
       "evidence": "苏州团队效率,横向对比王者荣耀世界",
       "direction": 1},
      {"description": "大世界天然拉 DAU + 降付费深度",
       "evidence": "GTA 风格 + 都市单元轻喜剧定位",
       "direction": 1}
    ],
    "manual_segments": [
      {"name": "老游戏(诛仙等 MMO)", "profit": 750000000,
       "pe_multiple": 15, "note": "时代眼泪,7-8亿利润已是乐观水平"},
      {"name": "新游戏(异环)", "profit": 2500000000,
       "pe_multiple": 15, "note": "异环预期 50 亿流水 × ~50% 净利率"},
      {"name": "影视业务", "profit": 0,
       "pe_multiple": 0, "note": "活人微死,不亏即可"}
    ],
    "action": "accumulate",
    "thesis_notes": "短期博弈窗口:开盘暴跌即量化做空,补仓;盘中等待 Taptap 舆情持平。"
  }
}
```

## Two Modes for SOTP (Step ④)

**Auto mode** — use `segment_pe_multiples` (a `{segment_name: pe_multiple}` map).
The pipeline pulls actual current segments from `business_segments.csv` (latest
fiscal year, 按产品分类) and multiplies each segment's profit by your supplied
PE. Good for steady-state companies where current segment profits are
representative.

```json
"segment_pe_multiples": {
  "PC端网络游戏": 12,
  "移动网络游戏": 15,
  "电视剧及短剧": 0,
  "主机游戏": 20
}
```

**Manual mode** — use `manual_segments` (a list of dicts with `name` / `profit`
/ `pe_multiple` / optional `note`). Bypasses the historical breakdown
entirely; use this for **forward-looking valuations**, especially when a new
business line isn't yet in the historical data (e.g. a product launching this
year).

When both are supplied, `manual_segments` wins.

## Where the Markdown Reports Land

`scripts/refresh.py` writes one Markdown file per symbol to
`outputs/research_data/thesis_reports/<SYMBOL>.md`. Files are overwritten on
each refresh; the thesis CSV history (TBD) will keep prior versions.

## Data Quality Indicator

Each report ends with a `⚠️ 数据缺失:...` line listing inputs that weren't
available (e.g. `peer_pe_median, capital_flow`). When push2 is flapping or
AKTools isn't running, expect this list to grow — the report still renders.

## Calibration Reference

The 002624.SZ thesis above (modeled on a published analyst note from 4-23)
produced an SOTP of **487.5 亿** vs the article's hand-calculated range
of **450-525 亿** — a tight match. The pipeline's contribution beyond what
a human did by hand:

- Auto-pulled current price (14.52), market cap (281.7 亿), upside (+73%).
- Auto-pulled 30-day northbound net buy.
- Documented exactly which inputs are missing for re-audit.
