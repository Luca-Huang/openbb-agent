# Dashboard / 择时 优化待办 (backlog)

> 记录于 2026-06-02，触发自给联想 0992.HK 加 SOTP + basket 后的一次复盘。
> 仅记录，未动代码。优先级：★硬伤 > 高 > 低。

## A. 分析准确性

### 1. verdict 缺中期超买惩罚 ★硬伤
- **位置**：`scripts/watchlist_dashboard.py` → `compute_verdict()` (~L259) 与 `analyze()` (~L279)
- **问题**：无 trigger 时只要 `close≥ma20 且 dist_high≥-5%` 就给「观察」；`dist_high` 只基于 `high_20d`（20 日），看不到中期拉伸。结果：联想「距 MA200 +135%、2 个月翻 3 倍、RSI 爆表」与一只刚温和站上 MA20 的票拿到同一个「观察」。
- **建议**：加一档「超买-勿追」——`vs_ma200 > 阈值` 或 `rsi14 > 80` 时强制降级。`vs_ma200`、`rsi` 字段 `analyze()` 已算出，verdict 没用上。改动小、应配单测。

### 2. SOTP 单点估值、对低利润硬件天然偏高 · 高
- **位置**：`build_sotp()` (~L175)，config `research_inputs/dashboard_config.json` 的 `segments`
- **问题**：「营收×PS」用在 IDG/ISG 这类 ~5% 净利率硬件上，倍数微动即放大失真；联想 SSG 一段(2.5×)贡献目标值 42%，整张表被一个主观倍数主导。输出只有单点（如 +48%），无区间。
- **建议**：(a) 出 PS 敏感性区间（各段 ±20% → upside 上下界）；(b) 或把 IDG/ISG 切到「利润×PE」(manual 路径，代码已支持)。

### 3. SOTP upside 与分析师共识反向并列、无协调 · 高
- **位置**：`analyze()` 同时产出 `sotp.upside` 与 `target_upside`；渲染在卡片
- **问题**：联想 SOTP +48% vs 共识目标 ~-27%，方向相反却无任何口径提示，易误读。
- **建议**：并列时加口径差异说明，或让 verdict 在二者打架时自动偏保守。

## B. 工具复用（对应 entry-timing initiative）

### 4. 回测脚本不可复用 · 高
- **位置**：`scripts/entry_timing_backtest.py`，`MARKETS` 写死、无 argparse
- **问题**：篮子硬编码 8 只；本次跑联想是 `import run()` 绕过。新标的无法一键回测。
- **建议**：加 CLI（`--symbol` / `--basket-from-config`），直接吃 dashboard 的 `baskets`。

### 5. 回测无交易成本 · 中
- **问题**：未计 HK 印花税(0.1%)+佣金+滑点，C/D 挂单战术优势被高估。
- **建议**：加成本参数。

## C. 工程鲁棒性（低优先）

### 6. FX 硬编码会过期
- `dashboard_config.json` 的 `fx_rates` 写死；汇率漂移直接错算跨币种 SOTP 目标（联想全靠 USD/HKD=7.82）。建议动态拉或加更新日期提醒。

### 7. `_annual_segments` 取 `hist[-1]` 不校验报告期/币种
- `watchlist_dashboard.py` ~L115。Longbridge 偶尔对 `af` 返回季度口径（已见 `report=qf`），静默用错期会算错。建议加断言/告警。

### 8. 性能
- 20 只串行 + 限流重试，跑几分钟。可并发，目前不痛。
