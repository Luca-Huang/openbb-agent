# 指标参考手册

> Research Workbench 信号引擎使用的所有指标、评分和触发逻辑的完整说明

---

## 一、技术指标

### MA20（20 日均线）

| 项目 | 说明 |
|---|---|
| 计算 | 最近 20 个交易日收盘价的简单算术平均 |
| 含义 | 短期趋势方向。价格站在 MA20 上方代表短期看多，跌破 MA20 代表短期走弱 |
| 引擎用途 | 用于计算移动止损（trailing stop）的参考位 |
| 代码位置 | `ingestion/providers.py → add_technical_indicators` |

### MA50（50 日均线）

| 项目 | 说明 |
|---|---|
| 计算 | 最近 50 个交易日收盘价的简单算术平均 |
| 含义 | 中期趋势方向。被广泛视为"趋势是否健康"的分水岭 |
| 引擎用途 | **pullback 触发的核心条件**——当价格距离 MA50 在 3% 以内时，判定为"回踩" |
| 阈值 | `ma50_distance_max = 0.03`（默认 3%） |
| 代码位置 | `signal_engine/radar.py → detect_trigger_type` |

### MA200（200 日均线）

| 项目 | 说明 |
|---|---|
| 计算 | 最近 200 个交易日收盘价的简单算术平均 |
| 含义 | 长期趋势方向。站在 MA200 上方一般视为长期看多 |
| 引擎用途 | 用于止损价的回退计算；在持仓建议中判断趋势结构是否完整 |
| 代码位置 | `signal_engine/radar.py → build_current_signals` |

---

### RSI14（14 日相对强弱指数）

| 项目 | 说明 |
|---|---|
| 计算 | `RSI = 100 - 100 / (1 + RS)`，其中 `RS = 14日平均涨幅 / 14日平均跌幅` |
| 范围 | 0 ~ 100 |
| 含义 | 衡量价格动量的超买/超卖状态 |
| 常用判断 | RSI < 30 → 超卖区（潜在反弹）；RSI > 70 → 超买区（潜在回调） |
| 引擎用途 | 在持仓卡片中展示，辅助判断短期动量。**当前不直接参与触发判断** |
| 代码位置 | `ingestion/providers.py → compute_rsi` |

---

### ATR14（14 日平均真实波幅）

| 项目 | 说明 |
|---|---|
| 计算 | `TR = max(H-L, |H-前收|, |L-前收|)`，ATR = 14 日 TR 的简单均值 |
| 含义 | 衡量价格波动幅度的绝对值，单位是价格 |
| 引擎用途 | 用于风险惩罚（ATR ≥ 12 扣 6 分，≥ 8 扣 3 分）；用于计算移动止损和止盈位 |
| 代码位置 | `signal_engine/radar.py → add_radar_features` |

---

### 支撑位（Primary / Secondary）

| 项目 | 说明 |
|---|---|
| Primary | 最近 20 个交易日的最低收盘价 |
| Secondary | Primary × 1.1（上方 10% 缓冲区） |
| 含义 | 近期价格底部参考。破 Primary 代表近期结构被打破 |
| 引擎用途 | pullback 触发要求价格 ≥ Primary（`require_support_above_primary`）；用于止损价和信号失效判断 |
| 代码位置 | `ingestion/providers.py → add_technical_indicators` |

---

### 成交量异动比（Volume Spike Ratio）

| 项目 | 说明 |
|---|---|
| 计算 | `当日成交量 / 20日均量` |
| 含义 | > 1.0 代表放量，< 1.0 代表缩量 |
| 引擎用途 | pullback 触发要求 ≥ 1.1；breakout 触发要求 ≥ 1.5；同时作为 volume_bonus 参与 opportunity_score 计算 |
| 代码位置 | `ingestion/providers.py → add_technical_indicators` |

---

### 20 日新高（high_20d）

| 项目 | 说明 |
|---|---|
| 计算 | 最近 20 个交易日最高价的最大值（shift 1，不含当日） |
| 含义 | 近期价格天花板。突破此位视为 breakout |
| 引擎用途 | **breakout 触发条件**——当收盘 > high_20d 且量比 ≥ 1.5 |
| 代码位置 | `signal_engine/radar.py → add_radar_features` |

---

### 60 日回撤（drawdown_60d）

| 项目 | 说明 |
|---|---|
| 计算 | `(60日滚动最高点 - 当前收盘) / 60日滚动最高点` |
| 范围 | 0 ~ 1（0 = 在高点，1 = 跌到 0） |
| 含义 | 衡量从近期高点的跌幅 |
| 引擎用途 | 当 drawdown > 22% 时开始扣减 opportunity_score（每超 1% 扣 1 分，上限 20 分） |
| 阈值 | `max_drawdown_60d_penalty_start = 0.22` |
| 代码位置 | `signal_engine/radar.py → compute_risk_penalty` |

---

### 收盘分位（close_percentile）

| 项目 | 说明 |
|---|---|
| 计算 | 当前收盘价在整段历史价格中的百分位排名 |
| 范围 | 0 ~ 1（0 = 历史最低，1 = 历史最高） |
| 含义 | 当前价格在历史中所处的位置 |
| 引擎用途 | 在 UI 和研究中展示，帮助判断当前是"高位"还是"低位" |

---

## 二、评分体系

### value_score（综合价值得分）

各子分之和，反映标的的综合投资吸引力。

| 子分 | 满分 | 评判依据 |
|---|---|---|
| `score_abs_valuation` | 20 | PE < 15 → +10；PE < 25 → +5；PS < 2 → +10；PS < 5 → +5；FCF Yield > 5% → +10 |
| `score_balance_sheet` | 10 | 现金 > 负债 → +10；负债 < 现金×2 → +5 |
| `score_shareholder_return` | 10 | 股息率 > 2% → +10 |
| `score_hist_valuation` | — | 历史估值分位（当前为 0，待实现） |
| `score_peer_valuation` | — | 可比公司估值（当前为 0，待实现） |
| `score_peg` | — | PEG 评分（当前为 0，待实现） |
| `score_growth_quality` | — | 增长质量（当前为 0，待实现） |
| `score_sentiment` | 5 | 默认中性分 |

**分级**：value_score > 30 → "合理区"；≤ 30 → "观望"

---

### opportunity_score（机会分）

```
opportunity_score = value_score + trigger_bonus + volume_bonus - risk_penalty
```

| 组成部分 | 计算 |
|---|---|
| trigger_bonus | breakout → +8；pullback → +6；无触发 → 0 |
| volume_bonus | `min(12, (volume_spike - 1) × 10)`，上限 12 分 |
| risk_penalty | drawdown_60d 超过 22% 的部分 × 100（上限 20）+ ATR 罚分（≥12 扣 6，≥8 扣 3） |

---

### conviction_score（综合信心分）

```
conviction_score = opportunity_score + valuation_score × 0.2 + quality_score × 0.2 - event_risk_score
```

| 组成部分 | 来源 |
|---|---|
| valuation_score | `score_hist_valuation + score_abs_valuation + score_peer_valuation` |
| quality_score | `score_peg + score_growth_quality + score_balance_sheet + score_shareholder_return` |
| event_risk_score | 来自手动事件表，负面高重要性 → +12，负面普通 → +6，中性 → +2，上限 25 |

**用途**：信号列表按 conviction_score 降序排列，分数越高 → 越值得优先复查。

---

## 三、触发类型

### pullback（回踩）

当价格回踩到 MA50 附近、量能温和放大、结构未破时触发。

**触发条件（全部满足）**：
1. 收盘价距 MA50 ≤ 3%（`ma50_distance_max = 0.03`）
2. 量比 ≥ 1.1（`volume_spike_min = 1.1`）
3. 收盘价 ≥ 20 日支撑位（`require_support_above_primary = True`）

**业务含义**：在上升趋势中，价格回到均线附近可能是加仓/建仓的机会。

---

### breakout（突破）

当价格突破近期高点、伴随放量时触发。

**触发条件（全部满足）**：
1. 收盘价 > 20 日最高价（`high_20d`）
2. 量比 ≥ 1.5（`volume_spike_min = 1.5`）

**业务含义**：放量新高代表市场对该价格区间的认可，可能是趋势延续的信号。

---

## 四、退出计划

### R 值

```
R = 入场价 - 止损价
```

R 是每股承担的风险，所有止盈位都以 R 的倍数来衡量。

### 止损价（stop_price）

优先使用以下位置中低于入场价的最高者：
1. 20 日支撑位（support_level_primary）
2. MA50
3. MA200

如果全部高于入场价，则回退到入场价 × 0.92（8% 止损）。

### 止盈 1（take_profit_1）

```
TP1 = 入场价 + 1R
```

到达后卖出一半仓位，锁定部分利润。

### 止盈 2（take_profit_2）

```
TP2 = 入场价 + 2R
```

到达后清仓剩余。

### 移动止损（trailing_stop）

```
trailing_stop = max(MA20, 20日最高收盘价 - 1.5 × ATR14)
```

随价格上涨动态上移，保护浮盈。

---

## 五、信号状态机

```
         ┌─────────────────────────────────┐
         │                                 ▼
      watch ──→ near_zone ──→ triggered ──→ invalidated
         ▲           │              │
         └───────────┴──────────────┘
                  (脱离区间)
```

| 状态 | 条件 |
|---|---|
| `watch` | 默认状态，持续关注 |
| `near_zone` | 价格距目标区间 ≤ 8%（zone_distance ≤ 0.08） |
| `triggered` | 满足 pullback 或 breakout 触发条件 |
| `invalidated` | 价格跌破止损价 |

---

## 六、持仓追踪指标

### 浮盈比例（floating_pnl_pct）

```
浮盈 = (当前市值 - 净投入) / 净投入
净投入 = 累计买入金额 - 累计卖出金额
```

使用摊薄成本（broker cost）计算，与券商显示的浮盈一致。

### 操作建议逻辑

| 条件 | 建议 |
|---|---|
| 空仓 + 触发 ACCEPT | 可考虑建仓 |
| 空仓 + 无触发 | 空仓观察 |
| 持仓 + 触发 ACCEPT | 可考虑加仓 |
| 持仓 + 价格 > MA50 且 > MA200 | 持有不动 |
| 持仓 + 价格 < MA50 且 < MA200 | 风险升高，考虑减仓 |
| 其他 | 观察 MA50 突破/破位 |
