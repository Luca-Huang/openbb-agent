# 多市场自选监控仪表板 · Runbook

## 概览

`scripts/watchlist_dashboard.py` 生成一个统一的 HTML 仪表板，覆盖港股 / A股 / 美股三市场的自选标的，支持 **tab 切换**、**实时数据**、**估值体检** 和 **分部估值拆解（SOTP）**。

**输出**: `outputs/watchlist_dashboard.html`  
**更新频率**: 每日 17:00（via launchd）、也可手动运行  
**数据源**: Longbridge CLI（price / PE分位 / 机构目标 / 分部收入 / 评级分布）

---

## 快速上手

### 1. 查看现有仪表板

```bash
open outputs/watchlist_dashboard.html
```

浏览器会显示：
- **顶部 tab 栏**：「港股（8）| A股（3）| 美股（6）」，括号内是各市场标的数
- **每张卡片**包含：
  - 股票名 + 代码 + 现价 + 1日涨跌 + **结论徽章**（建仓/等回踩/观察/不碰）
  - 基本面网格：PE / PB / EPS(TTM) / 股息率 / 市值 / 换手率
  - **估值体检条**：PE便宜分位 + 机构目标价 + 评级分布（强买/买/持/卖）
  - **分部估值表**（仅限多业务公司）：各段营收 × PS = 估值，加总隐含空间
  - 技术信号网格：触发 / 阶段 / RSI / 量比 / 距高 / 等
  - 定性分析：业务 / 多头 / 空头 / 估值

### 2. 手动刷新仪表板

```bash
python3 scripts/watchlist_dashboard.py
```

输出示例：
```
== 港股 (HK) ==
  ✓ 腾讯 700.HK: 不碰
  ✓ 泡泡玛特 9992.HK: 建仓
  ...
== A股 (CN) ==
  ✓ 完美世界 002624.SZ: 不碰
  ...
== 美股 (US) ==
  ✓ 苹果 AAPL.US: 观察
  ...
已生成: /Users/huangyuxiang/openbb-agent/outputs/watchlist_dashboard.html
```

---

## 配置与扩展

### 添加 / 修改标的

编辑 `scripts/watchlist_dashboard.py` 的 `BASKETS` 字典：

```python
BASKETS: dict[str, list[tuple[str, str]]] = {
    "HK": [
        ("700.HK", "腾讯"), 
        ("9988.HK", "阿里巴巴"),
        # 添加新港股
        ("1030.HK", "高鹰国际"),
    ],
    "CN": [
        ("002624.SZ", "完美世界"),
        # 添加新 A股
        ("000858.SZ", "五粮液"),
    ],
    "US": [
        ("MSFT.US", "微软"),
        # 添加新美股
        ("NVDA.US", "英伟达"),
    ],
}
```

保存后下次运行 `watchlist_dashboard.py` 会自动拉取新标的数据。

### 配置分部估值 (SOTP)

仅限于 **多业务公司**（如腾讯、阿里、小米、美团）。编辑 `SEGMENTS` 字典：

```python
SEGMENTS: dict[str, dict] = {
    "1030.HK": {  # 新公司的符号
        "segments": [
            {"match": ["核心业务"], "label": "XXX", "ps": 3.5, 
             "reason": "净利率~XX%×PE~YY → PS~3.5"},
            {"match": ["新业务"], "label": "YYY", "ps": 2.0,
             "reason": "亏损期权估值"},
        ],
        "extra": [  # 非营收价值（投资组合/净现金）
            {"label": "投资组合", "value_cny_yi": 500, "reason": "..."}
        ],
    },
}
```

**PS 倍数推导方法**：
```
PS = 该段净利率 × 该业务合理PE

示例：
- 核心本地商业：经营利润率19% × PE10-11 = PS 2.0
- 互联网服务：净利率50% × PE18 = PS 5.5
```

---

## 自动化刷新

### 配置 launchd（macOS）

两个定时任务已预配置在 `~/Library/LaunchAgents/`：

```
com.user.watchlist-dashboard-update.plist   # 每日 17:00 刷新
com.user.watchlist-dashboard-open.plist     # 每日 10:00 打开
```

### 查看状态 / 手动触发

```bash
# 查看注册状态
launchctl list | grep watchlist-dashboard

# 手动触发一次更新（不必等到 17:00）
launchctl kickstart -k gui/$(id -u)/com.user.watchlist-dashboard-update

# 查看日志
tail -f ~/openbb-agent/outputs/.launchd_logs/dashboard_update.out.log
tail -f ~/openbb-agent/outputs/.launchd_logs/dashboard_update.err.log
```

### 卸载定时任务

```bash
UID=$(id -u)
launchctl bootout gui/$UID ~/Library/LaunchAgents/com.user.watchlist-dashboard-update.plist
launchctl bootout gui/$UID ~/Library/LaunchAgents/com.user.watchlist-dashboard-open.plist
```

---

## 理解卡片内容

### 结论徽章（左侧色条）

| 徽章 | 含义 | 色条 |
|---|---|---|
| **建仓** | 突破达标且未越追高线，右侧确认机会 | 绿色 |
| **等回踩** | 已突破但冲高过头，挂回踩单别追 | 橙色 |
| **观察** | 接近突破、结构健康，等放量触发 | 蓝色 |
| **不碰** | 下降趋势无信号，接飞刀区 | 灰色 |

### 估值体检条

- **左侧进度条**：PE 5年便宜分位（0% = 最贵，100% = 最便宜）
- **右侧数字**：机构目标价 vs 现价的上行空间
- **评级分布**：强买/买/持/卖 的机构数量

### 分部估值表（SOTP）

示例：腾讯

| 业务段 | 营收 | PS | 估值 | 占比 |
|---|---|---|---|---|
| 增值服务 | 3693亿 | ×7.0 | 25850亿 | 52% |
| 金融科技 | 2294亿 | ×3.5 | 8030亿 | 16% |
| 营销服务 | 1450亿 | ×7.0 | 10148亿 | 20% |
| 投资组合 | — | — | 5800亿 | 12% |
| **加总** | | | **49828亿** | **100%** |

加总换汇后与现价市值比较，隐含上行/下行空间。

---

## 故障排查

### 问题：某个标的拉不到数据

```
✗ 某公司 XXXX.US: LongbridgeCLIError: ...
```

**原因通常**：
1. 符号拼写错误（应为 `STOCK.US` 或 `STOCK.HK` 格式）
2. Longbridge 无该证券（罕见，多为已退市或数据源未覆盖）
3. 网络超时或 Longbridge 服务临时不可用

**解决**：
- 验证符号，可在命令行试：`longbridge quote XXXX.US --format json`
- 检查网络和 `~/.longbridge/` 认证令牌
- 稍后重试

### 问题：SOTP 估值偏离市场很大

**通常原因**：PS 倍数设置不合理

**调试步骤**：
1. 验证 PS 推导：`段净利率 × 合理PE = PS`
2. 和机构目标价比较——目标价隐含的 PE 是多少？
3. 参考同行可比公司的 PE 或 PS

---

## 数据刷新频率与覆盖时间

| 数据 | 更新频率 | 延迟 |
|---|---|---|
| 现价 + 1日涨跌 | 每日 17:00 | <5分钟 |
| PE / PB / EPS | 每日 17:00 | 延迟 1-2 天 |
| 机构目标价 / 评级 | 每日 17:00 | 延迟数天 |
| 分部营收 | 每日 17:00 | 财报发布后 1-2 周 |

**最新数据**：点击 `<p class="sub">` 看生成时间。

---

## 常见用法

### 用法 1：早上 10 点查看前一晚的变化

```bash
# 前一晚 17:00 自动刷新了数据，早上 10:00 自动打开浏览器
# 你只需查看 HTML，看看哪些标的有新的「建仓」或「等回踩」
```

### 用法 2：手动检查某个标的的详细信息

```bash
# 若想立即看最新数据（不等 17:00），手动运行：
python3 scripts/watchlist_dashboard.py

# 然后在浏览器刷新 watchlist_dashboard.html
```

### 用法 3：扩充观察列表

```bash
# 编辑 BASKETS 字典，添加新标的
# 下次运行时会自动拉该标的所有数据
# PS：若要SOTP，需同时编辑 SEGMENTS 字典
```

### 用法 4：对比多市场行情

```bash
# 通过 tab 切换，一眼看到港股 / A股 / 美股的相对强弱
# 比如某天港股有 3 个「建仓」，美股 0 个，A股 1 个
# → 可能反映宏观风格或市场情绪的变化
```

---

## 代码主要逻辑

### 1. `analyze(symbol, market)` 

拉取单个标的的数据（价、PE、基本面）并计算技术信号及结论。

### 2. `build_sotp(symbol, mktcap, close)`

若标的在 `SEGMENTS` 中有定义，计算分部估值加总及隐含上行空间。

### 3. `render(by_market: dict)`

根据市场对卡片分组，生成 tab 栏 + 三个 panel，输出 HTML。

### 4. `showTab()` (JavaScript)

客户端 JS 处理 tab 切换，不涉及服务器。

---

## FAQ

**Q: 为什么有的公司没有 SOTP 表？**  
A: 仅多业务公司（多个独立业务线）显示 SOTP。单一主业公司（如泡泡玛特）不拆分。

**Q: PS 倍数怎么确定？**  
A: `PS ≈ 段净利率 × 该业务的合理PE`。需对标同行可比公司，并考虑成长性 / 确定性微调。

**Q: 为什么机构目标价和 SOTP 隐含价格不一致？**  
A: 正常。机构目标往往基于历史 PE / 行业平均，而 SOTP 是自下而上的分部拆分。两个都有参考价值。

**Q: 我能改 CSS / HTML 吗？**  
A: 可以直接编辑生成的 HTML，但每次运行 `watchlist_dashboard.py` 会覆盖。建议在脚本里改样式，而不是在 HTML 里改。

---

## 相关文档

- [stock_radar_runbook.md](./stock_radar_runbook.md) — 触发信号逻辑（何时「建仓」「等回踩」等）
- [investment_framework.md](./investment_framework.md) — 投资原则与决策框架
- [source_matrix.md](./source_matrix.md) — 数据源与字段映射

