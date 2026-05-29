# 多市场自选监控与信号引擎

面向个人日常使用的多市场（A 股 / 港股 / 美股）研究工作台，服务 10-50 只重点跟踪标的。

## 核心能力

1. **多市场买点检测**：A 股 / 港股 / 美股统一 watchlist，触发逻辑基于 20 日突破 + ATR 追高阈值 + 60 日回撤判断
2. **基本面过滤与估值体检**：PE 历史分位、机构目标价及评级分布、分部估值 (SOTP) 跨币种自动换算
3. **入场战术分析**：比对 4 种入场方式（收盘 / 次日开盘 / 限价回踩 / 不追高），量化执行成本
4. **历史验证**：每类信号都能查看过去的样本表现、胜率和最大回撤
5. **证据透明**：每个结论都能回溯到原始数据、触发条件和评分过程
6. **自动化监控**：日间 HTML 仪表板 + launchd 定时刷新 + 每日晨间自动打开

## 项目结构

```
src/research_workbench/
├── config.py                 # 统一配置入口
├── models.py                 # 核心数据模型
├── data_sources/             # Longbridge CLI 数据源与标准化
├── analysis/                 # summary / scoring 等纯分析逻辑
├── pipelines/                # 连接数据源、分析与输出的编排层
├── outputs/                  # 生成文件的读写边界
├── ingestion/                # watchlist 等用户输入加载
│   └── watchlist.py          # Watchlist 加载与过滤
├── research_store/           # schema 定义
│   └── schema.py             # 核心表 schema 定义
├── signal_engine/            # 信号生成层
│   ├── radar.py              # 触发检测 + 过滤 + 评分
│   └── changes.py            # 每日状态变化检测
├── validation/               # 历史验证层
│   └── replay.py             # 触发回放 + 验证统计
└── ui/                       # UI 边界（骨架级）
    └── README.md

app.py                                       # Streamlit 研究工作台主入口
scripts/
├── refresh.py                               # 统一信号与快照刷新入口
├── watchlist_dashboard.py                   # 多市场 HTML 仪表板（港股/A股/美股，纯渲染）
├── entry_timing_backtest.py                 # 入场战术对比回测
└── install_launchd.sh                       # macOS launchd 定时任务安装/卸载
research_inputs/
├── watchlist_cn.json                        # A 股 watchlist（Streamlit 工作台读）
└── dashboard_config.json                    # 仪表板配置：baskets / thesis / SOTP segments / FX
outputs/
├── research_data/                           # 输出数据（summary、history、signals）
├── watchlist_dashboard.html                 # 日间监控仪表板（自动生成，git ignored）
└── .launchd_logs/                           # 定时任务日志（git ignored）
tests/                                       # pytest 单元测试
~/Library/LaunchAgents/                      # 由 install_launchd.sh 写入
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 Supabase（可选）

```bash
export SUPABASE_URL="https://your-instance.supabase.co"
export SUPABASE_KEY="your-key"
```

> 不配置 Supabase 时，系统会自动回退到本地 CSV 文件。

### 3. 配置 Watchlist

仪表板的所有标的、基本面分析、SOTP 倍数都在 `research_inputs/dashboard_config.json`：

```json
{
  "baskets": {
    "HK": [["700.HK", "腾讯"], ...],
    "CN": [["002624.SZ", "完美世界"], ...],
    "US": [["MSFT.US", "微软"], ...]
  },
  "thesis": { "700.HK": { "业务": "...", "多头": "...", "空头": "...", "估值": "..." } },
  "segments": { "700.HK": { "segment_currency": "CNY", "segments": [...], "extra": [...] } },
  "fx_rates": { "CNY/HKD": 1.087 }
}
```

> Streamlit 工作台用的旧 `watchlist_cn.json` 仍单独维护，不与新仪表板共用。

### 4. 生成日间监控仪表板

```bash
python3 scripts/watchlist_dashboard.py
# 输出: outputs/watchlist_dashboard.html
```

### 5. 配置自动化刷新（macOS launchd）

```bash
# 一键安装：每日 17:00 刷新 + 10:00 自动打开。可重复运行（幂等）。
./scripts/install_launchd.sh

# 卸载
./scripts/install_launchd.sh uninstall

# 手动触发一次（不必等到 17:00）
launchctl kickstart -k gui/$(id -u)/com.user.watchlist-dashboard-update

# 查看注册状态 / 日志
launchctl list | grep watchlist-dashboard
tail -f outputs/.launchd_logs/dashboard_update.{out,err}.log
```

### 6. 刷新 Streamlit 工作台（可选）

```bash
python scripts/refresh.py              # Fetch data through Longbridge CLI and build signals
streamlit run app.py                   # Launch Streamlit research workbench
```

Streamlit 工作台包含四个页面：
- **首页**：今日重点变化、接近买点列表、基本面概览、公司事件、历史验证摘要
- **单票页**：价格证据、基本面证据、事件信息、历史同类信号、可展开解释层
- **验证页**：按触发类型的历史表现（胜率、平均收益、最大回撤）
- **设置页**：数据路径与 watchlist 预览

### 7. 入场战术分析（可选）

```bash
# 对比 4 种入场方式在历史数据上的成交价与前向收益
python3 scripts/entry_timing_backtest.py
# 输出: outputs/research_data/entry_timing_basket_trades.csv
```

## 信号结构

每个信号包含以下关键字段：

| 字段 | 说明 |
|------|------|
| `signal_state` | watch / near_zone / triggered / invalidated |
| `trigger_type` | pullback / breakout / none |
| `trigger_score` | 触发得分 |
| `valuation_score` | 估值得分 |
| `quality_score` | 增长质量得分 |
| `event_risk_score` | 近期事件风险分 |
| `conviction_score` | 综合信心分 |
| `reasons` | 信号原因（可回溯到具体数据） |
| `invalidation_conditions` | 失效条件 |

## 设计原则

- 先形成可用闭环，再追求完整架构
- 先服务少量标的深研究，再做全市场筛选
- 每个结论都必须能回溯到原始数据
- LLM 只作为可选解释层，不参与主判断
- UI 仅为可替换骨架，业务逻辑不耦合展示

## 数据采集

主数据管道通过 `scripts/refresh.py` 调用 Longbridge CLI，统一拉取 CN / HK / US 股票行情与估值字段。仪表板 (`watchlist_dashboard.py`) 直接调用 `LongbridgeCLIProvider` 的公开方法：`fetch_history` / `fetch_fundamentals` / `fetch_valuation` / `fetch_institution_rating` / `fetch_business_segments`。

## 测试

```bash
python3 -m pytest tests/                # 跑全部
python3 -m pytest tests/test_watchlist_dashboard.py -v  # 只跑仪表板单测
```
