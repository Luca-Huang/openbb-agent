# A 股个人研究工作台

面向个人日常使用的 A 股研究工作台，服务 10-30 只重点跟踪标的。

## 核心能力

1. **买点检测**：自动检测 watchlist 中的股票是否正在接近值得考虑的买点（回踩 MA50、放量突破等）
2. **基本面过滤**：估值、增长质量、财务健康、公司事件综合评分
3. **历史验证**：每类信号都能查看过去的样本表现、胜率和最大回撤
4. **证据透明**：每个结论都能回溯到原始数据、触发条件和评分过程
5. **每日变化**：首页自动高亮状态变化（新触发、进入/脱离观察区间、conviction 变化）

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

app.py                        # Streamlit 研究工作台主入口
scripts/refresh.py            # 统一信号与快照刷新入口
research_inputs/              # 输入数据（watchlist、手动事件）
outputs/research_data/        # 输出数据（summary、history、signals）
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

编辑 `research_inputs/watchlist_cn.json`，添加你关注的 A 股标的：

```json
[
  {
    "symbol": "002624.SZ",
    "name": "完美世界",
    "market": "CN",
    "sector": "传媒游戏",
    "status": "watch",
    "target_zone_low": 9.8,
    "target_zone_high": 10.5,
    "notes": "关注回踩 MA50 机会"
  }
]
```

### 4. 刷新数据

```bash
python scripts/refresh.py              # Fetch data through Longbridge CLI and build signals
```

### 5. 启动工作台

```bash
streamlit run app.py
```

工作台包含四个页面：
- **首页**：今日重点变化、接近买点列表、基本面概览、公司事件、历史验证摘要
- **单票页**：价格证据、基本面证据、事件信息、历史同类信号、可展开解释层
- **验证页**：按触发类型的历史表现（胜率、平均收益、最大回撤）
- **设置页**：数据路径与 watchlist 预览

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

## 旧代码

以下内容已移入 `archive/`，不再是主流程依赖：
- 加密货币相关脚本
- 美股分析师预测脚本
- 历史回填实验脚本
- TradingAgents 适配器

旧的全市场 Streamlit 面板 (`streamlit_app.py`) 保留但不再是主入口。

## 数据采集

主数据管道通过 `scripts/refresh.py` 调用 Longbridge CLI，统一拉取 CN / HK / US 股票行情与估值字段。
