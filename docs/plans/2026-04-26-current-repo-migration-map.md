# 当前仓库迁移地图

## 目标

把现有仓库拆成三类：

- `保留`：直接迁入新架构
- `重写`：保留业务意图，但不保留现有实现
- `归档`：暂不进入新主流程

本文档只覆盖当前主仓，不包含 `external/TradingAgents-CN` 的内部代码细节。

## 1. 保留

这些内容与新工作台的第一版目标直接相关，建议继续保留并逐步迁入 `src/`：

### 数据与信号主链路

- `fetch_equities_fmp.py`
  - 当前最接近主数据管道。
  - 已覆盖行情拉取、特征加工、雷达输出、Supabase 上传。
  - 后续应拆分为 `ingestion`、`signal_engine`、`research_store` 三层。

- `equity_config.json`
  - 当前股票池定义可直接复用。
  - 后续可作为 watchlist 初始化来源。

- `radar_config.json`
  - 当前雷达参数可作为第一版触发规则原型。

- `supabase_history_fields.txt`
  - 字段整理有保留价值。
  - 后续应转为正式 schema 或模型定义。

### 说明文档

- `README.md`
  - 有业务背景价值，但后续需要按新架构重写。

- `docs/sql/2026-02-24-create-equity-opportunity-radar.sql`
  - 可作为信号结果表设计参考。

- `docs/stock_radar_runbook.md`
  - 可提炼运行与验证步骤。

### 测试思路

- `tests/test_radar_*.py`
  - 雷达规则的测试拆法有保留价值。

- `tests/test_load_history_fallback.py`
  - 体现了数据回退链路要被测试。

## 2. 重写

这些内容的业务意图保留，但现有实现不应直接作为新主路径。

### 展示层

- `streamlit_app.py`
  - 现有页面体量过大，承载了太多展示和部分业务判断。
  - 应重写为研究工作台 UI，只保留信息结构，不继承现有样式与页面组织。

### 推送层

- `scripts/send_radar_email.py`
  - 业务意图保留，但应等待新信号结构稳定后重写。

- `.github/workflows/opening-radar-email.yml`
  - 现阶段不应作为第一优先级。

### TradingAgents 适配层

- `scripts/ta_adapter_stub.py`
- `scripts/ta_adapter_llm.py`
- `ta_signals_demo.py`
  - 都属于后续可选增强。
  - 第一版不进入主流程。
  - 后续保留为“解释层 / 外部分析层”候选，而不是主信号引擎。

## 3. 归档

这些内容当前不服务第一版目标，建议进入 `legacy/` 或维持现状但不纳入新实现。

### 非第一版市场 / 实验路径

- `fetch_crypto_supports.py`
- `crypto_config.json`
- `backfill_crypto_history.py`

### 历史/实验脚本

- `backfill_equities.py`
- `backfill_equities_history.py`
- `equity_backfill_sample.py`
- `equity_history_builder.py`
- `openbb_three_months.py`
- `fetch_us_analyst_estimates.py`

这些脚本可能提供局部逻辑参考，但不应成为新主流程的依赖。

### 备份与输出

- `local_pipeline_backup/`
- `openbb_outputs/`
- `logs/`

这些目录目前主要是运行产物或历史试验，不应被新 UI 直接依赖。

### 外部仓库

- `external/TradingAgents-CN`
- `TradingAgents-CN/`

只保留为参考对象，不进入新项目运行主链路。

## 4. 建议的新主路径

第一版新代码统一进入：

- `src/research_workbench/ingestion`
- `src/research_workbench/research_store`
- `src/research_workbench/signal_engine`
- `src/research_workbench/validation`
- `src/research_workbench/ui`

旧脚本在新主路径稳定前继续存在，但不再新增复杂逻辑。

## 5. 立即执行建议

### 本周应做

1. 把 `fetch_equities_fmp.py` 里的配置、数据模型、Supabase 上传逻辑拆出来。
2. 定义统一的 watchlist / signal 数据模型。
3. 做出最小 research workbench 首页骨架。

### 暂时不要做

1. 深度接 `TradingAgents-CN`
2. 扩到港股/美股
3. 重做邮件推送
4. 引入复杂多 Agent 编排

