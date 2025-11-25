# TradingAgents-CN 模块集成方案

此模块以子模块形式放在 `external/TradingAgents-CN`，与主应用解耦，便于独立升级/替换。

## 结构
- 子模块：`external/TradingAgents-CN`（固定版本，git submodule）
- 数据接口：Supabase。输入=行情/指标/链上特征；输出=信号/报告（建议写回 Supabase 的信号表，如 `ta_signals`）。
- 适配层（建议）：在主仓写一个小脚本/服务，从 Supabase 拉特征 → 调用 TradingAgents-CN 的 LLM 工作流 → 将结构化信号写回 Supabase。适配层与主 UI/TradingAgents-CN 通过 HTTP 或队列交互。

## 推荐使用方式
1) 拉取子模块：`git submodule update --init --recursive`
2) 在 `external/TradingAgents-CN` 内按其 README 安装依赖，单独的虚拟环境或容器，避免污染主项目。
3) 新建信号表（示例）：
   ```sql
   create table if not exists ta_signals (
     symbol text not null,
     as_of_date date not null,
     signal text,
     confidence numeric,
     rationale text,
     created_at timestamptz default now(),
     primary key (symbol, as_of_date)
   );
   ```
4) 适配层：
   - 从 Supabase 拉取最新 `equity_metrics` / `equity_metrics_history` / 链上数据。
   - 组装为 TradingAgents-CN 需要的 prompt/context，调用其 `trading_graph`/`signal_processing` 工作流。
   - 将返回的结构化信号写入 `ta_signals`。
5) 前端/Streamlit 只读取 `ta_signals`（及原数据表），不直接依赖 TradingAgents-CN 代码，保持松耦合。

## 部署建议
- 单独的 compose：创建 `docker-compose.tradingagents.yml`，运行 TradingAgents-CN 服务（LLM 调度、worker）。
- 配置隔离：为 TradingAgents-CN 使用独立 `.env.tradingagents`，存放 LLM key、Supabase key 等。
- 版本锁定：子模块固定到某个 commit，需要升级时再手动 bump，避免上游变更破坏兼容。

## 后续工作（可选）
- 增加一个简单的适配脚本模板，定时从 Supabase 拉特征、调用 TradingAgents-CN，并写回 `ta_signals`。
- 在前端增加 `ta_signals` 展示/筛选的 Tab。
- 如需在线推理，考虑为 TradingAgents-CN 增加轻量 API 层，供适配脚本调用。
