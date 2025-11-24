# 分析原理概要

面向当前产品（股票 + 加密资产）的“数据 → 特征 → LLM → 结构化输出”流程说明，以及与 TradingAgents-CN 的结合思路。

## 数据与特征
- 行情数据：股票（yfinance/FMP）与加密货币（Coingecko/DefiLlama），提取收盘价、成交量、MA/RSI/MACD 等技术指标，支撑/阻力（近低点、斐波/成交量节点）。
- 基本面/链上：TVL、DEX 量、费用、稳定币供应等（DefiLlama），可扩展财报、新闻摘要等文本特征。
- 统一入库：写入 Supabase 历史表，为前端展示和后续策略/LLM 提供输入。

## LLM 驱动的分析/报告
- 上下文构造：将结构化特征（价格、指标、链上数据等）与文本描述（新闻/财报摘要）打包进 Prompt，附带任务指令（如“给出入场/止盈/风控建议”）。
- 生成：LLM 产出完整的信号/报告文本（偏自由度），包含方向、理由、风险提示等。
- 解析：使用二次 Prompt（`signal_processing` 模块思路）提取结构化槽位，如方向、仓位、理由、风险、时间窗口，保证前端和自动化可用。
- 校验/提示：输出时附带风险提示、数据缺口提示，避免“无数据时硬给结论”。

## 规则/因子策略（可选增强）
- 现成仓库未提供硬编码阈值策略（例如“价格>MA50 且 RSI<30 买入”），只提供指标计算函数。
- 如需确定性策略，可在现有特征上自定义规则/回测（动量 + 均线 + 支撑位 + 成交量），生成信号后再交给 LLM 做解释或直接前端展示。

## TradingAgents-CN 的作用
- 充当“策略/信号工作流”与 LLM 调度层：在 `trading_graph` 中组织任务、调用 LLM，并用 `signal_processing` 解析输出。
- 提供指标工具集（`tools/analysis/indicators.py`）和 Docker/compose 部署模版，可复用其后端/前端壳，将本项目的数据源作为特征输入。
- 组合方式：用 Supabase 数据作为特征源 → 在 TradingAgents-CN 后端构造 Prompt → LLM 生成 + 解析 → 返回结构化信号给前端。

## 部署/运行要点
- 数据层：我们的 backfill/增量脚本写入 Supabase（支持环境变量注入 Supabase/Coingecko/DefiLlama 参数）。
- 服务层：可通过新增的 `docker-compose.data.yml` 把数据抓取/写库打包成独立容器，配合 TradingAgents-CN 主 compose 统一拉起。
- 前端：Streamlit 现有界面或 TradingAgents-CN 的 Vue 前端均可作为展示入口；信号/报告以结构化 JSON + 文本返回，方便渲染与搜索。
