# 价值监控与评分系统

该项目包含两部分：

1. **数据采集脚本** (`fetch_equities_fmp.py`)：
   - 美股优先通过 FMP `stable/historical-price-eod/light` 拉取行情，若接口受限则自动 fallback 到 yfinance。
   - 港股 / A 股 默认使用 yfinance 数据。
   - 计算 PE/P.S 分位、PEG、自由现金流收益率、滚动支撑位、远期 PE 等指标，并生成刷新频率、下一次刷新日期。
   - 内置“价值锚点评分体系”，从估值、成长、财务、技术/情绪四个维度打分，输出 `value_score` 与等级（黄金坑 / 白银坑 / 合理区 / 观望）。
   - 输出文件位于 `openbb_outputs/`：
     - `three_month_close_history.csv`
     - `three_month_summary.csv`
     - `us_analyst_estimates.csv`

2. **可视化面板** (`streamlit_app.py`)：
   - 通过 `streamlit run streamlit_app.py` 启动，支持按市场筛选（美股、港股、A 股）；
   - 展示归一化走势、PE/历史分位、支撑位、分析师预测、入场信号评分等；
   - “指标说明” 中可查看各指标及评分逻辑。

## 价值锚点评分体系（100分）

1. **估值吸引力 (30)**
   - 历史估值分位：取 PE/PB/P.S（当前实现 PE、P.S）中最低分位，按 10/7/4/1/0 分。
   - 绝对估值：按自由现金流收益率（≥6% 记 10 分）。
   - 同业比较：将当前 PE/P.S 与同市场中位数比较，低 20% 记 10 分，近似 5 分，高 0 分。

2. **成长性与性价比 (30)**
   - PEG：≤0.8 记 15 分，逐档递减；当前因 EPS 历史不足大多为 0。
   - 增长质量：基于 TTM 营收、净利增速，利润增速 > 营收增速记满分。

3. **财务健康与股东回报 (20)**
   - 资产负债表：净现金企业记 10 分，债务升高分值下降。
   - 股东回报：最近四季回购 + 分红记 10 分，仅一项记 7 分，否则为 0。

4. **技术面与市场情绪 (20)**
   - 关键支撑：现价相对 20 日滚动支撑位的偏离程度。
   - 情绪：依据 yfinance 的 `recommendationKey/Mean`（评级越悲观得分越高）。

- **总分解释**：
  - 80-100：黄金坑（积极建仓）；
  - 60-79：白银坑（分批建仓）；
  - 40-59：合理区（观望/持有）；
  - <40：观望或高估。

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 Supabase

脚本与 Streamlit 默认使用以下环境变量（可在运行脚本的终端或系统环境中设置）：

```bash
export SUPABASE_URL="https://wpyrevceqirzpwcpulqz.supabase.co"
export SUPABASE_KEY="YOUR_SUPABASE_SERVICE_ROLE_OR_ANON_KEY"
# 如有自定义表名，可覆盖默认值
export SUPABASE_SUMMARY_TABLE="equity_metrics"
export SUPABASE_HISTORY_TABLE="equity_metrics_history"
export SUPABASE_ANALYST_TABLE="us_analyst_estimates"
export SUPABASE_CRYPTO_TABLE="crypto_supports"
```

> 默认值已写入脚本，若直接复用仓库提供的 Supabase 实例，可暂不设置。若部署到自己的 Supabase，请务必覆盖 URL/KEY。

### 3. 运行采集脚本（写入 Supabase + CSV）

```bash
source .venv311/bin/activate  # 如果使用本地虚拟环境
python fetch_equities_fmp.py
python fetch_crypto_supports.py
python fetch_us_analyst_estimates.py
```

三份脚本会：
- 拉取/计算指标并生成 `openbb_outputs/*.csv`；
- 将数据 upsert 到 Supabase（包含失败重试、冲突兜底逻辑）。

> 建议在 Windows 任务计划程序、Linux `cron` 或 GitHub Actions 中定时运行，实现每日自动更新。

### 4. 启动可视化（本地 / Streamlit Cloud）

```bash
streamlit run streamlit_app.py
```

Streamlit 应用会优先从 Supabase REST API 读取 `equity_metrics` / `equity_metrics_history` / `crypto_supports` / `us_analyst_estimates`，仅当云端无数据时才回退到本地 CSV。

部署到 Streamlit Cloud 的步骤：
1. Fork 本仓库；
2. 在 App 的 Secrets 设置 `SUPABASE_URL` 与 `SUPABASE_KEY`；
3. Cloud 会自动安装 `requirements.txt` 并运行 `streamlit_app.py`，即得公开访问链接。

### 5. 自动更新建议

- **Windows**：任务计划程序 → 创建任务 → 触发器选“每日”，操作填  
  `cmd /c "cd C:\path\openbb-agent && C:\Python311\python fetch_equities_fmp.py"`.
- **Linux/NAS**：`crontab -e` 添加  
  `0 7 * * * cd /path/openbb-agent && /usr/bin/python3 fetch_equities_fmp.py >> /path/log/equity.log 2>&1`.
- **GitHub Actions**：建立 schedule workflow，runner 上执行三份脚本，适合无服务器成本的场景。

> 注意：PEG、PB 等部分指标受限于免费数据源，若上市未满 5 年或数据缺失，对应分位会显示 `N/A`，`pe_coverage_years` / `ps_coverage_years` 会显示实际覆盖年限。

## 加密货币“三维度确认法”数据采集

面对震荡下行行情，仅依赖单一技术位容易失效。`fetch_crypto_supports.py` + `crypto_config.json` 新增如下能力：

1. **技术面（CoinGecko / Binance / yfinance fallback）**  
   - 默认跟踪 BTC、ETH、SOL、DOGE（可在 `crypto_config.json` 扩展），拉取 420 天价格/成交量并生成 50/200 日均线、最近 3 个摆动低点、Volume Profile (HVN) 最高成交区、14 日 RSI、斐波那契 38.2%/50%/61.8% 关键位、7/30 日涨跌幅、距离均线的偏离百分比、摆动区间（Swing Range）上下沿、支撑测试次数、30 日均量等。
2. **链上数据（Glassnode，可选）**  
   - 支持 MVRV、长期持有者成本基础、交易所净流量、aSOPR、UTXO 成本分布等指标；在 `crypto_config.json` 中写入 Glassnode API Key 即可生效。
3. **市场情绪与宏观变量**  
   - 合并恐惧与贪婪指数、Binance 永续资金费率、交易所稳定币库存变化，帮助判断“弹药”是否充足。

运行方式：
```bash
python fetch_crypto_supports.py
```
输出位于 `openbb_outputs/crypto/`：
- `BTC_support_map.json` 等：汇总当前价格、支撑/阻力、链上与情绪信号，可直接投喂到 Agent/知识库。
- `crypto_support_dashboard.csv`：适合 Excel/Streamlit 统一查看，便于按“技术 + 链上 + 情绪”三重共振制定分批建仓 / 止损 / 减仓策略。

> 若暂未配置 Glassnode API Key，脚本会跳过链上指标，其余技术与情绪数据仍可正常使用。

### 名词解释（加密货币部分常用字段）
- **recent_supports / 支撑次数**：最近三次摆动低点及对应价格 / 被成功测试的次数，次数越高说明该区间成交确认度越高。
- **fib_38_2 / 50 / 61_8**：斐波那契关键回撤位，截取近 120 天摆动区间计算的 38.2%、50%、61.8% 目标价。
- **swing_low / swing_high / swing_range**：最近摆动区间的最低/最高价及跨度，有助于感知当前波动幅度。
- **pct_change_7d / pct_change_30d**：过去 7 / 30 日的百分比涨跌幅。
- **distance_ma50_pct / distance_ma200_pct**：现价相对于 50 / 200 日均线的偏离百分比，为负表示位于均线下方。
- **volume_avg_30d**：过去 30 日的平均成交量，用于观察筹码活跃度变化。
- **hvn_zones**：Volume Profile 的高成交量节点，每个节点包含价格区间上下沿及成交量。
- **fear_greed / funding_rate**：来自恐惧与贪婪指数 / Binance 永续合约资金费率的情绪信号。

## 后续迭代计划

目前系统已经具备「多市场数据入库 + Supabase 统一供给 + Streamlit 展示」，但对投资决策的辅助仍有较大提升空间。接下来计划：

1. **盘后日报与入场建议**  
   - 每天收盘后自动扫描 Supabase，将 `entry_recommendation=建议入场` 的标的整理成日报，输出建议入场价、原因、触发条件。  
   - 形式：脚本生成 Markdown/HTML，推送到 Telegram / 邮件 / n8n。

2. **持仓与风控模块**  
   - Streamlit 增设“持仓面板”，允许录入/导入持仓成本、仓位、目标仓。  
   - 每日盘后基于 200 日均线、滚动支撑位自动计算挂单区间、止盈/止损价格。

3. **AI 总结体系**  
   - 利用 GPT/Gemini 会员，通过 n8n 工作流拉取指数、新闻、宏观数据，得到当日大盘点评、风险提示、事件驱动结论。  
   - 参考 TauricResearch 达成“文字日报 + 结构化标签”的效果。

4. **“量化执行投资系统 V6.1” 对齐**  
   - **第一阶段 - 好公司过滤**：补齐经营现金流 vs 净利润、资产负债率、回购/管理层增持等硬性条款，不满足者永久剔除。  
   - **第二阶段 - 好价格信号**：将 PE 分位、PEG、FCF Yield、200 日均线偏离、恐惧贪婪指数量化为总分 11 分，≥7 分触发买入。  
   - **第三阶段 - 执行手册**：预计算试探仓/主力仓价格、初始止损、分批止盈、移动止盈、最大亏损金额等字段并写入 Supabase，供前端直接展示。  
   - 形成“资格过滤 → 信号 → 风控”闭环，便于自动化决策与复盘。

> 欢迎在 Issues 中提出其他需求或提交 PR，一起把系统升级为真正的「投前分析 + 交易信号 + 风控执行」一体化工具。
