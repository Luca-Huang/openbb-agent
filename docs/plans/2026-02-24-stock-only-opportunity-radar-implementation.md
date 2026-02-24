# 股票单一化机会雷达实施计划（Stock-Only Opportunity Radar）

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标（Goal）:** 下线加密货币相关 UI 与采集路径，交付一个面向 US/HK/CN 三市场的“每日股票机会雷达”，支持 2-8 周波段决策并给出可执行候选清单，同时实现开盘时段邮件自动推送。

**架构（Architecture）:** 以 `/Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py` 作为唯一数据主链路，统一产出 summary/history/radar。新增雷达评分层（流动性门槛 + 回踩/突破混合触发 + 中等风控扣分），并同时落地到 CSV 与 Supabase。Streamlit 改为股票单入口，首页优先展示“今日机会雷达”，再展示深度图表；通过 GitHub Actions + SMTP 在各市场开盘窗口自动发送邮件摘要。

**技术栈（Tech Stack）:** Python 3、pandas、numpy、requests、streamlit、Supabase REST/SDK、unittest。

## 前置约束

- 实施前使用 `@superpowers:using-git-worktrees` 创建隔离工作区。
- 每个任务按 `@superpowers:test-driven-development` 执行（先写失败测试，再实现）。
- 收尾前按 `@superpowers:verification-before-completion` 给出可验证证据。

### 任务 1：移除加密表层（UI + 采集入口）

**文件：**
- 修改：`/Users/huangyuxiang/openbb-agent/streamlit_app.py`
- 修改：`/Users/huangyuxiang/openbb-agent/README.md`
- 新建：`/Users/huangyuxiang/openbb-agent/tests/test_stock_only_ui.py`

**步骤 1：先写失败测试**

```python
# /Users/huangyuxiang/openbb-agent/tests/test_stock_only_ui.py
import unittest
from pathlib import Path

class TestStockOnlyUI(unittest.TestCase):
    def test_no_crypto_dashboard_symbols(self):
        src = Path("/Users/huangyuxiang/openbb-agent/streamlit_app.py").read_text(encoding="utf-8")
        self.assertNotIn("render_crypto_dashboard", src)
        self.assertNotIn("load_crypto_supports", src)
        self.assertNotIn("加密面板", src)

if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行并确认失败**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_stock_only_ui.py -v`  
预期：FAIL（当前仍有 crypto 相关代码）

**步骤 3：最小实现**

- `render_equity_dashboard()` 改为仅渲染股票内容。
- 删除 `CRYPTO_CSV_PATH`、`load_crypto_supports()`、`render_crypto_dashboard()`。
- README 去掉 crypto 环节与命令。

**步骤 4：复跑测试确认通过**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_stock_only_ui.py -v`  
预期：PASS

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/streamlit_app.py /Users/huangyuxiang/openbb-agent/README.md /Users/huangyuxiang/openbb-agent/tests/test_stock_only_ui.py
git commit -m "refactor: remove crypto ui paths and docs"
```

### 任务 2：新增雷达配置（跨市场规则、阈值、配额）

**文件：**
- 新建：`/Users/huangyuxiang/openbb-agent/radar_config.json`
- 修改：`/Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py`
- 新建：`/Users/huangyuxiang/openbb-agent/tests/test_radar_config.py`

**步骤 1：先写失败测试**

```python
# /Users/huangyuxiang/openbb-agent/tests/test_radar_config.py
import json
import unittest
from pathlib import Path

class TestRadarConfig(unittest.TestCase):
    def test_required_keys_exist(self):
        cfg = json.loads(Path("/Users/huangyuxiang/openbb-agent/radar_config.json").read_text(encoding="utf-8"))
        for key in ["markets", "liquidity", "triggers", "risk", "output"]:
            self.assertIn(key, cfg)

if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行并确认失败**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_config.py -v`  
预期：FAIL（配置文件未创建）

**步骤 3：最小实现**

创建 `radar_config.json`（首版默认值）：

```json
{
  "markets": ["US", "HK", "CN"],
  "liquidity": { "min_price_usd": 2.0, "min_dollar_volume20": 5000000 },
  "triggers": {
    "pullback": { "ma50_distance_max": 0.03, "volume_spike_min": 1.1, "require_support_above_primary": true },
    "breakout": { "volume_spike_min": 1.5, "lookback_high_days": 20 }
  },
  "risk": { "mode": "medium_penalty", "max_drawdown_60d_penalty_start": 0.22 },
  "output": { "min_candidates": 8, "max_candidates": 15 }
}
```

在 `fetch_equities_fmp.py` 增加配置加载器，并提供缺省兜底。

**步骤 4：复跑测试确认通过**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_config.py -v`  
预期：PASS

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/radar_config.json /Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py /Users/huangyuxiang/openbb-agent/tests/test_radar_config.py
git commit -m "feat: add stock opportunity radar config"
```

### 任务 3：补齐雷达特征字段

**文件：**
- 修改：`/Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py`
- 新建：`/Users/huangyuxiang/openbb-agent/tests/test_radar_features.py`

**步骤 1：先写失败测试**

```python
# /Users/huangyuxiang/openbb-agent/tests/test_radar_features.py
import unittest
import pandas as pd
import numpy as np
import fetch_equities_fmp as m

class TestRadarFeatures(unittest.TestCase):
    def test_feature_columns_exist(self):
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            "close": np.linspace(10, 20, 80),
            "high": np.linspace(10.5, 20.5, 80),
            "low": np.linspace(9.5, 19.5, 80),
            "volume": np.full(80, 1000000.0)
        })
        out = m.add_radar_features(df)
        for col in ["high_20d", "atr14", "drawdown_60d", "dollar_volume20"]:
            self.assertIn(col, out.columns)

if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行并确认失败**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_features.py -v`  
预期：FAIL（函数不存在）

**步骤 3：最小实现**

在 `fetch_equities_fmp.py` 添加：

```python
def add_radar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    out["high_20d"] = out["high"].rolling(20, min_periods=1).max().shift(1)
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        (out["high"] - out["low"]).abs(),
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=1).mean()
    rolling_peak = out["close"].rolling(60, min_periods=1).max()
    out["drawdown_60d"] = (rolling_peak - out["close"]) / rolling_peak.replace(0, np.nan)
    out["dollar_volume20"] = (out["close"] * out["volume"]).rolling(20, min_periods=1).mean()
    return out
```

并在主流程调用该函数。

**步骤 4：复跑测试确认通过**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_features.py -v`  
预期：PASS

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py /Users/huangyuxiang/openbb-agent/tests/test_radar_features.py
git commit -m "feat: add radar feature engineering fields"
```

### 任务 4：实现混合触发引擎（回踩 + 突破）

**文件：**
- 修改：`/Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py`
- 新建：`/Users/huangyuxiang/openbb-agent/tests/test_radar_triggers.py`

**步骤 1：先写失败测试**

```python
# /Users/huangyuxiang/openbb-agent/tests/test_radar_triggers.py
import unittest
import pandas as pd
import fetch_equities_fmp as m

class TestRadarTriggers(unittest.TestCase):
    def test_trigger_type_is_pullback_or_breakout_or_none(self):
        row = pd.Series({
            "close": 100, "ma50": 98, "support_level": 96, "volume_spike_ratio": 1.2,
            "high_20d": 102
        })
        t = m.detect_trigger_type(row, {
            "pullback": {"ma50_distance_max": 0.03, "volume_spike_min": 1.1, "require_support_above_primary": True},
            "breakout": {"volume_spike_min": 1.5}
        })
        self.assertIn(t, {"pullback", "breakout", None})

if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行并确认失败**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_triggers.py -v`  
预期：FAIL（函数不存在）

**步骤 3：最小实现**

新增：
- `detect_trigger_type(row, trigger_cfg)`
- `build_radar_record(summary_row, latest_hist_row, cfg)`

输出字段：
- `symbol, market, trigger_type, trigger_price, stop_price, opportunity_score, risk_flags, reason_1line`

触发逻辑：
- 回踩：靠近 MA50 + 价格在主支撑上方 + 轻量放量确认
- 突破：突破 `high_20d` + 放量显著

**步骤 4：复跑测试确认通过**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_triggers.py -v`  
预期：PASS

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py /Users/huangyuxiang/openbb-agent/tests/test_radar_triggers.py
git commit -m "feat: add mixed trigger engine for stock radar"
```

### 任务 5：实现中等风控扣分 + 8-15 候选筛选

**文件：**
- 修改：`/Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py`
- 新建：`/Users/huangyuxiang/openbb-agent/tests/test_radar_selection.py`

**步骤 1：先写失败测试**

```python
# /Users/huangyuxiang/openbb-agent/tests/test_radar_selection.py
import unittest
import pandas as pd
import fetch_equities_fmp as m

class TestRadarSelection(unittest.TestCase):
    def test_candidate_count_respects_bounds(self):
        df = pd.DataFrame([{"symbol": f"S{i}", "opportunity_score": 100-i, "trigger_type": "pullback"} for i in range(30)])
        out = m.select_radar_candidates(df, min_n=8, max_n=15)
        self.assertGreaterEqual(len(out), 8)
        self.assertLessEqual(len(out), 15)

if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行并确认失败**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_selection.py -v`  
预期：FAIL（函数不存在）

**步骤 3：最小实现**

新增：
- `compute_risk_penalty(drawdown_60d, atr14, cfg)`
- `score_opportunity(...)`
- `select_radar_candidates(df, min_n, max_n)`

要求：
- 日候选保持 8-15
- 在有数据前提下尽量保障 US/HK/CN 至少各 1 条

**步骤 4：复跑测试确认通过**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_selection.py -v`  
预期：PASS

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py /Users/huangyuxiang/openbb-agent/tests/test_radar_selection.py
git commit -m "feat: add risk-penalized radar ranking and candidate selection"
```

### 任务 6：雷达结果落地（CSV + Supabase）

**文件：**
- 修改：`/Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py`
- 新建：`/Users/huangyuxiang/openbb-agent/docs/sql/2026-02-24-create-equity-opportunity-radar.sql`
- 修改：`/Users/huangyuxiang/openbb-agent/supabase_history_fields.txt`
- 新建：`/Users/huangyuxiang/openbb-agent/tests/test_radar_output_schema.py`

**步骤 1：先写失败测试**

```python
# /Users/huangyuxiang/openbb-agent/tests/test_radar_output_schema.py
import unittest
import pandas as pd

class TestRadarOutputSchema(unittest.TestCase):
    def test_columns(self):
        cols = ["symbol","market","trigger_type","trigger_price","stop_price","opportunity_score","reason_1line"]
        df = pd.DataFrame(columns=cols)
        self.assertTrue(set(cols).issubset(df.columns))

if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行并确认失败**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_output_schema.py -v`  
预期：FAIL（将测试接入真实导出结果前会失败）

**步骤 3：最小实现**

- 输出 CSV：`/Users/huangyuxiang/openbb-agent/openbb_outputs/equity_opportunity_radar.csv`
- 新增 env/table 常量：`SUPABASE_RADAR_TABLE`（默认 `equity_opportunity_radar`）
- upsert 冲突键：`symbol,as_of_date`
- 提供建表 SQL 与索引（`as_of_date`、`market`、`opportunity_score`）

**步骤 4：复跑测试确认通过**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_radar_output_schema.py -v`  
预期：PASS

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py /Users/huangyuxiang/openbb-agent/docs/sql/2026-02-24-create-equity-opportunity-radar.sql /Users/huangyuxiang/openbb-agent/supabase_history_fields.txt /Users/huangyuxiang/openbb-agent/tests/test_radar_output_schema.py
git commit -m "feat: persist opportunity radar output to csv and supabase"
```

### 任务 7：改造 Streamlit 首页为“可执行雷达”

**文件：**
- 修改：`/Users/huangyuxiang/openbb-agent/streamlit_app.py`
- 新建：`/Users/huangyuxiang/openbb-agent/tests/test_streamlit_radar_block.py`

**步骤 1：先写失败测试**

```python
# /Users/huangyuxiang/openbb-agent/tests/test_streamlit_radar_block.py
import unittest
from pathlib import Path

class TestStreamlitRadarBlock(unittest.TestCase):
    def test_has_radar_section_text(self):
        src = Path("/Users/huangyuxiang/openbb-agent/streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("今日机会雷达", src)
        self.assertIn("机会类型", src)
        self.assertIn("触发价", src)
        self.assertIn("止损价", src)

if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行并确认失败**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_streamlit_radar_block.py -v`  
预期：FAIL

**步骤 3：最小实现**

新增 `load_radar()`（Supabase 优先，CSV 回退），并在股票首页顶部增加：
- KPI 卡片：`候选数`、`回踩数`、`突破数`、`平均机会分`
- 雷达主表：`市场, 标的, 机会类型, 触发价, 止损价, 机会分, 风险标记, 一句话理由`
- 原有深度图表下移，不删除

**步骤 4：复跑测试确认通过**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_streamlit_radar_block.py -v`  
预期：PASS

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/streamlit_app.py /Users/huangyuxiang/openbb-agent/tests/test_streamlit_radar_block.py
git commit -m "feat: add stock opportunity radar first-screen module"
```

### 任务 8：修复当前股票链路可靠性问题

**文件：**
- 修改：`/Users/huangyuxiang/openbb-agent/streamlit_app.py`
- 新建：`/Users/huangyuxiang/openbb-agent/tests/test_load_history_fallback.py`

**步骤 1：先写失败测试**

```python
# /Users/huangyuxiang/openbb-agent/tests/test_load_history_fallback.py
import unittest
import pandas as pd
import streamlit_app as app

class TestLoadHistoryFallback(unittest.TestCase):
    def test_load_history_returns_dataframe(self):
        df = app.load_history.__wrapped__()
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行并确认失败**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_load_history_fallback.py -v`  
预期：FAIL（当前 `load_history` 分支有变量路径隐患）

**步骤 3：最小实现**

- `load_history()` 一开始初始化 `df = pd.DataFrame()`
- 显式分支：
  - Supabase 成功且字段可用 -> 返回
  - Supabase 空/字段不全 -> fallback 本地 CSV
  - 本地 CSV 不存在 -> 返回空 DataFrame + warning

**步骤 4：复跑测试确认通过**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_load_history_fallback.py -v`  
预期：PASS

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/streamlit_app.py /Users/huangyuxiang/openbb-agent/tests/test_load_history_fallback.py
git commit -m "fix: harden stock history loading fallback paths"
```

### 任务 9：文档与日常运行手册

**文件：**
- 修改：`/Users/huangyuxiang/openbb-agent/README.md`
- 新建：`/Users/huangyuxiang/openbb-agent/docs/stock_radar_runbook.md`
- 修改：`/Users/huangyuxiang/openbb-agent/.env.example`（若不存在则在 README 补齐）

**步骤 1：先设失败检查清单**

手动检查失败条件：
- 缺少 `SUPABASE_RADAR_TABLE`
- 缺少每日运行命令
- 缺少回踩/突破解释

**步骤 2：执行检查并确认不通过**

运行：`rg -n "SUPABASE_RADAR_TABLE|机会雷达|pullback|breakout" /Users/huangyuxiang/openbb-agent/README.md /Users/huangyuxiang/openbb-agent/docs/stock_radar_runbook.md`  
预期：修改前关键词不全

**步骤 3：最小实现**

- README 改为股票单一化快速开始。
- Runbook 增加：
  - 每日任务命令
  - 故障排查
  - 字段字典
  - 典型决策流程示例

**步骤 4：复查通过**

运行：同上 `rg` 命令  
预期：关键词齐全

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/README.md /Users/huangyuxiang/openbb-agent/docs/stock_radar_runbook.md
git commit -m "docs: add stock-only radar runbook and operations guide"
```

### 任务 10：开盘邮件自动推送机制

**文件：**
- 新建：`/Users/huangyuxiang/openbb-agent/scripts/send_radar_email.py`
- 新建：`/Users/huangyuxiang/openbb-agent/.github/workflows/opening-radar-email.yml`
- 修改：`/Users/huangyuxiang/openbb-agent/README.md`
- 新建：`/Users/huangyuxiang/openbb-agent/tests/test_send_radar_email.py`

**步骤 1：先写失败测试**

```python
# /Users/huangyuxiang/openbb-agent/tests/test_send_radar_email.py
import unittest
import pandas as pd
import scripts.send_radar_email as m

class TestSendRadarEmail(unittest.TestCase):
    def test_build_html_contains_required_columns(self):
        df = pd.DataFrame([{
            "symbol": "NVDA", "market": "US", "trigger_type": "breakout",
            "trigger_price": 100.0, "stop_price": 95.0,
            "opportunity_score": 88.0, "reason_1line": "放量突破20日新高"
        }])
        html = m.build_email_html(df, as_of_date="2026-02-24")
        self.assertIn("机会类型", html)
        self.assertIn("触发价", html)
        self.assertIn("止损价", html)
        self.assertIn("NVDA", html)

if __name__ == "__main__":
    unittest.main()
```

**步骤 2：运行并确认失败**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_send_radar_email.py -v`  
预期：FAIL（脚本尚未实现）

**步骤 3：最小实现**

- 在 `send_radar_email.py` 实现：
  - 读取 `openbb_outputs/equity_opportunity_radar.csv`（无则回退 summary）
  - 构建邮件 HTML（按市场分组展示 Top 候选）
  - SMTP 发送（SSL/TLS）
  - 防重复发送（本地日志文件记录 `date+market` 或统一 `date`）
  - 开盘窗口判断（`Asia/Shanghai`、`Asia/Hong_Kong`、`America/New_York` 的 09:20-09:35）
- 新增环境变量：
  - `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`
  - `MAIL_FROM`, `MAIL_TO`
  - `RADAR_MAIL_SUBJECT_PREFIX`
- 新增 GitHub Actions 定时任务：
  - 工作日高频触发（例如每 15 分钟）
  - 作业中执行窗口判断，非开盘窗口直接退出不发送

**步骤 4：复跑测试确认通过**

运行：`python3 -m unittest /Users/huangyuxiang/openbb-agent/tests/test_send_radar_email.py -v`  
预期：PASS

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/scripts/send_radar_email.py /Users/huangyuxiang/openbb-agent/.github/workflows/opening-radar-email.yml /Users/huangyuxiang/openbb-agent/README.md /Users/huangyuxiang/openbb-agent/tests/test_send_radar_email.py
git commit -m "feat: add opening-hour radar email push automation"
```

### 任务 11：端到端验收与上线门禁

**文件：**
- 修改：`/Users/huangyuxiang/openbb-agent/docs/stock_radar_runbook.md`（补充验收证据）

**步骤 1：运行单元测试**

运行：`python3 -m unittest discover -s /Users/huangyuxiang/openbb-agent/tests -p "test_*.py" -v`  
预期：PASS

**步骤 2：运行数据管道**

运行：`python3 /Users/huangyuxiang/openbb-agent/fetch_equities_fmp.py`  
预期输出：
- `/Users/huangyuxiang/openbb-agent/openbb_outputs/three_month_summary.csv`
- `/Users/huangyuxiang/openbb-agent/openbb_outputs/three_month_close_history.csv`
- `/Users/huangyuxiang/openbb-agent/openbb_outputs/equity_opportunity_radar.csv`

**步骤 3：运行应用冒烟测试**

运行：`streamlit run /Users/huangyuxiang/openbb-agent/streamlit_app.py`  
预期：
- 首页显示“今日机会雷达”
- 无加密 tab
- 数据充足时候选数在 8-15

**步骤 4：留存验证证据**

- 将命令输出与截图写入 runbook 验证章节。

**步骤 5：提交**

```bash
git add /Users/huangyuxiang/openbb-agent/docs/stock_radar_runbook.md
git commit -m "chore: add stock radar verification evidence"
```

## 本期锁定业务规则（不再漂移）

- 市场范围：US + HK + CN。
- 交易框架：2-8 周波段。
- 每日候选：8-15。
- 触发模式：回踩 + 突破混合。
- 风控模式：中等风控（扣分，不做大面积一票否决；仅数据异常时剔除）。
- 本期不做：LLM 排名解释层、自动下单执行。
