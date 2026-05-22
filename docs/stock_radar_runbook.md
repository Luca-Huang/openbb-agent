# 股票机会雷达运行手册

## 1. 目标

本手册用于保障股票单一化系统稳定输出“今日机会雷达”，并支持开盘自动邮件推送。

## 2. 每日执行命令

```bash
cd /Users/huangyuxiang/openbb-agent
source .venv311/bin/activate
python fetch_equities_fmp.py
```

成功后应生成：

- `openbb_outputs/three_month_close_history.csv`
- `openbb_outputs/three_month_summary.csv`
- `openbb_outputs/equity_opportunity_radar.csv`

## 3. 环境变量

必需：

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `SUPABASE_SUMMARY_TABLE`（默认 `equity_metrics`）
- `SUPABASE_HISTORY_TABLE`（默认 `equity_metrics_history`）
- `SUPABASE_RADAR_TABLE`（默认 `equity_opportunity_radar`）

邮件推送（若启用）：

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASS`
- `MAIL_FROM`
- `MAIL_TO`

## 4. 字段解释（机会雷达）

- `trigger_type`: 机会类型
  - `pullback`: 回踩确认（靠近 MA50、站在主支撑上方、量能确认）
  - `breakout`: 放量突破（突破近 20 日高点且量能放大）
- `trigger_price`: 触发价
- `stop_price`: 止损价
- `take_profit_1`: 第一止盈位，按 1R 计算，建议先卖出 30%-50%
- `take_profit_2`: 第二止盈位，按 2R 计算，建议再卖出 20%-30%
- `trailing_stop`: 剩余仓位移动止盈位，使用 MA20 与 `20 日最高收盘 - 2 * ATR14` 的较高者
- `exit_plan`: 止盈执行说明
- `opportunity_score`: 机会分（价值分 + 触发加分 - 风险扣分）
- `risk_flags`: 风险标记（如 `drawdown60d_high`、`atr_high`）

## 5. 典型决策流程

1. 先看 `opportunity_score` 排名（默认 8-15 条）。
2. 同分时优先 `breakout` 中量能更强者。
3. 若出现 `risk_flags`，降低仓位或延迟执行。
4. 结合持仓面板中的止损位执行风险控制。
5. 到达 `take_profit_1` 先收回部分风险，到达 `take_profit_2` 再分批兑现，剩余仓位用 `trailing_stop` 跟踪趋势。

## 6. 常见故障排查

1. 雷达为空：确认 `fetch_equities_fmp.py` 执行成功且 `equity_opportunity_radar.csv` 存在。
2. Supabase 无数据：确认表结构已按 `docs/sql/2026-02-24-create-equity-opportunity-radar.sql` 创建。
3. 仅部分市场有结果：检查对应市场流动性过滤是否过严。
4. 邮件未发送：检查 SMTP 授权码、收件地址、GitHub Actions 定时任务日志。

## 7. 验证清单

- 机会雷达页面显示“今日机会雷达”。
- 表格包含“机会类型/触发价/止损价”。
- 候选数量落在配置区间（默认 8-15）。
- 数据表 `equity_opportunity_radar` 每日有新增记录。
