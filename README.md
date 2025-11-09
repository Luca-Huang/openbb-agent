# OpenBB Agent

这个仓库包含一个简单的 OpenBB 分析脚本 `openbb_three_months.py`，用于抓取完美世界、小米集团、Meta 在最近三个月的股价走势，并输出归一化收盘价、TTM EPS、PE 以及历史分位等指标。

## 使用方法
1. 安装依赖：
   ```bash
   pip install openbb yfinance pandas numpy
   ```
2. 运行脚本：
   ```bash
   python3 openbb_three_months.py
   ```
3. 结果会写入 `openbb_outputs/` 文件夹，包含：
   - `three_month_close_history.csv`
   - `three_month_summary.csv`

如需调整股票或时间窗口，可编辑脚本顶部的 `TICKERS`、`LOOKBACK_DAYS` 等配置。
