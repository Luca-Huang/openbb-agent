"""PE 历史校准 — 给 checklist 表 3 第 1/6 题提供数据基础。

用途：
- 输出某标的历史 PE 区间 + 当前所处分位
- 按 PE 区间分桶，算"在某 PE 桶入场，6/12 个月后的回报"
- 校准 checklist 里"退出 PE 倍数"假设是否合理

用法：
    python scripts/checklist/pe_calibration.py 0700.HK
    python scripts/checklist/pe_calibration.py 0700.HK 1810.HK AAPL

数据源：openbb_outputs/supabase_equity_history_backfill.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIST_PATH = PROJECT_ROOT / "openbb_outputs" / "supabase_equity_history_backfill.csv"


def load_history(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(HIST_PATH)
    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        return df
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    df = df.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["pe"] = pd.to_numeric(df["pe"], errors="coerce")
    df["ttm_eps"] = pd.to_numeric(df["ttm_eps"], errors="coerce")
    return df


def pe_summary(df: pd.DataFrame) -> dict:
    pe = df["pe"].dropna()
    if pe.empty:
        return {}
    latest_pe = df.dropna(subset=["pe"]).iloc[-1]["pe"]
    return {
        "覆盖天数": len(df),
        "有 PE 数据天数": len(pe),
        "最早日期": df["as_of_date"].min().date(),
        "最新日期": df["as_of_date"].max().date(),
        "PE 最小": round(pe.min(), 2),
        "PE p25": round(pe.quantile(0.25), 2),
        "PE 中位": round(pe.median(), 2),
        "PE p75": round(pe.quantile(0.75), 2),
        "PE 最大": round(pe.max(), 2),
        "当前 PE": round(latest_pe, 2),
        "当前分位": f"{round((pe < latest_pe).mean() * 100, 1)}%",
    }


def forward_return(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Compute forward-looking return over horizon_days for every row."""
    close = df["close"].copy()
    fwd = close.shift(-horizon_days)
    return (fwd / close - 1) * 100  # in percent


def pe_bucket_returns(df: pd.DataFrame, horizon_days: int, horizon_label: str) -> pd.DataFrame:
    valid = df.dropna(subset=["pe", "close"]).copy()
    valid["fwd_ret"] = forward_return(valid, horizon_days)
    valid = valid.dropna(subset=["fwd_ret"])  # drop tail rows without forward window
    if valid.empty:
        return pd.DataFrame()

    # bucket by PE quartiles of the historical distribution
    quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    edges = valid["pe"].quantile(quantiles).values
    # de-dup edges in case of low variance
    edges = sorted(set(np.round(edges, 4)))
    if len(edges) < 3:
        return pd.DataFrame()

    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"PE [{edges[i]:.1f}–{edges[i + 1]:.1f}]")
    valid["bucket"] = pd.cut(valid["pe"], bins=edges, labels=labels, include_lowest=True)

    out = valid.groupby("bucket", observed=True)["fwd_ret"].agg(
        样本数="count",
        平均回报=lambda s: round(s.mean(), 2),
        中位回报=lambda s: round(s.median(), 2),
        最差回报=lambda s: round(s.min(), 2),
        最好回报=lambda s: round(s.max(), 2),
    )
    out.columns = pd.MultiIndex.from_product([[horizon_label], out.columns])
    return out


def calibrate(symbol: str) -> None:
    df = load_history(symbol)
    print(f"\n{'=' * 70}")
    print(f"标的：{symbol}")
    print("=" * 70)

    if df.empty:
        print(f"  [警告] 历史数据中未找到 {symbol}")
        return

    summary = pe_summary(df)
    if not summary:
        print("  [警告] 该标的没有 PE 字段数据")
        return

    print("\n[PE 分布]")
    for k, v in summary.items():
        print(f"  {k:15s} {v}")

    print("\n[PE 桶 vs 持有 6 个月回报]")
    r6 = pe_bucket_returns(df, horizon_days=126, horizon_label="6mo")
    if r6.empty:
        print("  [跳过] 样本不足")
    else:
        print(r6.to_string())

    print("\n[PE 桶 vs 持有 12 个月回报]")
    r12 = pe_bucket_returns(df, horizon_days=252, horizon_label="12mo")
    if r12.empty:
        print("  [跳过] 数据不够 12 个月前向窗口")
    else:
        print(r12.to_string())

    # 提示：当前 PE 落在哪个历史桶
    pe_vals = df["pe"].dropna()
    latest_pe = df.dropna(subset=["pe"]).iloc[-1]["pe"]
    if not pe_vals.empty:
        q25, q50, q75 = pe_vals.quantile([0.25, 0.5, 0.75])
        if latest_pe < q25:
            zone = "低位（< p25）"
        elif latest_pe < q50:
            zone = "偏低（p25-p50）"
        elif latest_pe < q75:
            zone = "偏高（p50-p75）"
        else:
            zone = "高位（> p75）"
        print(f"\n[校准结论] 当前 PE {latest_pe:.2f} 处于 {zone}")
        print(f"  → checklist 表 3 第 1 题选择'退出 PE'倍数时，参考此分位")
        print(f"  → 不建议在'退出 PE'倍数填超过历史 p75（{q75:.2f}）的值，那是 wishful thinking")


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python scripts/checklist/pe_calibration.py <SYMBOL> [SYMBOL2] ...")
        print("示例：python scripts/checklist/pe_calibration.py 0700.HK 1810.HK")
        sys.exit(1)

    for symbol in sys.argv[1:]:
        calibrate(symbol.upper())


if __name__ == "__main__":
    main()
