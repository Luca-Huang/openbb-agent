#!/usr/bin/env python3
"""Send opening-window stock opportunity radar email via SMTP."""

from __future__ import annotations

import json
import logging
import os
import smtplib
import sys
from datetime import datetime, time, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Set
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RADAR_PATH = ROOT / "openbb_outputs" / "equity_opportunity_radar.csv"
SUMMARY_PATH = ROOT / "openbb_outputs" / "three_month_summary.csv"
SEND_LOG_PATH = ROOT / "openbb_outputs" / "radar_email_sent_log.json"
HOLDINGS_PATH = ROOT / "research_inputs" / "holdings.json"
HOLDINGS_SNAPSHOT_DIR = ROOT / "outputs" / "holdings_snapshots"

sys.path.insert(0, str(ROOT / "src"))
log = logging.getLogger("send_radar_email")

DEFAULT_MAIL_TO = "794166954@qq.com"

OPENING_WINDOWS: Dict[str, Dict[str, object]] = {
    "CN": {"tz": "Asia/Shanghai", "start": time(9, 20), "end": time(9, 35)},
    "HK": {"tz": "Asia/Hong_Kong", "start": time(9, 20), "end": time(9, 35)},
    "US": {"tz": "America/New_York", "start": time(9, 20), "end": time(9, 35)},
}


def active_opening_markets(now_utc: datetime | None = None) -> List[str]:
    now_utc = now_utc or datetime.now(timezone.utc)
    active: List[str] = []
    for market, cfg in OPENING_WINDOWS.items():
        local_dt = now_utc.astimezone(ZoneInfo(str(cfg["tz"])))
        if local_dt.weekday() > 4:
            continue
        local_time = local_dt.time()
        start: time = cfg["start"]  # type: ignore[assignment]
        end: time = cfg["end"]  # type: ignore[assignment]
        if start <= local_time <= end:
            active.append(market)
    return active


def load_send_log(path: Path = SEND_LOG_PATH) -> Set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if isinstance(payload, list):
        return set(str(x) for x in payload)
    return set()


def save_send_log(records: Set[str], path: Path = SEND_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(records), ensure_ascii=False, indent=2), encoding="utf-8")


def load_radar_dataframe() -> pd.DataFrame:
    if RADAR_PATH.exists():
        df = pd.read_csv(RADAR_PATH)
    elif SUMMARY_PATH.exists():
        summary = pd.read_csv(SUMMARY_PATH)
        if summary.empty:
            return pd.DataFrame()
        summary = summary.sort_values("value_score", ascending=False).head(12)
        df = pd.DataFrame(
            {
                "as_of_date": datetime.now().date().isoformat(),
                "symbol": summary.get("symbol"),
                "market": summary.get("market"),
                "trigger_type": "watchlist",
                "trigger_price": summary.get("end_close"),
                "stop_price": summary.get("support_level_primary"),
                "take_profit_1": "",
                "take_profit_2": "",
                "trailing_stop": "",
                "opportunity_score": summary.get("value_score"),
                "risk_flags": "",
                "reason_1line": summary.get("entry_recommendation").fillna("候选观察"),
                "exit_plan": "",
            }
        )
    else:
        return pd.DataFrame()

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    if "market" in df.columns:
        df["market"] = df["market"].astype(str).str.upper()
    return df


def _fetch_history_for_holding(code: str, lookback_days: int = 400) -> pd.DataFrame:
    """Pull daily OHLCV via akshare and bolt on the indicator schema the
    backtest module needs.  Imports inside the function so the email path
    keeps working when akshare/network is unavailable for the watchlist
    section.
    """
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
             "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    import urllib.request as _u
    _u.getproxies = lambda: {}

    import akshare as ak
    import numpy as np
    from research_workbench.signal_engine.radar import add_radar_features

    end = datetime.now().date()
    start = (end - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    raw = ak.stock_zh_a_hist(symbol=code, period="daily",
                             start_date=start, end_date=end.strftime("%Y%m%d"),
                             adjust="qfq")
    raw.columns = ["date", "code", "open", "close", "high", "low",
                   "volume", "amount", "amplitude", "pct_chg", "chg", "turnover"]
    raw["date"] = pd.to_datetime(raw["date"])
    for c in ("open", "close", "high", "low", "volume"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["symbol"] = code
    d = raw.sort_values("date").copy()

    # Mirror fetch_cn_data.py upstream indicator schema
    close = d["close"]
    d["ma50"] = close.rolling(50, min_periods=1).mean()
    d["ma200"] = close.rolling(200, min_periods=1).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - 100 / (1 + rs)
    rolling_min = close.rolling(20, min_periods=1).min()
    d["support_level_primary"] = rolling_min
    d["support_level_secondary"] = rolling_min * 1.1
    d["volume_ma20"] = d["volume"].rolling(20, min_periods=1).mean()
    d["volume_spike_ratio"] = d["volume"] / d["volume_ma20"]
    return add_radar_features(d)


def build_holdings_section_html() -> str:
    """Best-effort holdings section. Returns empty string on any failure so a
    transient akshare hiccup never blocks the main radar email.

    Set ``RADAR_SKIP_HOLDINGS=1`` to disable (used by tests / CI to avoid
    pulling network data).
    """
    if os.environ.get("RADAR_SKIP_HOLDINGS"):
        return ""
    if not HOLDINGS_PATH.exists():
        return ""
    try:
        from research_workbench.signal_engine.holdings_tracker import (
            analyze_holding, load_holdings, render_holding_cards_html,
            save_snapshot,
        )
        positions = load_holdings(HOLDINGS_PATH)
        if not positions:
            return ""
        snapshots = []
        for p in positions:
            try:
                history = _fetch_history_for_holding(p.code)
                snapshots.append(analyze_holding(history, p))
            except Exception as exc:  # noqa: BLE001 — keep email resilient
                log.warning("holdings analysis failed for %s: %s", p.code, exc)
        # Persist today's analysis so the next UI / email run can render a
        # 'vs 昨日' diff.  Best-effort: a save failure must not block the email.
        try:
            if snapshots:
                save_snapshot(snapshots, HOLDINGS_SNAPSHOT_DIR)
        except Exception as exc:  # noqa: BLE001
            log.warning("holdings snapshot save failed: %s", exc)
        return render_holding_cards_html(snapshots)
    except Exception as exc:  # noqa: BLE001
        log.warning("holdings section build failed: %s", exc)
        return ""


def build_email_html(df: pd.DataFrame, as_of_date: str) -> str:
    if df.empty:
        return f"<p>{as_of_date} 无可发送的机会雷达候选。</p>"

    display = df.copy()
    for col in ["trigger_price", "stop_price", "take_profit_1", "take_profit_2", "trailing_stop", "opportunity_score"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce")
    display["trigger_type"] = display.get("trigger_type", "").map(
        {"pullback": "回踩", "breakout": "突破", "watchlist": "观察"}
    ).fillna(display.get("trigger_type", ""))
    display = display.rename(
        columns={
            "market": "市场",
            "symbol": "标的",
            "trigger_type": "机会类型",
            "trigger_price": "触发价",
            "stop_price": "止损价",
            "take_profit_1": "止盈1",
            "take_profit_2": "止盈2",
            "trailing_stop": "移动止盈",
            "opportunity_score": "机会分",
            "risk_flags": "风险标记",
            "reason_1line": "一句话理由",
            "exit_plan": "止盈说明",
        }
    )

    columns = ["市场", "标的", "机会类型", "触发价", "止损价", "止盈1", "止盈2", "移动止盈", "机会分", "风险标记", "一句话理由", "止盈说明"]
    for col in columns:
        if col not in display.columns:
            display[col] = ""

    table = display[columns].to_html(index=False, border=0)
    holdings_section = build_holdings_section_html()

    return f"""
    <html>
      <body>
        <h2>开盘机会雷达 - {as_of_date}</h2>
        {holdings_section}
        <h3 style="margin-bottom:4px;">机会雷达候选</h3>
        <p style="color:#666;font-size:12px;margin-top:0;">
          以下为开盘窗口内可执行候选，请结合仓位纪律、止损规则与分批止盈计划执行。
        </p>
        {table}
      </body>
    </html>
    """.strip()


def send_email(subject: str, html_body: str) -> None:
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    mail_from = os.environ.get("MAIL_FROM", smtp_user or "")
    mail_to = os.environ.get("MAIL_TO", DEFAULT_MAIL_TO)

    missing = [
        key
        for key, value in {
            "SMTP_HOST": smtp_host,
            "SMTP_USER": smtp_user,
            "SMTP_PASS": smtp_pass,
            "MAIL_FROM": mail_from,
            "MAIL_TO": mail_to,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing email env vars: {', '.join(missing)}")

    recipients = [x.strip() for x in str(mail_to).split(",") if x.strip()]
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = str(mail_from)
    msg["To"] = ", ".join(recipients)

    if smtp_port == 465:
        with smtplib.SMTP_SSL(str(smtp_host), smtp_port, timeout=30) as server:
            server.login(str(smtp_user), str(smtp_pass))
            server.sendmail(str(mail_from), recipients, msg.as_string())
    else:
        with smtplib.SMTP(str(smtp_host), smtp_port, timeout=30) as server:
            server.starttls()
            server.login(str(smtp_user), str(smtp_pass))
            server.sendmail(str(mail_from), recipients, msg.as_string())


def main() -> int:
    active_markets = active_opening_markets()
    if not active_markets:
        print("[email] Not in any opening window, skip.")
        return 0

    radar_df = load_radar_dataframe()
    if radar_df.empty:
        print("[email] Radar dataset is empty, skip.")
        return 0

    as_of = None
    if "as_of_date" in radar_df.columns:
        series = pd.to_datetime(radar_df["as_of_date"], errors="coerce").dropna()
        if not series.empty:
            as_of = series.max().date().isoformat()
    if not as_of:
        as_of = datetime.now().date().isoformat()

    sent_log = load_send_log()
    unsent_markets = [m for m in active_markets if f"{as_of}:{m}" not in sent_log]
    if not unsent_markets:
        print(f"[email] Already sent for {as_of} / {active_markets}, skip.")
        return 0

    send_df = radar_df
    if "market" in radar_df.columns:
        scoped = radar_df[radar_df["market"].isin(unsent_markets)]
        if not scoped.empty:
            send_df = scoped

    html_body = build_email_html(send_df, as_of_date=as_of)
    subject_prefix = os.environ.get("RADAR_MAIL_SUBJECT_PREFIX", "[Stock Radar]")
    subject = f"{subject_prefix} 开盘机会雷达 {as_of} ({'/'.join(unsent_markets)})"
    send_email(subject, html_body)

    for market in unsent_markets:
        sent_log.add(f"{as_of}:{market}")
    save_send_log(sent_log)
    print(f"[email] Sent radar email to {os.environ.get('MAIL_TO', DEFAULT_MAIL_TO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
