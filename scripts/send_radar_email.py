#!/usr/bin/env python3
"""Send opening-window stock opportunity radar email via SMTP."""

from __future__ import annotations

import json
import os
import smtplib
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
                "opportunity_score": summary.get("value_score"),
                "risk_flags": "",
                "reason_1line": summary.get("entry_recommendation").fillna("候选观察"),
            }
        )
    else:
        return pd.DataFrame()

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    if "market" in df.columns:
        df["market"] = df["market"].astype(str).str.upper()
    return df


def build_email_html(df: pd.DataFrame, as_of_date: str) -> str:
    if df.empty:
        return f"<p>{as_of_date} 无可发送的机会雷达候选。</p>"

    display = df.copy()
    for col in ["trigger_price", "stop_price", "opportunity_score"]:
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
            "opportunity_score": "机会分",
            "risk_flags": "风险标记",
            "reason_1line": "一句话理由",
        }
    )

    for col in ["市场", "标的", "机会类型", "触发价", "止损价", "机会分", "风险标记", "一句话理由"]:
        if col not in display.columns:
            display[col] = ""

    table = display[
        ["市场", "标的", "机会类型", "触发价", "止损价", "机会分", "风险标记", "一句话理由"]
    ].to_html(index=False, border=0)

    return f"""
    <html>
      <body>
        <h2>开盘机会雷达 - {as_of_date}</h2>
        <p>以下为开盘窗口内可执行候选，请结合仓位纪律与止损规则执行。</p>
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
