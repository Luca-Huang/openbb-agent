import requests
from pathlib import Path

BARK_KEYS = [
    "q9XsN49BXTDzGCo7T4mb6J",   # iPhone
    "mxf7TVDwRzDKqQ6VSQ4LMa",   # Mac
]

report = Path("report.md").read_text(encoding="utf-8").strip()
lines = report.split("\n", 1)
title = lines[0].strip()
body  = lines[1].strip() if len(lines) > 1 else ""

print(f"[push] title length: {len(title)}")
print(f"[push] body length:  {len(body)}")
print(f"[push] body preview: {body[:120]}..." if len(body) > 120 else f"[push] body preview: {body}")

payload = {
    "title": title,
    "body": body,
    "group": "持仓复盘",
    "url": "https://claude.ai/code/routines/trig_01GbYsaLNaNmTqjQdD3X3KAG",
    "level": "timeSensitive",
    "sound": "glass",
}

for key in BARK_KEYS:
    try:
        r = requests.post(f"https://api.day.app/{key}", json=payload, timeout=15)
        print(f"[push] {key[:8]}... → HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e:
        print(f"[push] {key[:8]}... → FAILED: {e}")
