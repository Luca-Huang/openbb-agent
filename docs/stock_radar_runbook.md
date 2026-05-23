# Stock Radar Runbook

## Daily Refresh

```bash
python scripts/refresh.py
```

The refresh command uses Longbridge CLI through `research_workbench.data_sources.longbridge`.

Expected generated files:

- `outputs/research_data/three_month_close_history.csv`
- `outputs/research_data/three_month_summary.csv`
- `outputs/research_data/signals_snapshot.csv`

## Optional Email

```bash
python scripts/send_radar_email.py
```

Required environment variables:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASS`
- `MAIL_FROM`
- `MAIL_TO`

## Troubleshooting

1. Run `longbridge check --format json` and confirm quote access works.
2. Run `python scripts/refresh.py --only history` to isolate data acquisition.
3. Run `python scripts/refresh.py --no-fetch --only signals` to isolate analysis from data acquisition.
4. Check `outputs/research_data/` for generated CSVs.
