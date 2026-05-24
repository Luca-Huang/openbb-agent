# Source Matrix

The pipeline has two data sources, isolated behind `data_sources/`:

- **Longbridge CLI** — primary market-data source (all markets). Subprocess +
  JSON; OAuth token at `~/.longbridge/openapi/tokens/<client_id>`. Module:
  `research_workbench.data_sources.longbridge`.
- **AKShare via AKTools HTTP** — A-share deep fundamentals + events that
  Longbridge OpenAPI does not cover. Local FastAPI service started with
  `python -m aktools` (port 8080); URL overrideable via the `AKTOOLS_URL`
  env var. Module: `research_workbench.data_sources.akshare_cn`.

## What each source covers

| Data need | Source | CLI / function |
|---|---|---|
| OHLCV history (CN / HK / US) | Longbridge | `longbridge kline history` |
| Static equity reference (EPS / BPS / shares / dividend) | Longbridge | `longbridge static` |
| Current PE / PB / div yield / market cap / turnover rate | Longbridge | `longbridge calc-index` |
| **Today's capital-flow snapshot (大/中/小单 in/out)** | Longbridge | `longbridge capital` |
| Annual income / balance / cash-flow statements (CN) | AKShare (sina) | `fetch_annual_financials` |
| **All-period (quarterly + annual) financials (CN)** | AKShare (sina) | `fetch_financials_all` |
| **Daily historical PE / PB / PS / PEG series (CN)** | AKShare (eastmoney) | `fetch_historical_valuation` |
| **Earnings preannouncement 业绩预告 (CN)** | AKShare (eastmoney) | `fetch_earnings_preannouncement` + market batch |
| **Earnings express 业绩快报 (CN)** | AKShare (eastmoney) | `fetch_earnings_express` + market batch |
| Industry classification + peer PE median (CN) | AKShare (eastmoney) | `fetch_industry_classification` + `fetch_industry_peers` |
| Dividend history (CN) | AKShare (sina) | `fetch_dividend_history` |
| Shareholder buy/sell records (CN) | AKShare (同花顺) | `fetch_shareholder_changes` |
| **Northbound (沪深港通) holdings daily series (CN)** | AKShare (eastmoney) | `fetch_northbound_holding` |
| Local watchlist metadata | local JSON | `research_workbench.ingestion.watchlist` |
| Manual events (analyst notes) | local CSV | `research_workbench.outputs.files.load_manual_events` |

A-share-only data is dispatched through `pipelines/refresh.fetch_cn_enrichment`;
both the per-symbol fetches and the two market-wide tables (yjyg + yjbb) are
called there.

## Coverage gaps

Longbridge OpenAPI's A-share dataset is intentionally shallow (no income
statements, no historical PE, no dividend history, no industry peers, no
preannouncements). All of those rely on AKShare. If AKTools is unreachable
the pipeline degrades gracefully — Longbridge data still flows, and the
five CN-enrichment-backed scores fall to zero.

`news` / `filing` / `topic` on Longbridge return `403 Target API is not in
authorized scope` for our CLI's OAuth scopes (4+6+10+11); the CLI binary
hard-codes those, so re-authorizing does not help. AKShare news endpoints
are an alternative if those signals ever matter.

## Policy

- Do not add a third market-data provider without isolating it behind
  `data_sources/` with its own provider class.
- Do not fetch live prices from web search, UI code, or scripts outside
  the `pipelines/` layer.
- Keep provider-specific field names inside the data-source adapter; the
  schema in `research_store/schema.py` is the canonical contract.
- Per-symbol fetchers that download market-wide tables (yjyg, yjbb) must
  expose a batch / `market_df` parameter so the pipeline can amortize the
  download across the whole watchlist.
