# Source Matrix

Longbridge CLI is the active market-data source for this project.

| Data Need | Active Source | Code Boundary |
|---|---|---|
| CN / HK / US quotes and history | `longbridge kline history` | `research_workbench.data_sources.longbridge` |
| Static equity fields | `longbridge static` | `research_workbench.data_sources.longbridge` |
| PE, PB, dividend yield, market cap | `longbridge calc-index` | `research_workbench.data_sources.longbridge` |
| Local watchlist metadata | `research_inputs/watchlist_cn.json` | `research_workbench.ingestion.watchlist` |
| Manual events | `research_inputs/manual_events.csv` | `research_workbench.outputs.files` |

## Policy

- Do not add a second market-data provider without isolating it behind `data_sources/`.
- Do not fetch live prices from web search or UI code.
- Keep provider-specific field names inside the data-source adapter.
