from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload = payload.get("watchlist", payload.get("equities", []))
    if not isinstance(payload, list):
        raise ValueError(f"Watchlist payload must be a list: {path}")
    return payload


def load_watchlist(
    watchlist_path: Path,
    fallback_equity_config_path: Path | None = None,
    market: str = "CN",
) -> pd.DataFrame:
    entries: list[dict] = []
    if watchlist_path.exists():
        entries = _load_json(watchlist_path)
    elif fallback_equity_config_path and fallback_equity_config_path.exists():
        entries = _load_json(fallback_equity_config_path)

    if not entries:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "market",
                "sector",
                "status",
                "target_zone_low",
                "target_zone_high",
                "notes",
            ]
        )

    normalized: list[dict] = []
    for item in entries:
        item_market = str(item.get("market", market)).upper()
        if item_market != market.upper():
            continue
        normalized.append(
            {
                "symbol": str(item.get("symbol", "")).upper(),
                "name": item.get("name") or item.get("name_cn") or item.get("name_en") or item.get("symbol", ""),
                "market": item_market,
                "sector": item.get("sector", ""),
                "status": item.get("status", "watch"),
                "target_zone_low": pd.to_numeric(item.get("target_zone_low"), errors="coerce"),
                "target_zone_high": pd.to_numeric(item.get("target_zone_high"), errors="coerce"),
                "notes": item.get("notes", ""),
            }
        )
    return pd.DataFrame(normalized)


def watchlist_symbols(watchlist_df: pd.DataFrame) -> list[str]:
    if watchlist_df.empty or "symbol" not in watchlist_df.columns:
        return []
    return watchlist_df["symbol"].dropna().astype(str).str.upper().tolist()


def filter_to_watchlist(df: pd.DataFrame, watchlist_df: pd.DataFrame) -> pd.DataFrame:
    symbols = set(watchlist_symbols(watchlist_df))
    if not symbols or "symbol" not in df.columns:
        return df.copy()
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    return out[out["symbol"].isin(symbols)]


def merge_watchlist_metadata(df: pd.DataFrame, watchlist_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or watchlist_df.empty or "symbol" not in df.columns:
        return df.copy()
    return df.merge(
        watchlist_df.drop_duplicates("symbol"),
        on="symbol",
        how="left",
        suffixes=("", "_watch"),
    )

