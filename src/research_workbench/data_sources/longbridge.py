"""Longbridge CLI backed data source for CN / HK / US equities."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import date
from typing import Any, Iterable

import pandas as pd

log = logging.getLogger("data_sources.longbridge")

LOOKBACK_DAYS = 730
SUPPORTED_MARKETS = {"CN", "HK", "US"}
DEFAULT_CLI_TIMEOUT_SECONDS = 60


class LongbridgeCLIError(RuntimeError):
    """Raised when the Longbridge CLI cannot return usable JSON data."""


def _to_float(value: Any) -> float | None:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    return float(num)


def _run_longbridge(
    args: Iterable[str],
    *,
    attempts: int = 3,
    timeout_seconds: int = DEFAULT_CLI_TIMEOUT_SECONDS,
) -> Any:
    """Run Longbridge CLI and parse JSON output, retrying rate-limit responses."""
    cmd = ["longbridge", *args, "--format", "json"]
    last_error = ""
    for attempt in range(attempts):
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise LongbridgeCLIError(
                f"Longbridge command timed out after {timeout_seconds}s: {' '.join(cmd)}"
            ) from exc

        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout or "null")
            except json.JSONDecodeError as exc:
                raise LongbridgeCLIError(f"Longbridge returned invalid JSON for {' '.join(args)}") from exc

        last_error = (proc.stderr or proc.stdout or "").strip()
        if "429002" in last_error and attempt < attempts - 1:
            time.sleep(2 * (attempt + 1))
            continue
        break

    raise LongbridgeCLIError(last_error or f"Longbridge command failed: {' '.join(cmd)}")


class LongbridgeCLIProvider:
    """Single provider for supported equity markets via `longbridge`."""

    def __init__(self, market: str):
        market = market.upper()
        if market not in SUPPORTED_MARKETS:
            raise ValueError(f"No provider for market: {market}. Available: {sorted(SUPPORTED_MARKETS)}")
        self.market = market

    def fetch_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        log.info("Longbridge: fetching history for %s (%s -> %s)", symbol, start, end)
        payload = _run_longbridge(
            [
                "kline",
                "history",
                symbol,
                "--period",
                "day",
                "--start",
                start.isoformat(),
                "--end",
                end.isoformat(),
                "--adjust",
                "forward",
            ]
        )
        if not payload:
            return pd.DataFrame()

        df = pd.DataFrame(payload)
        if df.empty:
            return df

        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        df["symbol"] = symbol.upper()
        for col in ("open", "high", "low", "close", "volume", "turnover"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return (
            df.dropna(subset=["date"])
            .sort_values("date")
            [["date", "symbol", "open", "high", "low", "close", "volume", "turnover"]]
            .reset_index(drop=True)
        )

    def fetch_capital_flow_snapshot(self, symbol: str) -> dict[str, Any]:
        """Fetch today's capital-flow snapshot (大/中/小单 in/out 万元).

        Returns a flat dict with normalized English keys:
            timestamp, large_in, large_out, medium_in, medium_out,
            small_in, small_out, net_large, net_medium, net_small,
            net_main (大+中), net_total.

        Units are 万元 (10k CNY) per the Longbridge CLI output. Empty dict
        when the snapshot is unavailable (e.g. before market open).
        """
        log.info("Longbridge: fetching capital flow snapshot for %s", symbol)
        try:
            payload = _run_longbridge(["capital", symbol])
        except LongbridgeCLIError as exc:
            log.warning("capital snapshot failed for %s: %s", symbol, exc)
            return {}
        if not payload or not isinstance(payload, dict):
            return {}
        cin = payload.get("capital_in") or {}
        cout = payload.get("capital_out") or {}

        def _f(d: dict, k: str) -> float:
            v = _to_float(d.get(k))
            return float(v) if v is not None else 0.0

        large_in, large_out = _f(cin, "large"), _f(cout, "large")
        med_in, med_out = _f(cin, "medium"), _f(cout, "medium")
        small_in, small_out = _f(cin, "small"), _f(cout, "small")
        net_large = large_in - large_out
        net_medium = med_in - med_out
        net_small = small_in - small_out

        return {
            "symbol": symbol.upper(),
            "timestamp": payload.get("timestamp"),
            "large_in": large_in, "large_out": large_out, "net_large": net_large,
            "medium_in": med_in, "medium_out": med_out, "net_medium": net_medium,
            "small_in": small_in, "small_out": small_out, "net_small": net_small,
            "net_main": net_large + net_medium,
            "net_total": net_large + net_medium + net_small,
        }

    def fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        log.info("Longbridge: fetching fundamentals for %s", symbol)
        static_rows = _run_longbridge(["static", symbol, "--lang", "zh-CN"])
        index_rows = _run_longbridge(
            [
                "calc-index",
                symbol,
                "--fields",
                "pe,pb,dps_rate,turnover_rate,mktcap",
            ]
        )

        static = static_rows[0] if isinstance(static_rows, list) and static_rows else {}
        indexes = index_rows[0] if isinstance(index_rows, list) and index_rows else {}
        dps_rate = _to_float(indexes.get("dps_rate"))

        return {
            "name": static.get("name", ""),
            "currency": static.get("currency", ""),
            "exchange": static.get("exchange", ""),
            "lot_size": _to_float(static.get("lot_size")),
            "eps": _to_float(static.get("eps")),
            "eps_ttm": _to_float(static.get("eps_ttm")),
            "bps": _to_float(static.get("bps")),
            "dividend": _to_float(static.get("dividend")),
            "total_shares": _to_float(static.get("total_shares")),
            "circ_shares": _to_float(static.get("circ._shares")),
            "end_pe": _to_float(indexes.get("pe")),
            "current_pb": _to_float(indexes.get("pb")),
            "dividend_yield": (dps_rate / 100.0) if dps_rate is not None else None,
            "turnover_rate": _to_float(indexes.get("turnover_rate")),
            "market_cap": _to_float(indexes.get("mktcap")),
        }

    def fetch_valuation(self, symbol: str) -> dict[str, Any]:
        """Fetch PE/PB time series + Longbridge's ai_summary for valuation context.

        Returns raw payload with keys: history (metrics.{pe,pb}.list time series),
        overview (ai_summary HTML-formatted Chinese conclusion), peers, stocks.
        Empty dict on failure.
        """
        log.info("Longbridge: fetching valuation for %s", symbol)
        try:
            payload = _run_longbridge(["valuation", symbol])
            return payload if isinstance(payload, dict) else {}
        except LongbridgeCLIError as exc:
            log.warning("valuation failed for %s: %s", symbol, exc)
            return {}

    def fetch_institution_rating(self, symbol: str) -> dict[str, Any]:
        """Fetch analyst rating distribution + target price consensus.

        Returns raw payload with keys: analyst (industry stats + target range),
        instratings (recommendation + target + evaluate.{strong_buy,buy,hold,sell}).
        Empty dict on failure.
        """
        log.info("Longbridge: fetching institution rating for %s", symbol)
        try:
            payload = _run_longbridge(["institution-rating", symbol])
            return payload if isinstance(payload, dict) else {}
        except LongbridgeCLIError as exc:
            log.warning("institution-rating failed for %s: %s", symbol, exc)
            return {}

    def fetch_business_segments(
        self, symbol: str, *, history: bool = False, report: str = "af"
    ) -> dict[str, Any]:
        """Fetch business segment revenue breakdown.

        Without history: latest snapshot. With history=True: time series; `report`
        chooses period (af=annual, saf=semi-annual, qf=quarterly).
        Returns raw payload — for snapshot has top-level `business`; for history
        has `historical[]`. Empty dict on failure.
        """
        log.info("Longbridge: fetching business segments for %s (history=%s)", symbol, history)
        args = ["business-segments", symbol]
        if history:
            args.extend(["--history", "--report", report])
        try:
            payload = _run_longbridge(args)
            return payload if isinstance(payload, dict) else {}
        except LongbridgeCLIError as exc:
            log.warning("business-segments failed for %s: %s", symbol, exc)
            return {}


def get_provider(market: str) -> LongbridgeCLIProvider:
    return LongbridgeCLIProvider(market)
