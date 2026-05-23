import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_workbench.data_sources.indicators import add_technical_indicators
from research_workbench.data_sources.longbridge import (
    LongbridgeCLIError,
    get_provider,
)


def _proc(payload, returncode=0, stderr=""):
    stdout = json.dumps(payload) if returncode == 0 else ""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class TestLongbridgeProvider(unittest.TestCase):
    @patch("research_workbench.data_sources.longbridge.subprocess.run")
    def test_fetch_history_uses_longbridge_kline_history(self, run):
        run.return_value = _proc(
            [
                {
                    "time": "2026-05-21 16:00:00",
                    "open": "14.41",
                    "high": "14.62",
                    "low": "14.13",
                    "close": "14.52",
                    "volume": "524211",
                    "turnover": "754389614.58",
                }
            ]
        )

        df = get_provider("CN").fetch_history(
            "002624.SZ",
            date(2026, 5, 1),
            date(2026, 5, 23),
        )

        self.assertEqual(df.iloc[0]["symbol"], "002624.SZ")
        self.assertAlmostEqual(float(df.iloc[0]["close"]), 14.52)
        self.assertIn("longbridge", run.call_args.args[0][0])
        self.assertIn("kline", run.call_args.args[0])
        self.assertIn("history", run.call_args.args[0])
        self.assertIn("--adjust", run.call_args.args[0])
        self.assertIn("forward", run.call_args.args[0])

    @patch("research_workbench.data_sources.longbridge.subprocess.run")
    def test_fetch_fundamentals_maps_static_and_calc_index(self, run):
        run.side_effect = [
            _proc(
                [
                    {
                        "symbol": "META.US",
                        "name": "Meta",
                        "currency": "USD",
                        "exchange": "NASD",
                        "eps_ttm": "27.8",
                        "bps": "96.0",
                    }
                ]
            ),
            _proc(
                [
                    {
                        "symbol": "META.US",
                        "pe": "21.95",
                        "pb": "6.36",
                        "dps_rate": "0.34",
                        "mktcap": "1549098205499.04",
                    }
                ]
            ),
        ]

        data = get_provider("US").fetch_fundamentals("META.US")

        self.assertEqual(data["name"], "Meta")
        self.assertEqual(data["currency"], "USD")
        self.assertAlmostEqual(data["end_pe"], 21.95)
        self.assertAlmostEqual(data["current_pb"], 6.36)
        self.assertAlmostEqual(data["dividend_yield"], 0.0034)
        self.assertAlmostEqual(data["market_cap"], 1549098205499.04)

    @patch("research_workbench.data_sources.longbridge.subprocess.run")
    def test_cli_error_is_reported(self, run):
        run.return_value = _proc(None, returncode=1, stderr="Error: openapi error: code=403308")

        with self.assertRaises(LongbridgeCLIError):
            get_provider("HK").fetch_history("9992.HK", date(2026, 5, 1), date(2026, 5, 23))

    def test_add_technical_indicators_keeps_expected_schema(self):
        df = get_provider("CN")
        self.assertEqual(df.market, "CN")
        history = add_technical_indicators(
            pd.DataFrame(
                {
                    "date": pd.date_range("2026-01-01", periods=3),
                    "symbol": ["002624.SZ"] * 3,
                    "open": [10, 11, 12],
                    "high": [11, 12, 13],
                    "low": [9, 10, 11],
                    "close": [10, 11, 12],
                    "volume": [100, 200, 300],
                }
            )
        )
        self.assertIn("ma50", history.columns)
        self.assertIn("support_level_primary", history.columns)
        self.assertIn("volume_spike_ratio", history.columns)


if __name__ == "__main__":
    unittest.main()
