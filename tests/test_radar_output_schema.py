import sys
import types
import unittest

import pandas as pd

if "yfinance" not in sys.modules:
    yf_stub = types.ModuleType("yfinance")
    yf_stub.Ticker = object
    sys.modules["yfinance"] = yf_stub

import fetch_equities_fmp as m


class TestRadarOutputSchema(unittest.TestCase):
    def test_columns(self):
        summary = pd.DataFrame(
            [
                {
                    "symbol": "NVDA",
                    "market": "US",
                    "value_score": 82.0,
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "date": "2026-02-24",
                    "symbol": "NVDA",
                    "market": "US",
                    "close": 100.0,
                    "ma50": 98.0,
                    "ma200": 92.0,
                    "support_level": 95.0,
                    "support_level_secondary": 90.0,
                    "high_20d": 99.0,
                    "volume_spike_ratio": 1.8,
                    "dollar_volume20": 12000000.0,
                    "drawdown_60d": 0.08,
                    "atr14": 4.0,
                }
            ]
        )
        radar_df = m.build_opportunity_radar(summary, history, m.RADAR_CONFIG)
        cols = [
            "symbol",
            "market",
            "trigger_type",
            "trigger_price",
            "stop_price",
            "opportunity_score",
            "reason_1line",
        ]
        self.assertTrue(set(cols).issubset(radar_df.columns))


if __name__ == "__main__":
    unittest.main()
