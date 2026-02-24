import sys
import types
import unittest

import pandas as pd

if "yfinance" not in sys.modules:
    yf_stub = types.ModuleType("yfinance")
    yf_stub.Ticker = object
    sys.modules["yfinance"] = yf_stub

import fetch_equities_fmp as m


class TestRadarTriggers(unittest.TestCase):
    def test_trigger_type_is_pullback_or_breakout_or_none(self):
        row = pd.Series(
            {
                "close": 100,
                "ma50": 98,
                "support_level": 96,
                "volume_spike_ratio": 1.2,
                "high_20d": 102,
            }
        )
        t = m.detect_trigger_type(
            row,
            {
                "pullback": {
                    "ma50_distance_max": 0.03,
                    "volume_spike_min": 1.1,
                    "require_support_above_primary": True,
                },
                "breakout": {"volume_spike_min": 1.5},
            },
        )
        self.assertIn(t, {"pullback", "breakout", None})


if __name__ == "__main__":
    unittest.main()
