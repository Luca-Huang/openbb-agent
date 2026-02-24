import sys
import types
import unittest

import pandas as pd

if "yfinance" not in sys.modules:
    yf_stub = types.ModuleType("yfinance")
    yf_stub.Ticker = object
    sys.modules["yfinance"] = yf_stub

import fetch_equities_fmp as m


class TestRadarSelection(unittest.TestCase):
    def test_candidate_count_respects_bounds(self):
        df = pd.DataFrame(
            [{"symbol": f"S{i}", "opportunity_score": 100 - i, "trigger_type": "pullback"} for i in range(30)]
        )
        out = m.select_radar_candidates(df, min_n=8, max_n=15)
        self.assertGreaterEqual(len(out), 8)
        self.assertLessEqual(len(out), 15)


if __name__ == "__main__":
    unittest.main()
