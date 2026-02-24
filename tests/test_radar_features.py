import unittest
import sys
import types

import numpy as np
import pandas as pd

# Stub yfinance dependency so unit tests focus on local feature logic.
if "yfinance" not in sys.modules:
    yf_stub = types.ModuleType("yfinance")
    yf_stub.Ticker = object
    sys.modules["yfinance"] = yf_stub

import fetch_equities_fmp as m


class TestRadarFeatures(unittest.TestCase):
    def test_feature_columns_exist(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=80, freq="D"),
                "close": np.linspace(10, 20, 80),
                "high": np.linspace(10.5, 20.5, 80),
                "low": np.linspace(9.5, 19.5, 80),
                "volume": np.full(80, 1_000_000.0),
            }
        )
        out = m.add_radar_features(df)
        for col in ["high_20d", "atr14", "drawdown_60d", "dollar_volume20"]:
            self.assertIn(col, out.columns)


if __name__ == "__main__":
    unittest.main()
