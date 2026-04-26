import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_workbench.signal_engine.radar import add_radar_features, build_current_signals


class TestResearchWorkbenchSignalEngine(unittest.TestCase):
    def test_build_current_signals_returns_expected_columns(self):
        summary = pd.DataFrame(
            [
                {
                    "symbol": "002624.SZ",
                    "name": "完美世界",
                    "market": "CN",
                    "value_score": 62,
                    "entry_recommendation": "可评估入场",
                    "score_hist_valuation": 7,
                    "score_abs_valuation": 4,
                    "score_peer_valuation": 5,
                    "score_peg": 0,
                    "score_growth_quality": 10,
                    "score_balance_sheet": 7,
                    "score_shareholder_return": 7,
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "date": "2026-04-01",
                    "symbol": "002624.SZ",
                    "close": 10.0,
                    "high": 10.2,
                    "low": 9.8,
                    "volume": 1000000,
                    "ma50": 10.0,
                    "ma200": 9.4,
                    "support_level_primary": 9.7,
                    "support_level_secondary": 9.2,
                    "volume_spike_ratio": 1.3,
                }
            ]
        )
        history["date"] = pd.to_datetime(history["date"])
        history = add_radar_features(history)
        watchlist = pd.DataFrame(
            [
                {
                    "symbol": "002624.SZ",
                    "name": "完美世界",
                    "market": "CN",
                    "target_zone_low": 9.8,
                    "target_zone_high": 10.2,
                }
            ]
        )
        out = build_current_signals(summary, history, watchlist, pd.DataFrame())
        self.assertFalse(out.empty)
        for col in ["signal_state", "trigger_type", "conviction_score", "reasons"]:
            self.assertIn(col, out.columns)


if __name__ == "__main__":
    unittest.main()

