import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_workbench.signal_engine.radar import add_radar_features, build_current_signals, compute_exit_plan


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
        for col in [
            "signal_state",
            "trigger_type",
            "conviction_score",
            "reasons",
            "risk_unit",
            "take_profit_1",
            "take_profit_2",
            "trailing_stop",
            "exit_plan",
            "exit_plan_notes",
        ]:
            self.assertIn(col, out.columns)
        self.assertAlmostEqual(float(out.iloc[0]["risk_unit"]), 0.8)
        self.assertAlmostEqual(float(out.iloc[0]["take_profit_1"]), 10.8)
        self.assertAlmostEqual(float(out.iloc[0]["take_profit_2"]), 11.6)
        self.assertIn("TP1", out.iloc[0]["exit_plan"])

    def test_compute_exit_plan_uses_r_multiples_and_trailing_stop(self):
        plan = compute_exit_plan(
            entry_price=100.0,
            stop_price=95.0,
            atr14=2.0,
            ma20=98.0,
            highest_close_20d=104.0,
            cfg={"take_profit_1_r": 1.0, "take_profit_2_r": 2.0, "trailing_atr_multiple": 2.0},
        )
        self.assertAlmostEqual(plan["risk_unit"], 5.0)
        self.assertAlmostEqual(plan["take_profit_1"], 105.0)
        self.assertAlmostEqual(plan["take_profit_2"], 110.0)
        self.assertAlmostEqual(plan["trailing_stop"], 100.0)
        self.assertIn("R = entry - stop", plan["exit_plan_notes"])


if __name__ == "__main__":
    unittest.main()
