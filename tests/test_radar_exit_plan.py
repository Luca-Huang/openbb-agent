"""Tests for compute_exit_plan function."""
import math
import unittest

from research_workbench.signal_engine.radar import compute_exit_plan

DEFAULT_CFG = {
    "take_profit_1_r": 1.0,
    "take_profit_2_r": 2.0,
    "take_profit_1_allocation": "30%-50%",
    "take_profit_2_allocation": "20%-30%",
    "trailing_atr_multiple": 2.0,
}


def _plan(entry, stop, atr14=None, ma20=None, highest_close_20d=None, cfg=None):
    return compute_exit_plan(
        entry, stop, atr14, ma20, highest_close_20d, cfg or DEFAULT_CFG
    )


class TestComputeExitPlan(unittest.TestCase):
    def test_basic_r_multiple(self):
        """TP1 = entry + 1R, TP2 = entry + 2R."""
        plan = _plan(100.0, 92.0)
        self.assertAlmostEqual(plan["risk_unit"], 8.0)
        self.assertAlmostEqual(plan["take_profit_1"], 108.0)  # 100 + 8
        self.assertAlmostEqual(plan["take_profit_2"], 116.0)  # 100 + 16

    def test_custom_r_multiples(self):
        cfg = dict(DEFAULT_CFG, take_profit_1_r=1.5, take_profit_2_r=3.0)
        plan = _plan(50.0, 45.0, cfg=cfg)
        self.assertAlmostEqual(plan["take_profit_1"], 57.5)  # 50 + 1.5 * 5
        self.assertAlmostEqual(plan["take_profit_2"], 65.0)  # 50 + 3.0 * 5

    def test_trailing_stop_from_ma20(self):
        # candidates: ma20=97 (<=entry), highest-2*atr=101-6=95 (<=entry); max=97
        plan = _plan(100.0, 92.0, atr14=3.0, ma20=97.0, highest_close_20d=101.0)
        self.assertAlmostEqual(plan["trailing_stop"], 97.0)

    def test_trailing_stop_from_atr_trail(self):
        # candidates: ma20=94, highest-2*atr=102-4=98; max=98
        plan = _plan(100.0, 92.0, atr14=2.0, ma20=94.0, highest_close_20d=102.0)
        self.assertAlmostEqual(plan["trailing_stop"], 98.0)

    def test_no_plan_when_stop_above_entry(self):
        """When stop >= entry there is no valid R, so no take-profit plan."""
        plan = _plan(100.0, 105.0)
        self.assertTrue(math.isnan(plan["take_profit_1"]))
        self.assertTrue(math.isnan(plan["take_profit_2"]))
        self.assertIn("No take-profit plan", plan["exit_plan"])

    def test_trailing_stop_no_valid_candidate(self):
        """With no usable ma20/atr inputs, trailing stop is NaN."""
        plan = _plan(100.0, 92.0, atr14=float("nan"), ma20=float("nan"))
        self.assertTrue(math.isnan(plan["trailing_stop"]))

    def test_returns_dict_with_required_keys(self):
        plan = _plan(100.0, 92.0)
        for key in (
            "risk_unit",
            "take_profit_1",
            "take_profit_2",
            "trailing_stop",
            "exit_plan",
            "exit_plan_notes",
        ):
            self.assertIn(key, plan)


if __name__ == "__main__":
    unittest.main()
