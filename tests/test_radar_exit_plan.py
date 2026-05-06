"""Tests for compute_exit_plan function."""
import unittest

from research_workbench.signal_engine.radar import compute_exit_plan


class TestComputeExitPlan(unittest.TestCase):
    def test_basic_r_multiple(self):
        """TP1 = entry + 1R, TP2 = entry + 2R."""
        plan = compute_exit_plan(entry_price=100.0, stop_price=92.0)
        R = 100.0 - 92.0  # = 8
        self.assertAlmostEqual(plan["take_profit_1"], 108.0)  # 100 + 8
        self.assertAlmostEqual(plan["take_profit_2"], 116.0)  # 100 + 16

    def test_custom_r_multiples(self):
        cfg = {"tp1_r_multiple": 1.5, "tp2_r_multiple": 3.0}
        plan = compute_exit_plan(entry_price=50.0, stop_price=45.0, cfg=cfg)
        R = 5.0
        self.assertAlmostEqual(plan["take_profit_1"], 57.5)  # 50 + 7.5
        self.assertAlmostEqual(plan["take_profit_2"], 65.0)  # 50 + 15

    def test_trailing_stop_from_ma20(self):
        plan = compute_exit_plan(
            entry_price=100.0, stop_price=92.0,
            ma20=97.0, atr14=3.0, highest_close_20d=101.0,
        )
        # trail candidates: stop=92, ma20=97, highest-1.5*atr=101-4.5=96.5
        # max = 97
        self.assertAlmostEqual(plan["trailing_stop"], 97.0)

    def test_trailing_stop_from_atr_trail(self):
        plan = compute_exit_plan(
            entry_price=100.0, stop_price=92.0,
            ma20=94.0, atr14=2.0, highest_close_20d=102.0,
        )
        # trail candidates: stop=92, ma20=94, highest-1.5*atr=102-3=99
        # max = 99
        self.assertAlmostEqual(plan["trailing_stop"], 99.0)

    def test_fallback_when_stop_above_entry(self):
        """When stop >= entry, R falls back to 8% of entry."""
        plan = compute_exit_plan(entry_price=100.0, stop_price=105.0)
        R = 100.0 * 0.08  # = 8
        self.assertAlmostEqual(plan["take_profit_1"], 108.0)
        self.assertAlmostEqual(plan["take_profit_2"], 116.0)

    def test_trailing_stop_nan_inputs(self):
        """NaN ma20/atr should be handled gracefully."""
        import math
        plan = compute_exit_plan(
            entry_price=100.0, stop_price=92.0,
            ma20=float("nan"), atr14=float("nan"),
        )
        self.assertAlmostEqual(plan["trailing_stop"], 92.0)  # falls back to stop

    def test_returns_dict_with_required_keys(self):
        plan = compute_exit_plan(entry_price=100.0, stop_price=92.0)
        self.assertIn("take_profit_1", plan)
        self.assertIn("take_profit_2", plan)
        self.assertIn("trailing_stop", plan)


if __name__ == "__main__":
    unittest.main()
