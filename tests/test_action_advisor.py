"""Tests for action_advisor module."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_workbench.signal_engine.action_advisor import (
    advise_actions,
    format_action_text,
)


def _make_signal(**overrides) -> pd.DataFrame:
    defaults = {
        "symbol": "002241.SZ",
        "name": "歌尔股份",
        "signal_state": "watch",
        "trigger_type": None,
        "conviction_score": 30.0,
        "trigger_price": 35.0,
        "stop_price": 32.0,
        "target_zone_low": 33.0,
        "target_zone_high": 36.0,
        "zone_distance": 0.15,
        "event_risk_score": 0.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestAdviseActions(unittest.TestCase):
    def test_triggered_breakout(self):
        df = _make_signal(signal_state="triggered", trigger_type="breakout")
        result = advise_actions(df)
        self.assertEqual(len(result), 1)
        self.assertIn("入场", result.iloc[0]["action"])
        self.assertIn("突破", result.iloc[0]["action"])
        self.assertEqual(result.iloc[0]["action_priority"], 1)
        # Should have entry plan
        self.assertIsNotNone(result.iloc[0]["suggested_entry"])
        self.assertIsNotNone(result.iloc[0]["suggested_stop"])

    def test_triggered_pullback(self):
        df = _make_signal(signal_state="triggered", trigger_type="pullback")
        result = advise_actions(df)
        self.assertIn("回踩", result.iloc[0]["action"])

    def test_triggered_with_event_risk(self):
        df = _make_signal(
            signal_state="triggered", trigger_type="breakout",
            event_risk_score=15.0,
        )
        result = advise_actions(df)
        self.assertIn("事件", result.iloc[0]["action_detail"])
        # Priority should be 2 (important but wait for event)
        self.assertEqual(result.iloc[0]["action_priority"], 2)

    def test_near_zone(self):
        df = _make_signal(signal_state="near_zone", zone_distance=0.05)
        result = advise_actions(df)
        self.assertIn("准备", result.iloc[0]["action"])
        self.assertEqual(result.iloc[0]["action_priority"], 2)
        self.assertIn("5.0%", result.iloc[0]["action_detail"])

    def test_invalidated(self):
        df = _make_signal(signal_state="invalidated")
        result = advise_actions(df)
        self.assertIn("暂停", result.iloc[0]["action"])
        self.assertEqual(result.iloc[0]["action_priority"], 3)

    def test_watch_high_conviction(self):
        df = _make_signal(signal_state="watch", conviction_score=50.0)
        result = advise_actions(df)
        self.assertIn("重点", result.iloc[0]["action"])
        self.assertEqual(result.iloc[0]["action_priority"], 3)

    def test_watch_low_conviction(self):
        df = _make_signal(signal_state="watch", conviction_score=10.0)
        result = advise_actions(df)
        self.assertIn("低优先级", result.iloc[0]["action"])

    def test_empty_signals(self):
        result = advise_actions(pd.DataFrame())
        self.assertTrue(result.empty)
        self.assertIn("action", result.columns)

    def test_sorted_by_priority(self):
        df = pd.concat([
            _make_signal(symbol="A", signal_state="watch", conviction_score=10),
            _make_signal(symbol="B", signal_state="triggered", trigger_type="breakout"),
            _make_signal(symbol="C", signal_state="near_zone", zone_distance=0.05),
        ], ignore_index=True)
        result = advise_actions(df)
        priorities = result["action_priority"].tolist()
        self.assertEqual(priorities, sorted(priorities))
        self.assertEqual(result.iloc[0]["symbol"], "B")  # triggered = priority 1

    def test_entry_plan_has_valid_levels(self):
        df = _make_signal(
            signal_state="triggered", trigger_type="pullback",
            trigger_price=35.0, stop_price=32.0,
        )
        result = advise_actions(df)
        row = result.iloc[0]
        self.assertAlmostEqual(row["suggested_entry"], 35.0)
        self.assertAlmostEqual(row["suggested_stop"], 32.0)
        self.assertGreater(row["suggested_tp1"], 35.0)  # TP1 > entry
        self.assertGreater(row["suggested_tp2"], row["suggested_tp1"])  # TP2 > TP1


class TestFormatActionText(unittest.TestCase):
    def test_format_triggered(self):
        df = _make_signal(signal_state="triggered", trigger_type="breakout")
        actions = advise_actions(df)
        text = format_action_text(actions.iloc[0])
        self.assertIn("入场参考", text)
        self.assertIn("止损位", text)
        self.assertIn("止盈1", text)

    def test_format_watch(self):
        df = _make_signal(signal_state="watch", conviction_score=10)
        actions = advise_actions(df)
        text = format_action_text(actions.iloc[0])
        self.assertIn("低优先级", text)
        self.assertNotIn("入场参考", text)


if __name__ == "__main__":
    unittest.main()
