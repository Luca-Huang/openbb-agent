import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_workbench.signal_engine.changes import detect_daily_changes


class TestDetectDailyChanges(unittest.TestCase):
    def _make_signals(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_new_trigger_without_previous(self):
        current = self._make_signals([
            {"symbol": "000001.SZ", "name": "平安银行", "signal_state": "triggered", "trigger_type": "pullback", "conviction_score": 60.0},
        ])
        changes = detect_daily_changes(current, None)
        self.assertFalse(changes.empty)
        self.assertEqual(changes.iloc[0]["change_type"], "新触发")

    def test_state_transition_watch_to_near_zone(self):
        prev = self._make_signals([
            {"symbol": "000001.SZ", "name": "平安银行", "signal_state": "watch", "trigger_type": None, "conviction_score": 40.0},
        ])
        curr = self._make_signals([
            {"symbol": "000001.SZ", "name": "平安银行", "signal_state": "near_zone", "trigger_type": None, "conviction_score": 45.0},
        ])
        changes = detect_daily_changes(curr, prev)
        self.assertFalse(changes.empty)
        self.assertEqual(changes.iloc[0]["change_type"], "进入观察区间")

    def test_conviction_significant_change(self):
        prev = self._make_signals([
            {"symbol": "000001.SZ", "name": "平安银行", "signal_state": "triggered", "trigger_type": "pullback", "conviction_score": 50.0},
        ])
        curr = self._make_signals([
            {"symbol": "000001.SZ", "name": "平安银行", "signal_state": "triggered", "trigger_type": "pullback", "conviction_score": 60.0},
        ])
        changes = detect_daily_changes(curr, prev)
        self.assertFalse(changes.empty)
        self.assertIn("conviction", changes.iloc[0]["change_type"])

    def test_no_change_returns_empty(self):
        signals = self._make_signals([
            {"symbol": "000001.SZ", "name": "平安银行", "signal_state": "watch", "trigger_type": None, "conviction_score": 40.0},
        ])
        changes = detect_daily_changes(signals, signals)
        self.assertTrue(changes.empty)

    def test_empty_current_returns_empty(self):
        changes = detect_daily_changes(pd.DataFrame(), None)
        self.assertTrue(changes.empty)


if __name__ == "__main__":
    unittest.main()
