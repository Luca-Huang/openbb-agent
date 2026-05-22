"""Equivalent tests for radar trigger/features logic, migrated to new architecture.

Replaces the old test_radar_triggers.py, test_radar_features.py,
test_radar_output_schema.py, and test_radar_selection.py which imported
the deleted fetch_equities_fmp module.
"""
import unittest

import numpy as np
import pandas as pd

from research_workbench.signal_engine.radar import (
    DEFAULT_RADAR_CONFIG,
    add_radar_features,
    build_current_signals,
    detect_trigger_type,
)


class TestDetectTriggerType(unittest.TestCase):
    def test_pullback_trigger(self):
        row = pd.Series({
            "close": 100,
            "ma50": 98,
            "support_level_primary": 96,
            "volume_spike_ratio": 1.2,
            "high_20d": 102,
        })
        trigger_cfg = DEFAULT_RADAR_CONFIG["triggers"]
        t = detect_trigger_type(row, trigger_cfg)
        self.assertEqual(t, "pullback")

    def test_breakout_trigger(self):
        row = pd.Series({
            "close": 105,
            "ma50": 98,
            "support_level_primary": 96,
            "volume_spike_ratio": 1.8,
            "high_20d": 102,
        })
        trigger_cfg = DEFAULT_RADAR_CONFIG["triggers"]
        t = detect_trigger_type(row, trigger_cfg)
        self.assertEqual(t, "breakout")

    def test_no_trigger(self):
        row = pd.Series({
            "close": 80,
            "ma50": 98,
            "support_level_primary": 96,
            "volume_spike_ratio": 0.8,
            "high_20d": 102,
        })
        trigger_cfg = DEFAULT_RADAR_CONFIG["triggers"]
        t = detect_trigger_type(row, trigger_cfg)
        self.assertIsNone(t)

    def test_trigger_is_valid_type(self):
        row = pd.Series({
            "close": 100,
            "ma50": 98,
            "support_level_primary": 96,
            "volume_spike_ratio": 1.2,
            "high_20d": 102,
        })
        trigger_cfg = DEFAULT_RADAR_CONFIG["triggers"]
        t = detect_trigger_type(row, trigger_cfg)
        self.assertIn(t, {"pullback", "breakout", None})


class TestAddRadarFeatures(unittest.TestCase):
    def test_feature_columns_exist(self):
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            "close": np.linspace(10, 20, 80),
            "high": np.linspace(10.5, 20.5, 80),
            "low": np.linspace(9.5, 19.5, 80),
            "volume": np.full(80, 1_000_000.0),
        })
        out = add_radar_features(df)
        for col in ["high_20d", "atr14", "drawdown_60d", "dollar_volume20", "ma20", "highest_close_20d"]:
            self.assertIn(col, out.columns, f"Missing column: {col}")

    def test_output_length_matches_input(self):
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=30, freq="D"),
            "close": np.linspace(10, 15, 30),
            "high": np.linspace(10.5, 15.5, 30),
            "low": np.linspace(9.5, 14.5, 30),
            "volume": np.full(30, 500_000.0),
        })
        out = add_radar_features(df)
        self.assertEqual(len(out), len(df))


class TestBuildCurrentSignals(unittest.TestCase):
    def test_signal_output_columns(self):
        summary = pd.DataFrame([{
            "symbol": "002241.SZ",
            "name": "歌尔股份",
            "market": "CN",
            "value_score": 30.0,
            "score_hist_valuation": 5.0,
            "score_abs_valuation": 10.0,
            "score_peer_valuation": 5.0,
            "score_peg": 3.0,
            "score_growth_quality": 5.0,
            "score_balance_sheet": 5.0,
            "score_shareholder_return": 0.0,
        }])
        dates = pd.date_range("2025-01-01", periods=250, freq="D")
        history = pd.DataFrame({
            "date": dates,
            "symbol": "002241.SZ",
            "close": np.linspace(30, 40, 250),
            "high": np.linspace(30.5, 40.5, 250),
            "low": np.linspace(29.5, 39.5, 250),
            "volume": np.full(250, 1_000_000.0),
            "ma50": np.linspace(29, 39, 250),
            "ma200": np.linspace(28, 38, 250),
            "support_level_primary": np.linspace(28, 38, 250),
            "support_level_secondary": np.linspace(27, 37, 250),
            "volume_spike_ratio": np.full(250, 1.0),
        })
        watchlist = pd.DataFrame([{
            "symbol": "002241.SZ",
            "name": "歌尔股份",
            "market": "CN",
            "sector": "消费电子",
            "status": "watch",
            "target_zone_low": 32.0,
            "target_zone_high": 35.0,
            "notes": "",
        }])
        events = pd.DataFrame(
            columns=["symbol", "event_date", "event_type", "importance", "impact", "summary"]
        )
        signals = build_current_signals(summary, history, watchlist, events)
        required_cols = [
            "symbol", "signal_state", "trigger_type", "conviction_score",
            "reasons", "invalidation_conditions",
        ]
        for col in required_cols:
            self.assertIn(col, signals.columns, f"Missing signal column: {col}")


if __name__ == "__main__":
    unittest.main()
