import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_workbench.validation.replay import replay_trigger_history, build_validation_summary


class TestResearchWorkbenchValidation(unittest.TestCase):
    def test_replay_and_summary(self):
        periods = 40
        history = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=periods, freq="D"),
                "symbol": ["002624.SZ"] * periods,
                "close": np.linspace(10, 14, periods),
                "high": np.linspace(10.1, 14.2, periods),
                "low": np.linspace(9.9, 13.8, periods),
                "volume": np.full(periods, 1_200_000),
                "ma50": np.linspace(10, 13.8, periods),
                "support_level_primary": np.linspace(9.8, 13.5, periods),
                "support_level_secondary": np.linspace(9.5, 13.0, periods),
                "volume_spike_ratio": np.full(periods, 1.6),
            }
        )
        replay = replay_trigger_history(history, symbols=["002624.SZ"])
        summary = build_validation_summary(replay)
        self.assertIsInstance(replay, pd.DataFrame)
        self.assertIsInstance(summary, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
