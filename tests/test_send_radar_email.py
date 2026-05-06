import os
import unittest

import pandas as pd

os.environ.setdefault("RADAR_SKIP_HOLDINGS", "1")

import scripts.send_radar_email as m


class TestSendRadarEmail(unittest.TestCase):
    def test_build_html_contains_required_columns(self):
        df = pd.DataFrame(
            [
                {
                    "symbol": "NVDA",
                    "market": "US",
                    "trigger_type": "breakout",
                    "trigger_price": 100.0,
                    "stop_price": 95.0,
                    "take_profit_1": 105.0,
                    "take_profit_2": 110.0,
                    "trailing_stop": 101.0,
                    "opportunity_score": 88.0,
                    "reason_1line": "放量突破20日新高",
                    "exit_plan": "TP1 105.00; TP2 110.00; trail rest.",
                }
            ]
        )
        html = m.build_email_html(df, as_of_date="2026-02-24")
        self.assertIn("机会类型", html)
        self.assertIn("触发价", html)
        self.assertIn("止损价", html)
        self.assertIn("止盈1", html)
        self.assertIn("止盈说明", html)
        self.assertIn("NVDA", html)


if __name__ == "__main__":
    unittest.main()
