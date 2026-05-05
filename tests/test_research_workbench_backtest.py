"""Tests for signal_engine.backtest."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_workbench.signal_engine.backtest import (
    compute_position_pnl,
    decision_review,
    simulate_strategy,
)


HUATONG_TRADES = [
    {"date": "2025-09-09", "side": "BUY",  "price": 17.98, "shares": 400},
    {"date": "2025-09-11", "side": "BUY",  "price": 18.90, "shares": 400},
    {"date": "2025-09-24", "side": "SELL", "price": 21.15, "shares": 400},
    {"date": "2025-10-15", "side": "BUY",  "price": 18.89, "shares": 400},
    {"date": "2025-10-17", "side": "BUY",  "price": 18.25, "shares": 300},
    {"date": "2025-10-20", "side": "BUY",  "price": 17.34, "shares": 200},
    {"date": "2025-10-28", "side": "BUY",  "price": 17.83, "shares": 500},
    {"date": "2025-10-30", "side": "BUY",  "price": 17.31, "shares": 300},
    {"date": "2025-11-12", "side": "SELL", "price": 19.37, "shares": 700},
    {"date": "2025-11-17", "side": "BUY",  "price": 16.70, "shares": 500},
    {"date": "2025-11-17", "side": "BUY",  "price": 16.40, "shares": 600},
    {"date": "2025-12-15", "side": "BUY",  "price": 16.40, "shares": 500},
    {"date": "2026-01-05", "side": "SELL", "price": 18.10, "shares": 500},
    {"date": "2026-01-08", "side": "SELL", "price": 18.60, "shares": 500},
    {"date": "2026-01-30", "side": "SELL", "price": 20.00, "shares": 700},
    {"date": "2026-03-12", "side": "BUY",  "price": 16.89, "shares": 500},
]


class TestComputePositionPnl(unittest.TestCase):
    def test_simple_round_trip_breakeven(self):
        trades = [
            {"side": "BUY",  "price": 10.0, "shares": 100},
            {"side": "SELL", "price": 12.0, "shares": 50},
            {"side": "SELL", "price": 8.0,  "shares": 50},
        ]
        p = compute_position_pnl(trades, last_price=10.0)
        self.assertEqual(p.holdings, 0)
        self.assertAlmostEqual(p.realized_pnl, 0.0)
        self.assertAlmostEqual(p.unrealized_pnl, 0.0)
        self.assertAlmostEqual(p.total_pnl, 0.0)
        self.assertAlmostEqual(p.floating_pnl, 0.0)

    def test_open_position_two_views_match(self):
        trades = [
            {"side": "BUY",  "price": 10.0, "shares": 100},
            {"side": "SELL", "price": 12.0, "shares": 50},
        ]
        p = compute_position_pnl(trades, last_price=11.0)
        # avg cost view
        self.assertAlmostEqual(p.avg_cost, 10.0)
        self.assertAlmostEqual(p.realized_pnl, 100.0)   # (12-10)*50
        self.assertAlmostEqual(p.unrealized_pnl, 50.0)  # (11-10)*50
        # broker view: net invested = 1000 - 600 = 400, on 50 shares = 8.0/share
        self.assertAlmostEqual(p.net_invested, 400.0)
        self.assertAlmostEqual(p.broker_cost, 8.0)
        self.assertAlmostEqual(p.floating_pnl, 150.0)  # 50*11 - 400
        # Two views agree on total
        self.assertAlmostEqual(p.total_pnl, p.floating_pnl)
        self.assertAlmostEqual(p.total_pnl, 150.0)

    def test_huatong_real_trades(self):
        """Reality check against the user's actual broker numbers."""
        p = compute_position_pnl(HUATONG_TRADES, last_price=16.01)
        self.assertEqual(p.holdings, 1800)
        # Broker view (摊薄)
        self.assertAlmostEqual(p.net_invested, 25825.0, places=2)
        self.assertAlmostEqual(p.broker_cost, 25825.0 / 1800, places=4)  # ~14.347
        self.assertAlmostEqual(p.floating_pnl, 1800 * 16.01 - 25825.0, places=2)  # ~2993
        self.assertAlmostEqual(p.floating_pnl_pct, p.floating_pnl / 25825.0, places=4)  # ~11.6%
        # Moving-avg view
        self.assertAlmostEqual(p.avg_cost, 30853.0 / 1800, places=3)  # ~17.14
        # Invariant: total agrees
        self.assertAlmostEqual(p.total_pnl, p.floating_pnl, places=4)
        # Documented total
        self.assertAlmostEqual(round(p.total_pnl), 2993)


# ---- decision review / simulate use radar's production trigger rule.
# To avoid pulling real market data, we build a tiny synthetic dataframe.

def _make_history(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestDecisionReview(unittest.TestCase):
    def test_breakout_buy_is_accepted(self):
        # Bar where close just printed a 20d high on heavy volume → breakout
        history = _make_history([
            {"date": "2026-04-01", "close": 12.0, "high": 12.1, "low": 11.5,
             "volume": 1000, "ma20": 11.0, "ma50": 10.5, "ma200": 9.5,
             "atr14": 0.4, "high_20d": 11.8,
             "support_level_primary": 10.8, "volume_spike_ratio": 2.0},
        ])
        review = decision_review(history, [
            {"date": "2026-04-01", "side": "BUY", "price": 12.0, "shares": 100},
        ])
        self.assertEqual(review.iloc[0]["trigger_type"], "breakout")
        self.assertEqual(review.iloc[0]["verdict"], "ACCEPT")

    def test_far_from_ma_buy_is_rejected(self):
        # Close is 8% below MA50 and below support → neither pullback nor breakout
        history = _make_history([
            {"date": "2026-04-01", "close": 9.2, "high": 9.4, "low": 8.9,
             "volume": 800, "ma20": 9.8, "ma50": 10.0, "ma200": 9.5,
             "atr14": 0.3, "high_20d": 11.0,
             "support_level_primary": 9.5, "volume_spike_ratio": 0.9},
        ])
        review = decision_review(history, [
            {"date": "2026-04-01", "side": "BUY", "price": 9.2, "shares": 100},
        ])
        self.assertEqual(review.iloc[0]["verdict"], "REJECT")
        self.assertEqual(review.iloc[0]["trigger_type"], "")


class TestSimulateStrategy(unittest.TestCase):
    def test_initial_only_exits_when_close_below_ma50(self):
        # Day 1 forced buy at 10. Day 2 close drops below ma50 → exit.
        history = _make_history([
            {"date": "2026-04-01", "close": 10.0, "high": 10.1, "low": 9.9,
             "volume": 1000, "ma20": 9.5, "ma50": 9.0, "ma200": 8.5,
             "atr14": 0.3, "high_20d": 10.0,
             "support_level_primary": 9.5, "volume_spike_ratio": 1.0},
            {"date": "2026-04-02", "close": 8.5, "high": 9.0, "low": 8.4,
             "volume": 1200, "ma20": 9.4, "ma50": 9.0, "ma200": 8.5,
             "atr14": 0.4, "high_20d": 10.0,
             "support_level_primary": 9.5, "volume_spike_ratio": 1.5},
        ])
        log, summary = simulate_strategy(
            history,
            {"date": "2026-04-01", "price": 10.0, "shares": 100},
            allow_reentry=False,
        )
        actions = log["action"].tolist()
        self.assertIn("FORCED_BUY", actions)
        self.assertIn("SELL_EXIT", actions)
        self.assertLess(summary["pnl"], 0)


if __name__ == "__main__":
    unittest.main()
