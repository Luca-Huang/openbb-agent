"""Tests for signal_engine.holdings_tracker."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_workbench.signal_engine.holdings_tracker import (
    Position,
    analyze_holding,
    load_holdings,
    render_holding_card,
    render_holding_cards_html,
)


def _history_row(**overrides) -> dict:
    base = {
        "date": "2026-04-30",
        "close": 16.0, "high": 16.2, "low": 15.8, "volume": 1000,
        "ma20": 17.0, "ma50": 17.5, "ma200": 18.0,
        "atr14": 0.4, "high_20d": 17.8, "highest_close_20d": 17.8,
        "support_level_primary": 15.5, "support_level_secondary": 17.05,
        "volume_spike_ratio": 0.9, "rsi14": 42.0,
    }
    base.update(overrides)
    return base


def _df(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestLoadHoldings(unittest.TestCase):
    def test_load_minimum_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.json"
            path.write_text(json.dumps({
                "positions": [
                    {"code": "002602", "name": "世纪华通", "trades": [
                        {"date": "2025-09-09", "side": "BUY", "price": 17.98, "shares": 400}
                    ]}
                ]
            }), encoding="utf-8")
            positions = load_holdings(path)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].code, "002602")
        self.assertEqual(len(positions[0].trades), 1)

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_holdings(Path("/nonexistent/holdings.json")), [])


class TestAnalyzeHolding(unittest.TestCase):
    def test_holding_below_all_mas_recommends_de_risk(self):
        history = _df([_history_row()])
        position = Position(
            code="002602", name="华通",
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 18.0, "shares": 100}],
        )
        snap = analyze_holding(history, position)
        self.assertEqual(snap.holdings, 100)
        self.assertEqual(snap.verdict, "REJECT")
        # close=16, ma20=17 → -5.88%
        self.assertAlmostEqual(snap.pct_to_ma20, (16.0 - 17.0) / 17.0, places=4)
        # broker_cost = 18.0 * 100 / 100 = 18.0
        self.assertAlmostEqual(snap.broker_cost, 18.0, places=4)
        # floating = (16.0 - 18.0) * 100 / (18.0 * 100) = -11.11%
        self.assertAlmostEqual(snap.floating_pnl_pct, (16.0 - 18.0) / 18.0, places=4)
        self.assertIn("减仓", snap.action)

    def test_zero_holdings_after_full_exit(self):
        history = _df([_history_row()])
        position = Position(
            code="002624", name="完美",
            trades=[
                {"date": "2025-09-16", "side": "BUY", "price": 18.9, "shares": 500},
                {"date": "2026-02-06", "side": "SELL", "price": 19.57, "shares": 500},
            ],
        )
        snap = analyze_holding(history, position)
        self.assertEqual(snap.holdings, 0)
        # No floating P&L for a flat position
        self.assertTrue(pd.isna(snap.floating_pnl_pct))
        self.assertIn("空仓", snap.action)

    def test_breakout_with_holdings_recommends_add(self):
        history = _df([_history_row(
            close=18.5, high=18.7, ma20=17.0, ma50=17.5, ma200=18.0,
            high_20d=18.0, support_level_primary=17.0,
            volume_spike_ratio=2.0,
        )])
        position = Position(
            code="X", name="X",
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 16.0, "shares": 100}],
        )
        snap = analyze_holding(history, position)
        self.assertEqual(snap.trigger_type, "breakout")
        self.assertEqual(snap.verdict, "ACCEPT")
        self.assertIn("加仓", snap.action)


class TestRender(unittest.TestCase):
    def test_card_does_not_leak_absolutes(self):
        history = _df([_history_row()])
        position = Position(
            code="002602", name="华通",
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 18.0, "shares": 1234}],
        )
        snap = analyze_holding(history, position)
        html = render_holding_card(snap)
        # Critical privacy invariant: the card must NEVER expose absolute share
        # counts or dollar amounts.  Only percentages and prices.
        self.assertNotIn("1234", html)
        self.assertNotIn("shares", html.lower())
        # Price levels are OK to render (they're publicly known anyway)
        # but absolute P&L in 元 must never appear
        self.assertNotIn("元", html)
        # And the rendered floating P&L must be in % form
        self.assertIn("%", html)

    def test_cards_html_includes_section_header(self):
        history = _df([_history_row()])
        position = Position(code="002602", name="华通",
                            trades=[{"date": "2025-09-09", "side": "BUY", "price": 18.0, "shares": 100}])
        snap = analyze_holding(history, position)
        html = render_holding_cards_html([snap])
        self.assertIn("持仓追踪", html)
        self.assertIn("华通", html)

    def test_empty_input_returns_empty(self):
        self.assertEqual(render_holding_cards_html([]), "")


if __name__ == "__main__":
    unittest.main()
