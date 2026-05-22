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
    diff_against_previous,
    load_holdings,
    load_previous_snapshot,
    render_holding_card,
    render_holding_cards_html,
    save_snapshot,
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
    def test_stop_violated_triggers_clear(self):
        # close=16, cost=18, structural stop = max(support=15.5, ma50=17.5)
        # ⇒ stop=17.5, close=16 ≤ 17.5 → 清仓
        history = _df([_history_row()])
        position = Position(
            code="002602", name="华通",
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 18.0, "shares": 100}],
        )
        snap = analyze_holding(history, position)
        self.assertEqual(snap.holdings, 100)
        self.assertEqual(snap.verdict, "REJECT")
        self.assertAlmostEqual(snap.floating_pnl_pct, (16.0 - 18.0) / 18.0, places=4)
        self.assertEqual(snap.action, "清仓")
        self.assertEqual(snap.priority, 1)

    def test_trend_broken_with_heavy_loss_triggers_clear(self):
        # close just above the structural stop but below MA50 AND MA200 with
        # pnl < -10% → still 清仓 via the trend-broken rule.
        history = _df([_history_row(
            close=17.6, ma50=18.0, ma200=18.5, support_level_primary=10.0,
        )])
        position = Position(
            code="X", name="X",
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 20.0, "shares": 100}],
        )
        snap = analyze_holding(history, position)
        # pnl = (17.6 - 20) / 20 = -12%
        self.assertEqual(snap.action, "清仓")
        self.assertEqual(snap.priority, 1)

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
        self.assertEqual(snap.action, "空仓观察")
        self.assertEqual(snap.priority, 4)

    def test_pullback_with_holdings_recommends_add(self):
        # Pullback to MA50 with strong holding, trend intact → 加仓
        history = _df([_history_row(
            close=17.6, high=17.7, ma20=17.5, ma50=17.5, ma200=16.0,
            high_20d=18.5, support_level_primary=17.0,
            volume_spike_ratio=1.5,
        )])
        position = Position(
            code="X", name="X",
            # High enough cost that TP1 is well above current close (no trim).
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 19.0, "shares": 100}],
        )
        snap = analyze_holding(history, position)
        self.assertEqual(snap.trigger_type, "pullback")
        self.assertEqual(snap.verdict, "ACCEPT")
        self.assertEqual(snap.action, "加仓")
        self.assertEqual(snap.priority, 3)

    def test_tp1_hit_triggers_trim_25(self):
        # Cost=10, structural stop = max(support=9.5) = 9.5 (ma50=10 not strictly
        # below cost), R=0.5, TP1=10.5, TP2=11.0.
        # Close=10.7 → past TP1, before TP2 → 减仓 25%.
        history = _df([_history_row(
            close=10.7, high=10.8, ma20=10.3, ma50=10.0, ma200=9.5,
            high_20d=10.5, support_level_primary=9.5,
            volume_spike_ratio=1.0,
        )])
        position = Position(
            code="X", name="X",
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 10.0, "shares": 100}],
        )
        snap = analyze_holding(history, position)
        self.assertEqual(snap.action, "减仓 25%")
        self.assertEqual(snap.priority, 2)
        self.assertIn("TP1", snap.note)

    def test_tp2_hit_triggers_trim_50(self):
        # Cost=10, R=0.5 (stop=9.5), TP1=10.5, TP2=11.0.
        # Close=11.5 → past TP2 → 减仓 50%.
        history = _df([_history_row(
            close=11.5, high=11.6, ma20=10.5, ma50=10.0, ma200=9.5,
            high_20d=10.5, support_level_primary=9.5,
            volume_spike_ratio=1.0,
        )])
        position = Position(
            code="X", name="X",
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 10.0, "shares": 100}],
        )
        snap = analyze_holding(history, position)
        self.assertEqual(snap.action, "减仓 50%")
        self.assertEqual(snap.priority, 2)
        self.assertIn("TP2", snap.note)

    def test_strong_uptrend_holds(self):
        # Trend intact, no trigger, TPs not hit → 持有不动.
        # Cost=20, structural stop = max(support=19.0, ma50=19.5) = 19.5 → R=0.5.
        # TP1=20.5, TP2=21.0.  Close=20.3 → below TP1, above all MAs.
        history = _df([_history_row(
            close=20.3, high=20.4, ma20=20.0, ma50=19.5, ma200=19.0,
            high_20d=20.5, support_level_primary=19.0,
            volume_spike_ratio=0.9,
        )])
        position = Position(
            code="X", name="X",
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 20.0, "shares": 100}],
        )
        snap = analyze_holding(history, position)
        self.assertEqual(snap.action, "持有不动")
        self.assertEqual(snap.priority, 4)


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


class TestSnapshotDiff(unittest.TestCase):
    def _build_snapshot(self, action: str = "持有不动") -> "Position":
        history = _df([_history_row(
            close=20.3, ma20=20.0, ma50=19.5, ma200=19.0,
            high_20d=20.5, support_level_primary=19.0,
            volume_spike_ratio=0.9,
        )])
        position = Position(
            code="X", name="测试",
            trades=[{"date": "2025-09-09", "side": "BUY", "price": 20.0, "shares": 100}],
        )
        return analyze_holding(history, position)

    def test_save_and_reload_roundtrip(self):
        snap = self._build_snapshot()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            save_snapshot([snap], outdir, as_of="2026-05-13")
            # save_snapshot to a *previous* date so today's load sees it
            prev = load_previous_snapshot(outdir, before="2026-05-14")
            self.assertIsNotNone(prev)
            self.assertEqual(prev["as_of"], "2026-05-13")
            self.assertEqual(prev["positions"][0]["code"], "X")
            self.assertEqual(prev["positions"][0]["action"], snap.action)

    def test_load_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_previous_snapshot(Path(tmp)))

    def test_load_skips_today_and_future(self):
        snap = self._build_snapshot()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            save_snapshot([snap], outdir, as_of="2026-05-14")  # "today"
            save_snapshot([snap], outdir, as_of="2026-05-15")  # future
            # When loading "before 2026-05-14" both should be skipped.
            self.assertIsNone(load_previous_snapshot(outdir, before="2026-05-14"))

    def test_diff_unchanged_action(self):
        snap = self._build_snapshot()
        previous = {
            "as_of": "2026-05-13",
            "positions": [{"code": "X", "action": snap.action}],
        }
        self.assertEqual(diff_against_previous(snap, previous), f"延续 ({snap.action})")

    def test_diff_action_changed(self):
        snap = self._build_snapshot()  # action = 持有不动
        previous = {
            "as_of": "2026-05-13",
            "positions": [{"code": "X", "action": "减仓 25%"}],
        }
        out = diff_against_previous(snap, previous)
        self.assertIn("减仓 25%", out)
        self.assertIn("持有不动", out)
        self.assertIn("→", out)

    def test_diff_no_prior_data(self):
        snap = self._build_snapshot()
        self.assertEqual(diff_against_previous(snap, None), "无昨日数据")

    def test_diff_new_position_not_in_prior(self):
        snap = self._build_snapshot()
        previous = {"as_of": "2026-05-13", "positions": []}
        self.assertIn("新持仓", diff_against_previous(snap, previous))


if __name__ == "__main__":
    unittest.main()
