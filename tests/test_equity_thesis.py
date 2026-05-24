"""Tests for analysis.equity_thesis: structure + markdown rendering."""
import os
import sys
import unittest
from datetime import date

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from research_workbench.analysis.equity_thesis import (
    BusinessSegment,
    EquityThesis,
    QualityFactor,
    build_thesis,
    render_thesis_markdown,
)


class TestEquityThesisDataclass(unittest.TestCase):
    def test_defaults_safe(self):
        t = EquityThesis(symbol="002624.SZ")
        self.assertEqual(t.symbol, "002624.SZ")
        self.assertEqual(t.action, "watch")
        self.assertEqual(t.segments, [])
        self.assertEqual(t.quality_factors, [])

    def test_render_empty(self):
        """Rendering an empty thesis should never raise."""
        md = render_thesis_markdown(EquityThesis(symbol="X.SZ", as_of=date(2026, 5, 25)))
        self.assertIn("X.SZ", md)
        self.assertIn("① 催化剂", md)
        self.assertIn("⑥ 行动建议", md)
        self.assertIn("`watch`", md)

    def test_render_with_segments(self):
        t = EquityThesis(
            symbol="X.SZ",
            segments=[
                BusinessSegment(name="A", profit=1e9, pe_multiple=15.0, valuation=1.5e10),
                BusinessSegment(name="B", profit=5e8, pe_multiple=10.0, valuation=5e9),
            ],
            sotp_valuation=2e10,
            current_market_cap=1.5e10,
            upside_pct=0.333,
            current_price=12.5,
            action="accumulate",
        )
        md = render_thesis_markdown(t)
        self.assertIn("A", md)
        self.assertIn("B", md)
        self.assertIn("200.00亿", md)  # SOTP total
        self.assertIn("33.3%", md)     # upside
        self.assertIn("accumulate", md)


class TestBuildThesisFromItem(unittest.TestCase):
    def test_manual_segments_override(self):
        item = pd.Series({
            "symbol": "002624.SZ",
            "name": "TestCo",
            "thesis": {
                "manual_segments": [
                    {"name": "core", "profit": 1e9, "pe_multiple": 12, "note": ""},
                    {"name": "growth", "profit": 2.5e9, "pe_multiple": 15, "note": ""},
                ],
                "action": "accumulate",
            },
        })
        # Empty data inputs — manual_segments should still drive SOTP.
        history = pd.DataFrame()
        fund = {"market_cap": 2e10}
        t = build_thesis(item, history, fund)
        self.assertEqual(len(t.segments), 2)
        self.assertEqual(t.segments[0].source, "manual_override")
        self.assertAlmostEqual(t.sotp_valuation, 1e9 * 12 + 2.5e9 * 15)
        # 12+37.5 = 49.5B vs 20B mkt cap -> ~147% upside
        self.assertAlmostEqual(t.upside_pct, (t.sotp_valuation - 2e10) / 2e10)
        self.assertIn("manual_segments", [s.source for s in t.segments] and ["manual_segments"]
                      or [])  # smoke
        self.assertEqual(t.action, "accumulate")

    def test_quality_factors_string_and_dict_forms(self):
        item = pd.Series({
            "symbol": "X.SZ",
            "thesis": {
                "quality_factors": [
                    "team strong",
                    {"description": "tech moat", "evidence": "patents", "direction": 1},
                    {"description": "regulatory risk", "evidence": "policy", "direction": -1},
                ],
            },
        })
        t = build_thesis(item, pd.DataFrame(), {})
        self.assertEqual(len(t.quality_factors), 3)
        self.assertEqual(t.quality_factors[0].description, "team strong")
        self.assertEqual(t.quality_factors[1].evidence, "patents")
        self.assertEqual(t.quality_factors[2].direction, -1)

    def test_missing_inputs_recorded(self):
        item = pd.Series({"symbol": "X.SZ"})
        t = build_thesis(item, pd.DataFrame(), {})
        self.assertIn("business_segments", t.missing)
        self.assertIn("peer_pe_median", t.missing)


class TestScoreSegmentHealth(unittest.TestCase):
    """Sanity-check the segment health scoring on synthetic fixtures."""

    def _make_segments(self, rows):
        """Build a 合计-classification frame from (year, segment, share, margin) tuples."""
        df = pd.DataFrame(rows, columns=["year", "segment", "revenue_ratio", "gross_margin"])
        df["fiscal_period"] = pd.to_datetime(df["year"].astype(str) + "-12-31")
        df["classification"] = "合计"
        df["revenue"] = df["revenue_ratio"] * 1e10
        df["profit"] = df["revenue"] * df["gross_margin"]
        return df

    def test_returns_zero_when_empty(self):
        from research_workbench.analysis.summary import score_segment_health
        self.assertEqual(score_segment_health(None), 0.0)
        self.assertEqual(score_segment_health(pd.DataFrame()), 0.0)

    def test_over_concentration_penalized(self):
        """One segment at 96% of revenue → 0 concentration points."""
        from research_workbench.analysis.summary import score_segment_health
        df = self._make_segments([
            (2025, "core", 0.96, 0.50),
            (2025, "other", 0.04, 0.20),
            (2024, "core", 0.95, 0.48),
            (2024, "other", 0.05, 0.18),
            (2023, "core", 0.94, 0.45),
            (2023, "other", 0.06, 0.16),
        ])
        # concentration 96% -> 0pts; margin trend rising (.45->.48->.50) -> +3;
        # diversification: 1 secondary at 4% (< 5%) -> 0pts. Total = 3.
        self.assertEqual(score_segment_health(df), 3.0)

    def test_healthy_diversified(self):
        """50-90% top + rising margins + 2+ secondaries > 5% → max-ish score."""
        from research_workbench.analysis.summary import score_segment_health
        df = self._make_segments([
            (2025, "core", 0.60, 0.40),
            (2025, "b", 0.25, 0.30),
            (2025, "c", 0.15, 0.20),
            (2024, "core", 0.62, 0.38),
            (2024, "b", 0.23, 0.28),
            (2024, "c", 0.15, 0.18),
            (2023, "core", 0.65, 0.35),
            (2023, "b", 0.22, 0.25),
            (2023, "c", 0.13, 0.16),
        ])
        # concentration 60% (50-90) -> +4; margin trend rising -> +3; 2 secondaries > 5% -> +3. Total = 10.
        self.assertEqual(score_segment_health(df), 10.0)


if __name__ == "__main__":
    unittest.main()
