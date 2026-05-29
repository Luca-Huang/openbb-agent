"""Unit tests for watchlist_dashboard pure functions.

Covers the pieces that are easy to break when iterating PS multiples, currencies,
or verdict rules. Does NOT touch Longbridge — fetchers are tested separately.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from watchlist_dashboard import (  # noqa: E402
    _annual_segments,
    _fx,
    _match_seg,
    _ratings_metrics,
    _valuation_metrics,
    build_sotp,
    compute_verdict,
    load_config,
)


class TestVerdict(unittest.TestCase):
    """Pin the 5 verdict branches so changes to compute_verdict are intentional."""

    def test_breakout_within_cap_is_buy(self):
        # close = 100, h20 = 100, atr = 5, cap = 105 → not extended
        v, stop = compute_verdict(close=100, ma20=95, ma200=90, h20=100, atr=5, trigger="breakout")
        self.assertEqual(v, "建仓")
        self.assertEqual(stop, 100)

    def test_breakout_past_cap_is_wait(self):
        # close = 110 > cap (105) → too extended
        v, _ = compute_verdict(close=110, ma20=95, ma200=90, h20=100, atr=5, trigger="breakout")
        self.assertEqual(v, "等回踩")

    def test_pullback_is_cautious(self):
        v, stop = compute_verdict(close=95, ma20=94, ma200=90, h20=100, atr=5, trigger="pullback")
        self.assertEqual(v, "回踩-谨慎")
        self.assertEqual(stop, min(94, 100 * 0.97))

    def test_no_trigger_near_high_above_ma20_is_watch(self):
        # close 96 vs h20 100 = -4%, above ma20 → 观察
        v, _ = compute_verdict(close=96, ma20=95, ma200=90, h20=100, atr=5, trigger=None)
        self.assertEqual(v, "观察")

    def test_no_trigger_below_ma20_is_avoid(self):
        v, _ = compute_verdict(close=90, ma20=95, ma200=100, h20=100, atr=5, trigger=None)
        self.assertEqual(v, "不碰")

    def test_no_trigger_far_below_high_is_avoid(self):
        # close 94 vs h20 100 = -6% → too far, even if above ma20
        v, _ = compute_verdict(close=94, ma20=93, ma200=90, h20=100, atr=5, trigger=None)
        self.assertEqual(v, "不碰")


class TestSegmentMatching(unittest.TestCase):
    """Ensure keyword matching for SOTP segments stays robust to LB renames."""

    def test_case_insensitive_match(self):
        segs = [{"match": ["云"], "label": "云智能", "ps": 4.5, "reason": "x"}]
        m = _match_seg("云智能集团", segs)
        self.assertIsNotNone(m)
        self.assertEqual(m["label"], "云智能")

    def test_english_keyword_matches_chinese_name(self):
        segs = [{"match": ["AIDC", "国际"], "label": "AIDC国际", "ps": 1.0, "reason": "x"}]
        self.assertIsNotNone(_match_seg("阿里国际数字商业集团（AIDC）", segs))

    def test_no_match_returns_none(self):
        segs = [{"match": ["游戏"], "label": "游戏", "ps": 5.0, "reason": "x"}]
        self.assertIsNone(_match_seg("分部间抵消", segs))


class TestFx(unittest.TestCase):
    def test_same_currency_is_identity(self):
        self.assertEqual(_fx({}, "HKD", "HKD"), 1.0)

    def test_known_pair_returns_rate(self):
        self.assertEqual(_fx({"CNY/HKD": 1.087}, "CNY", "HKD"), 1.087)

    def test_unknown_pair_falls_back_to_identity(self):
        # Logs a warning but does not crash — bias toward visible-but-wrong over missing data.
        self.assertEqual(_fx({}, "CNY", "USD"), 1.0)


class TestValuationMetrics(unittest.TestCase):
    def test_pe_percentile_counts_cheaper_history(self):
        payload = {
            "history": {"metrics": {"pe": {"list": [
                {"value": "20"}, {"value": "18"}, {"value": "10"},
                {"value": "12"}, {"value": "8"},  # current = 8, 4/5 = 80% history was more expensive
            ]}}},
            "overview": {"ai_summary": "<strong>cheap</strong>"},
        }
        out = _valuation_metrics(payload)
        self.assertAlmostEqual(out["pe_cheap_pct"], 80.0)
        self.assertEqual(out["val_summary"], "cheap")

    def test_empty_payload_safe(self):
        out = _valuation_metrics({})
        self.assertIsNone(out["pe_cheap_pct"])
        self.assertEqual(out["val_summary"], "")


class TestRatingsMetrics(unittest.TestCase):
    def test_parses_target_and_evaluate(self):
        payload = {"instratings": {
            "target": "709.05", "ccy_symbol": "HK$", "recommend": "strong_buy",
            "evaluate": {"strong_buy": 35, "buy": 8, "hold": 2, "sell": 1},
        }}
        out = _ratings_metrics(payload)
        self.assertEqual(out["target"], 709.05)
        self.assertEqual(out["target_ccy"], "HK$")
        self.assertEqual(out["rate_sb"], 35)


class TestAnnualSegments(unittest.TestCase):
    def test_picks_last_annual_report(self):
        payload = {"historical": [
            {"business": [{"name": "old", "value": "1"}]},
            {"business": [{"name": "增值服务", "value": "153983000000"},
                          {"name": "广告", "value": "40439000000"},
                          {"name": "blank", "value": ""}]},  # filtered
        ]}
        rows = _annual_segments(payload)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "增值服务")
        self.assertAlmostEqual(rows[0][1], 153983000000.0)


class TestBuildSotp(unittest.TestCase):
    """Mock the provider so we test SOTP math + currency conversion in isolation."""

    class _FakeProvider:
        def __init__(self, segments):
            self._segments = segments

        def fetch_business_segments(self, symbol, history=False, report="af"):
            return {"historical": [{"business": self._segments}]}

    def test_sotp_math_with_currency_conversion(self):
        # Segments report 100亿 CNY × PS 5 = 500亿 CNY of value.
        # Market cap 400亿 HKD. fx CNY→HKD = 1.1 → target 550亿 HKD vs 400 = +37.5%
        provider = self._FakeProvider([{"name": "云", "value": "10000000000"}])
        cfg = {
            "segments": {"TEST.HK": {
                "segment_currency": "CNY",
                "segments": [{"match": ["云"], "label": "云", "ps": 5.0, "reason": "x"}],
            }},
            "fx_rates": {"CNY/HKD": 1.1},
        }
        out = build_sotp("TEST.HK", cfg, provider, mktcap=400e8, mktcap_ccy="HKD", fx_rates=cfg["fx_rates"])
        self.assertIsNotNone(out)
        self.assertEqual(out["seg_ccy"], "CNY")
        self.assertEqual(out["mc_ccy"], "HKD")
        self.assertAlmostEqual(out["fx"], 1.1)
        self.assertAlmostEqual(out["target_yi"], 550.0)  # 100亿 × 5 × 1.1
        self.assertAlmostEqual(out["mktcap_yi"], 400.0)
        self.assertAlmostEqual(out["upside"], 37.5)

    def test_extra_off_revenue_lines_added(self):
        # Segment 100亿 × 1 = 100亿. extra 50亿. Total 150亿 (same currency).
        provider = self._FakeProvider([{"name": "云", "value": "10000000000"}])
        cfg = {
            "segments": {"TEST.HK": {
                "segment_currency": "HKD",
                "segments": [{"match": ["云"], "label": "云", "ps": 1.0, "reason": "x"}],
                "extra": [{"label": "净现金", "value_yi": 50, "reason": "y"}],
            }},
            "fx_rates": {},
        }
        out = build_sotp("TEST.HK", cfg, provider, mktcap=100e8, mktcap_ccy="HKD", fx_rates={})
        self.assertEqual(len(out["rows"]), 2)
        self.assertAlmostEqual(out["target_yi"], 150.0)

    def test_no_config_returns_none(self):
        provider = self._FakeProvider([])
        out = build_sotp("UNKNOWN.HK", {"segments": {}}, provider, 1e10, "HKD", {})
        self.assertIsNone(out)

    def test_no_matching_segments_returns_none(self):
        # Provider returns segments but none match config keywords → None (not empty rows).
        provider = self._FakeProvider([{"name": "分部间抵消", "value": "1000000000"}])
        cfg = {
            "segments": {"TEST.HK": {
                "segment_currency": "HKD",
                "segments": [{"match": ["游戏"], "label": "游戏", "ps": 5.0, "reason": "x"}],
            }},
            "fx_rates": {},
        }
        self.assertIsNone(build_sotp("TEST.HK", cfg, provider, 1e10, "HKD", {}))


class TestLoadConfig(unittest.TestCase):
    """Smoke test that the shipped JSON config parses and has expected shape."""

    def test_shipped_config_loads(self):
        cfg = load_config()
        self.assertIn("baskets", cfg)
        self.assertIn("HK", cfg["baskets"])
        self.assertIn("thesis", cfg)
        # Every basket entry should have a thesis (warns about config drift)
        for market_basket in cfg["baskets"].values():
            for sym, _ in market_basket:
                self.assertIn(sym, cfg["thesis"], f"missing thesis for {sym}")


if __name__ == "__main__":
    unittest.main()
