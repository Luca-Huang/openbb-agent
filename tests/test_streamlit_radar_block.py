import unittest
from pathlib import Path


class TestStreamlitRadarBlock(unittest.TestCase):
    def test_has_radar_section_text(self):
        src = Path("/Users/huangyuxiang/openbb-agent/streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("今日机会雷达", src)
        self.assertIn("机会类型", src)
        self.assertIn("触发价", src)
        self.assertIn("止损价", src)


if __name__ == "__main__":
    unittest.main()
