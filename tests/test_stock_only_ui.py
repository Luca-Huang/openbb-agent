import unittest
from pathlib import Path


class TestStockOnlyUI(unittest.TestCase):
    def test_no_crypto_dashboard_symbols(self):
        src = Path("/Users/huangyuxiang/openbb-agent/streamlit_app.py").read_text(encoding="utf-8")
        self.assertNotIn("render_crypto_dashboard", src)
        self.assertNotIn("load_crypto_supports", src)
        self.assertNotIn("加密面板", src)


if __name__ == "__main__":
    unittest.main()
