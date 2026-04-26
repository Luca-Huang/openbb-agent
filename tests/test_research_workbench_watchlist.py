import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_workbench.ingestion.watchlist import load_watchlist


class TestResearchWorkbenchWatchlist(unittest.TestCase):
    def test_load_watchlist_filters_cn_market(self):
        root = Path(__file__).resolve().parents[1]
        df = load_watchlist(
            root / "research_inputs" / "watchlist_cn.json",
            root / "equity_config.json",
            market="CN",
        )
        self.assertFalse(df.empty)
        self.assertTrue((df["market"] == "CN").all())


if __name__ == "__main__":
    unittest.main()

