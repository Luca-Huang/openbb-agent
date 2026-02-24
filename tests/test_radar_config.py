import json
import unittest
from pathlib import Path


class TestRadarConfig(unittest.TestCase):
    def test_required_keys_exist(self):
        cfg = json.loads(Path("/Users/huangyuxiang/openbb-agent/radar_config.json").read_text(encoding="utf-8"))
        for key in ["markets", "liquidity", "triggers", "risk", "output"]:
            self.assertIn(key, cfg)


if __name__ == "__main__":
    unittest.main()
