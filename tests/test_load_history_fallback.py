import unittest
from unittest.mock import patch

import pandas as pd

import streamlit_app as app


class TestLoadHistoryFallback(unittest.TestCase):
    def test_load_history_returns_dataframe(self):
        with patch.object(app, "fetch_supabase_table", return_value=None):
            df = app.load_history.__wrapped__()
            self.assertIsInstance(df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
