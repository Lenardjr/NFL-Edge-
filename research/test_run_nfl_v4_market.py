import unittest
import pandas as pd
from research.run_nfl_v4_market import evaluate

class V4MarketTests(unittest.TestCase):
    def make_data(self):
        rows=[]
        for season in (2022,2023,2024):
            for i in range(220):
                rows.append({"season":season,"game_id":f"{season}-{i}",
                             "home_win":i%2,"v3_prob":.4 if i%2 else .6,
                             "market_prob":.45 if i%2 else .55})
        return pd.DataFrame(rows)
    def test_prior_season_only(self):
        summary,p=evaluate(self.make_data())
        self.assertEqual(summary["games"],440)
        self.assertEqual(set(p.season),{2023,2024})
    def test_duplicate_rejected(self):
        d=self.make_data()
        with self.assertRaises(ValueError):
            evaluate(pd.concat([d,d.iloc[:1]]))
    def test_missing_price_rejected(self):
        d=self.make_data()
        d.loc[0,"market_prob"]=float("nan")
        with self.assertRaises(ValueError):evaluate(d)
