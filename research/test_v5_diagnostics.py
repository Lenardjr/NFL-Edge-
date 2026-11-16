import unittest
import pandas as pd
from research.v5_diagnostics import analyze

class V5DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame({"game_id":["a","b","c","d"],"season":[2024]*4,
            "home_win":[1,0,1,0],"v3_prob":[.8,.3,.4,.6],
            "market_prob":[.7,.4,.6,.4]})
    def test_partition_counts(self):
        r=analyze(self.df)
        self.assertEqual(r["overall"]["games"],4)
        self.assertEqual(sum(g["games"] for g in r["market_probability_bins"]),4)
        self.assertEqual(sum(g["games"] for g in r["absolute_model_market_gap_bins"]),4)
        self.assertEqual(sum(g["games"] for g in r["disagreement"]),4)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):analyze(pd.concat([self.df,self.df.iloc[:1]]))
    def test_invalid_probability(self):
        d=self.df.copy();d.loc[0,"v3_prob"]=1.2
        with self.assertRaises(ValueError):analyze(d)
