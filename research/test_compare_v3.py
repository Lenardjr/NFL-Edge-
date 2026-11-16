import unittest
import pandas as pd
from research.compare_v3 import compare

class V3ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "game_id":["a","b","c","d"],"season":[2022,2022,2023,2023],
            "home_win":[1,0,1,0],
            "baseline_prob":[.6,.4,.6,.4],
            "v3_prob":[.8,.2,.4,.6],
            "market_prob":[.7,.3,.7,.3]})
    def test_counts_and_disagreement(self):
        r=compare(self.df,draws=100)
        self.assertEqual(r["paired_games"],4)
        self.assertEqual(r["v3_market_disagreement"]["disagree"]["games"],2)
        self.assertEqual(r["v3_market_disagreement"]["market_correct_v3_wrong"],2)
        self.assertEqual(r["v3_market_disagreement"]["v3_correct_market_wrong"],0)
    def test_duplicate_rejected(self):
        d=pd.concat([self.df,self.df.iloc[:1]])
        with self.assertRaises(ValueError): compare(d,draws=10)
    def test_bad_probability_rejected(self):
        d=self.df.copy()
        d.loc[0,"v3_prob"]=1.2
        with self.assertRaises(ValueError): compare(d,draws=10)
