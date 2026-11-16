import unittest
import pandas as pd
from research.compare_v2 import compare

class V2ComparisonTests(unittest.TestCase):
    def test_paired_score_and_counts(self):
        df=pd.DataFrame({"season":[2022,2022,2023,2023],
            "game_id":["a","b","c","d"],"home_win":[1,0,1,0],
            "baseline_prob":[.6,.4,.6,.4],"v2_prob":[.8,.2,.8,.2],
            "market_prob":[.7,.3,.7,.3]})
        result=compare(df,draws=100)
        self.assertEqual(result["games"],4)
        self.assertLess(result["v2_minus_baseline_brier"],0)
        self.assertLess(result["v2_minus_market_brier"],0)
        self.assertEqual(len(result["seasons"]),2)
    def test_duplicate_games_rejected(self):
        df=pd.DataFrame({"season":[2022,2022],"game_id":["a","a"],
            "home_win":[1,0],"baseline_prob":[.6,.4],
            "v2_prob":[.8,.2],"market_prob":[.7,.3]})
        with self.assertRaises(ValueError):compare(df,draws=10)
if __name__=="__main__":unittest.main()
