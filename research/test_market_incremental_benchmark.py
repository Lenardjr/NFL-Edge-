import unittest
import numpy as np
import pandas as pd
from research.market_incremental_benchmark import evaluate
class MarketIncrementalTests(unittest.TestCase):
    def sample(self):
        rows=[]
        for season in (2022,2023,2024):
            for i in range(110):
                rows.append({"game_id":f"{season}_{i}","season":season,"home_win":int(i%3!=0),
                  "market_prob":.35+.3*(i%7)/6,"v3_prob":.3+.4*(i%9)/8})
        return pd.DataFrame(rows)
    def test_strict_holdout(self):
        s,p=evaluate(self.sample())
        self.assertEqual(set(p.season),{2023,2024})
        self.assertEqual(s["overall"]["games"],220)
        self.assertTrue(p.market_plus_v3.between(0,1).all())
    def test_future_outcomes_do_not_change_prior_predictions(self):
        d=self.sample()
        _,p=evaluate(d)
        d.loc[d.season==2024,"home_win"]=1-d.loc[d.season==2024,"home_win"]
        _,q=evaluate(d)
        a=p[p.season==2023].reset_index(drop=True)
        b=q[q.season==2023].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
    def test_duplicate_rejected(self):
        d=self.sample();d.loc[1,"game_id"]=d.loc[0,"game_id"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):evaluate(d)
    def test_bad_probabilities_rejected(self):
        d=self.sample();d.loc[0,"market_prob"]=1.5
        with self.assertRaisesRegex(ValueError,"Invalid"):evaluate(d)
