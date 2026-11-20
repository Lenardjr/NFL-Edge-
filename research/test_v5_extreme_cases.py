import unittest
import pandas as pd
from research.v5_extreme_cases import build_cases,summarize

class ExtremeCaseTests(unittest.TestCase):
    def setUp(self):
        self.d=pd.DataFrame({"game_id":["a","b","c"],"season":[2023,2023,2024],
           "week":[1,2,1],"home_win":[1,0,1],
           "v3_prob":[.8,.4,.3],"market_prob":[.6,.42,.6]})
    def test_outcome_blind_selection(self):
        a=build_cases(self.d)
        flipped=self.d.copy();flipped["home_win"]=1-flipped.home_win
        b=build_cases(flipped)
        self.assertEqual(a.game_id.tolist(),b.game_id.tolist())
        self.assertEqual(a.game_id.tolist(),["c","a"])
        self.assertEqual(summarize(a)["cases"],2)
    def test_review_fields_unverified(self):
        a=build_cases(self.d)
        self.assertTrue((a.review_status=="unreviewed").all())
        self.assertTrue((a.source_urls=="").all())
    def test_duplicate_fails(self):
        with self.assertRaises(ValueError):build_cases(pd.concat([self.d,self.d.iloc[:1]]))
    def test_invalid_probability_fails(self):
        d=self.d.copy();d.loc[0,"market_prob"]=1.1
        with self.assertRaises(ValueError):build_cases(d)
