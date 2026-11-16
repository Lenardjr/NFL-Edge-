import unittest
import pandas as pd
from research.v5_calibration_audit import audit

class CalibrationAuditTests(unittest.TestCase):
    def setUp(self):
        self.d=pd.DataFrame({"game_id":["a","b","c","d"],"season":[2023,2023,2024,2024],
           "home_win":[0,1,0,1],"v3_prob":[.2,.6,.9,.5],"market_prob":[.3,.7,.6,.5]})
    def test_counts(self):
        r=audit(self.d)
        self.assertEqual(r["games"],4)
        self.assertEqual(sum(x["games"] for x in r["signed_gap_bins"]),4)
        for v in r["probability_reliability"].values():
            self.assertEqual(sum(x["games"] for x in v),4)
    def test_duplicate_fails(self):
        with self.assertRaises(ValueError):audit(pd.concat([self.d,self.d.iloc[:1]]))
    def test_nan_fails(self):
        d=self.d.copy();d.loc[0,"v3_prob"]=float("nan")
        with self.assertRaises(ValueError):audit(d)
    def test_out_of_range_fails(self):
        d=self.d.copy();d.loc[0,"market_prob"]=1.1
        with self.assertRaises(ValueError):audit(d)
