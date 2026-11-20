import unittest
import pandas as pd
from research.market_incremental_gap_diagnostics import analyze
class MarketIncrementalGapDiagnosticsTests(unittest.TestCase):
    def sample(self):
        original=pd.DataFrame({"game_id":["a","b","c","d"],"v3_prob":[.51,.55,.62,.85],"market_prob":[.5,.5,.5,.5]})
        preds=pd.DataFrame({"game_id":["a","b","c","d"],"season":[2023]*4,"home_win":[1,0,1,0],
          "raw_market":[.5]*4,"market_calibrated":[.5]*4,"market_plus_v3":[.51,.55,.62,.85]})
        return preds,original
    def test_fixed_gap_bins_and_count(self):
        p,o=self.sample();r=analyze(p,o)
        self.assertEqual(sum(x["games"] for x in r["gap_bins"].values()),4)
        self.assertEqual(r["gap_bins"]["10pp_plus"]["games"],2)
    def test_outcomes_do_not_change_bin_membership(self):
        p,o=self.sample();a=analyze(p,o)
        p["home_win"]=1-p.home_win
        b=analyze(p,o)
        self.assertEqual({k:v["games"] for k,v in a["gap_bins"].items()},
                         {k:v["games"] for k,v in b["gap_bins"].items()})
    def test_mismatch_rejected(self):
        p,o=self.sample();p.loc[0,"raw_market"]=.6
        with self.assertRaisesRegex(ValueError,"do not match"):analyze(p,o)
    def test_duplicates_rejected(self):
        p,o=self.sample();o.loc[1,"game_id"]="a"
        with self.assertRaisesRegex(ValueError,"Duplicate"):analyze(p,o)
