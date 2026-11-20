import unittest
import pandas as pd
from research.market_incremental_uncertainty import analyze
class MarketIncrementalUncertaintyTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([{"game_id":f"{s}_{i}","season":s,"home_win":i%2,
           "raw_market":.55,"market_calibrated":.53,"market_plus_v3":.56}
           for s in (2023,2024,2025,2026) for i in range(12)])
    def test_reproducible(self):
        a=analyze(self.sample(),draws=100,seed=17)
        b=analyze(self.sample(),draws=100,seed=17)
        self.assertEqual(a,b)
        self.assertEqual(a["paired_games"],48)
    def test_identical_models_have_zero_delta(self):
        d=self.sample();d["market_plus_v3"]=d.raw_market
        a=analyze(d,draws=50)
        self.assertEqual(a["combined_minus_raw_market_brier"]["mean"],0)
        self.assertEqual(a["combined_minus_raw_market_brier"]["season_cluster_bootstrap_95pct"],[0,0])
    def test_duplicates_rejected(self):
        d=self.sample();d.loc[1,"game_id"]=d.loc[0,"game_id"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):analyze(d)
    def test_invalid_probabilities_rejected(self):
        d=self.sample();d.loc[0,"market_plus_v3"]=1.2
        with self.assertRaisesRegex(ValueError,"Invalid"):analyze(d)
