import unittest
import pandas as pd
from research.sample_prediction_market_audit import compare,implied
class SampleMarketAuditTests(unittest.TestCase):
    def test_implied(self):
        self.assertAlmostEqual(implied(-110),110/210)
        self.assertAlmostEqual(implied(150),100/250)
    def test_no_vig_and_no_roi(self):
        q=pd.DataFrame([{"market":"ml","side":s,"book":"a","quote_phase":p,"american_odds":v}
             for p in ["earliest","close"] for s,v in [("home",-150),("away",130)]])
        pred=pd.DataFrame([{"game_id":"2026_04_NYJ_CHI","model_home_prob":.75,"market_home_prob":.60,"home_win":1}])
        d,s=compare(q,pred,"2026_04_NYJ_CHI")
        self.assertEqual(len(d),2)
        self.assertEqual(s["backtest_wagers"],0)
        self.assertIsNone(s["realized_roi"])
        self.assertFalse(s["historical_prediction_timestamp_verified"])
    def test_missing_prediction(self):
        with self.assertRaisesRegex(ValueError,"exactly one"):
            compare(pd.DataFrame(),pd.DataFrame({"game_id":["other"]}),"missing")
