import unittest
import pandas as pd
from research.audit_v2_parity import audit

class ParityTests(unittest.TestCase):
    def setUp(self):
        self.baseline=pd.DataFrame([{"game_id":"a","season":2022,"home_win":1,"model_home_prob":.65}])
        self.v2=pd.DataFrame([{"game_id":"a","season":2022,"home_win":1,"baseline_prob":.65,"v2_prob":.7}])
    def test_matching(self):
        self.assertTrue(audit(self.baseline,self.v2)["passed"])
    def test_prediction_drift(self):
        self.v2.loc[0,"baseline_prob"]=.66
        self.assertFalse(audit(self.baseline,self.v2)["passed"])
    def test_missing_game(self):
        extra=pd.DataFrame([{"game_id":"b","season":2022,"home_win":0,"model_home_prob":.4}])
        self.assertFalse(audit(pd.concat([self.baseline,extra],ignore_index=True),self.v2)["passed"])
if __name__=="__main__":unittest.main()
