import unittest
import pandas as pd
from research.qb_priority_review import prioritize
class PriorityTests(unittest.TestCase):
    def fixture(self):
        c=pd.DataFrame([{"game_id":"g1","gameday":"2025-09-01","home_team":"DET",
          "away_team":"GB","v3_prob":.75,"market_prob":.50,"home_win":1},
          {"game_id":"g2","gameday":"2025-09-02","home_team":"CHI",
          "away_team":"MIN","v3_prob":.55,"market_prob":.50,"home_win":0}])
        q=pd.DataFrame([{"game_id":g,"side":side,"historical_qb_name":g+side,
          "verification":"unreviewed","eligible_for_model":False}
          for g in ("g1","g2") for side in ("home","away")])
        return c,q
    def test_only_high_disagreements_selected(self):
        rows,s=prioritize(*self.fixture())
        self.assertEqual(s["priority_games"],1)
        self.assertEqual(len(rows),2)
        self.assertEqual(set(rows.game_id),{"g1"})
        self.assertNotIn("home_win",rows.columns)
        self.assertFalse(rows.eligible_for_model.any())
    def test_missing_qb_side_fails(self):
        c,q=self.fixture()
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            prioritize(c,q[q.side=="home"])
    def test_outcome_change_does_not_change_selection(self):
        c,q=self.fixture()
        a,_=prioritize(c,q)
        c["home_win"]=1-c.home_win
        b,_=prioritize(c,q)
        pd.testing.assert_frame_equal(a,b)
