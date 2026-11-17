import unittest
import pandas as pd
from research.model_integrity_audit import audit
from research.market_validation import no_vig_two_way
from research.date_safe_features import date_safe_features

class IntegrityAuditTests(unittest.TestCase):
    def setUp(self):
        self.g=pd.DataFrame({"game_id":["a","b"],"gameday":["2025-09-07","2025-09-14"],
          "season":[2025,2025],"week":[1,2],"home_team":["DET","GB"],"away_team":["GB","DET"],
          "home_score":[24,10],"away_score":[17,20],"home_moneyline":[-150,130],
          "away_moneyline":[130,-150]})
        self.p=pd.DataFrame({"game_id":["a","b"],"season":[2025,2025],"week":[1,2],
          "home_win":[1,0],"v3_prob":[.6,.4],
          "market_prob":[no_vig_two_way(-150,130)[0],no_vig_two_way(130,-150)[0]]})
    def test_valid(self):
        self.assertEqual(audit(self.g,self.p)["status"],"passed_structural_checks")
    def test_market_orientation_rejected(self):
        p=self.p.copy();p.loc[0,"market_prob"]=1-p.loc[0,"market_prob"]
        with self.assertRaisesRegex(ValueError,"Market probability"):audit(self.g,p)
    def test_home_win_orientation_rejected(self):
        p=self.p.copy();p.loc[0,"home_win"]=0
        with self.assertRaisesRegex(ValueError,"orientation"):audit(self.g,p)
    def test_same_team_twice_rejected(self):
        g=self.g.copy();g.loc[1,"gameday"]=g.loc[0,"gameday"]
        with self.assertRaisesRegex(ValueError,"twice"):audit(g,self.p)
    def test_future_result_does_not_change_past_features(self):
        g=self.g.copy()
        a=date_safe_features(g).set_index("game_id")
        g.loc[g.game_id=="b","home_score"]=100
        b=date_safe_features(g).set_index("game_id")
        self.assertEqual(a.loc["a","elo_diff"],b.loc["a","elo_diff"])
        self.assertEqual(a.loc["b","elo_diff"],b.loc["b","elo_diff"])
