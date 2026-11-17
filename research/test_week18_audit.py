import unittest
import pandas as pd
from research.week18_audit import audit
class Week18AuditTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([
            {"game_id":"a","season":2024,"week":18,"home_win":1,"v3_prob":.9,"market_prob":.6},
            {"game_id":"b","season":2024,"week":17,"home_win":0,"v3_prob":.2,"market_prob":.4},
            {"game_id":"c","season":2025,"week":18,"home_win":0,"v3_prob":.8,"market_prob":.5}])
    def test_segments_and_gap(self):
        s=audit(self.sample())
        self.assertEqual(s["segments"]["week_18"]["games"],2)
        self.assertEqual(s["segments"]["week_18"]["large_gap_games"],2)
        self.assertEqual(s["segments"]["other_weeks"]["games"],1)
        self.assertIn("2024",s["by_season"])
    def test_duplicate_rejected(self):
        p=self.sample();p.loc[1,"game_id"]="a"
        with self.assertRaisesRegex(ValueError,"Duplicate"):audit(p)
    def test_invalid_probability_rejected(self):
        p=self.sample();p.loc[0,"v3_prob"]=1.2
        with self.assertRaisesRegex(ValueError,"Invalid"):audit(p)
