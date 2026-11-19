import unittest
import pandas as pd
from research.free_odds_sample import validate_sample
class FreeOddsSampleTests(unittest.TestCase):
    def row(self):
        return {"snapshot_ts":"2026-10-20T15:00:00Z","sport":"americanfootball_nfl",
         "away":"A","home":"B","commence_time":"2026-10-20T23:00:00Z",
         "market":"ml","side":"home","book":"draftkings","american_odds":-110}
    def test_accepts_pregame(self):
        d,s=validate_sample(pd.DataFrame([self.row()]))
        self.assertEqual(s["accepted_pregame_sample_rows"],1)
        self.assertFalse(bool(d.iloc[0].model_eligible))
    def test_rejects_post_kickoff(self):
        r=self.row();r["snapshot_ts"]="2026-10-21T00:00:00Z"
        d,s=validate_sample(pd.DataFrame([r]));self.assertEqual(len(d),0)
    def test_rejects_other_week(self):
        r=self.row();r["commence_time"]="2026-11-01T23:00:00Z"
        d,s=validate_sample(pd.DataFrame([r]));self.assertEqual(len(d),0)
    def test_rejects_invalid_odds(self):
        r=self.row();r["american_odds"]=0
        d,s=validate_sample(pd.DataFrame([r]));self.assertEqual(len(d),0)
    def test_missing_columns(self):
        with self.assertRaisesRegex(ValueError,"Missing odds columns"):
            validate_sample(pd.DataFrame([{"sport":"americanfootball_nfl"}]))
