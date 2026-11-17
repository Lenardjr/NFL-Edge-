import unittest
import pandas as pd
from research.injury_timing_audit import audit
class InjuryTimingAuditTests(unittest.TestCase):
    def fixtures(self):
        k=pd.DataFrame([dict(game_id="g1",season=2024,week=1,home_team="DET",away_team="GB",kickoff_utc="2024-09-08T17:00:00+00:00")])
        i=pd.DataFrame([dict(season=2024,week=1,team="DET",date_modified="2024-09-07T12:00:00+00:00")])
        return k,i
    def test_aware_prior_is_unverified(self):
        k,i=self.fixtures();d,s=audit(k,i)
        self.assertEqual(d.timing_class.iloc[0],"modified_before_kickoff_unverified")
        self.assertEqual(s["verified_pregame_rows"],0)
        self.assertFalse(d.eligible_for_model.iloc[0])
    def test_naive_not_assumed_utc(self):
        k,i=self.fixtures();i.loc[0,"date_modified"]="2024-09-07 12:00:00"
        d,_=audit(k,i);self.assertEqual(d.timing_class.iloc[0],"timestamp_timezone_unverified")
    def test_after_kickoff(self):
        k,i=self.fixtures();i.loc[0,"date_modified"]="2024-09-08T17:01:00Z"
        d,_=audit(k,i);self.assertEqual(d.timing_class.iloc[0],"modified_at_or_after_kickoff")
    def test_unmatched_team_week(self):
        k,i=self.fixtures();i.loc[0,"team"]="KC"
        d,_=audit(k,i);self.assertEqual(d.timing_class.iloc[0],"unmatched_team_week")
    def test_ambiguous_team_week_rejected(self):
        k,i=self.fixtures();k=pd.concat([k,k.assign(game_id="g2")],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Ambiguous"):audit(k,i)
    def test_duplicate_game_id_rejected(self):
        k,i=self.fixtures();k=pd.concat([k,k],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Invalid kickoff"):audit(k,i)
