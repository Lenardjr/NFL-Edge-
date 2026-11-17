import unittest
import pandas as pd
from research.kickoff_reference import build_kickoff_reference
class KickoffReferenceTests(unittest.TestCase):
    def data(self):
        p=pd.DataFrame({"game_id":["2024_01_DET_KC"]})
        s=pd.DataFrame([{"game_id":"2024_01_DET_KC","gameday":"2024-09-05","gametime":"20:20",
          "home_team":"KC","away_team":"DET","season":2024,"week":1}])
        return p,s
    def test_eastern_to_utc(self):
        p,s=self.data();result,summary=build_kickoff_reference(p,s)
        self.assertEqual(result.kickoff_utc.iloc[0],"2024-09-06T00:20:00+00:00")
        self.assertEqual(summary["verified_pregame_kickoffs"],0)
    def test_winter_offset(self):
        p,s=self.data();s.loc[0,"gameday"]="2024-12-05"
        result,_=build_kickoff_reference(p,s)
        self.assertEqual(result.kickoff_utc.iloc[0],"2024-12-06T01:20:00+00:00")
    def test_missing_match_rejected(self):
        p,s=self.data();s.loc[0,"game_id"]="other"
        with self.assertRaisesRegex(ValueError,"Unmatched"):build_kickoff_reference(p,s)
    def test_missing_time_rejected(self):
        p,s=self.data();s.loc[0,"gametime"]=None
        with self.assertRaisesRegex(ValueError,"Invalid game kickoff"):build_kickoff_reference(p,s)
    def test_duplicate_prediction_rejected(self):
        p,s=self.data();p=pd.concat([p,p],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Invalid prediction"):build_kickoff_reference(p,s)
