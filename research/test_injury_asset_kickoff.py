import unittest
import pandas as pd
from research.injury_asset_kickoff import evaluate
class AssetKickoffTests(unittest.TestCase):
    def fixtures(self):
        a=pd.DataFrame([{"game_id":"g1","asset_found":True,"asset_uploaded_at":"2024-09-08T16:00:00Z","asset_name":"g1.rds","asset_url":"https://example.com/a"},{"game_id":"g2","asset_found":False,"asset_uploaded_at":"","asset_name":"g2.rds","asset_url":""}])
        q=pd.DataFrame([{"game_id":"g1","kickoff_utc":"2024-09-08T17:00:00Z"},{"game_id":"g2","kickoff_utc":"2024-09-08T17:00:00Z"}])
        return a,q
    def test_before_kickoff_not_eligible(self):
        a,q=self.fixtures();d,s=evaluate(a,q)
        self.assertEqual(s["uploaded_before_kickoff"],1)
        self.assertEqual(s["unmatched_assets"],1)
        self.assertFalse(d.model_eligible.any())
        self.assertAlmostEqual(d.upload_hours_before_kickoff.iloc[0],1)
    def test_post_kickoff(self):
        a,q=self.fixtures();a.loc[0,"asset_uploaded_at"]="2024-09-08T18:00:00Z"
        _,s=evaluate(a,q);self.assertEqual(s["uploaded_at_or_after_kickoff"],1)
    def test_conflicting_kickoffs(self):
        a,q=self.fixtures();q=pd.concat([q,pd.DataFrame([{"game_id":"g1","kickoff_utc":"2024-09-09T17:00:00Z"}])])
        with self.assertRaisesRegex(ValueError,"Conflicting"):evaluate(a,q)
    def test_missing_upload(self):
        a,q=self.fixtures();a.loc[0,"asset_uploaded_at"]=""
        with self.assertRaisesRegex(ValueError,"missing upload"):evaluate(a,q)
    def test_unmatched_kickoff(self):
        a,q=self.fixtures();q=q[q.game_id!="g1"]
        with self.assertRaisesRegex(ValueError,"Unmatched"):evaluate(a,q)
