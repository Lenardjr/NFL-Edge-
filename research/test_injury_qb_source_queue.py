import unittest
import pandas as pd
from research.injury_qb_source_queue import build_queue
class QBSourceQueueTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(game_id="g1",team="DET",side="home",full_name="Example QB",
          report_status="Questionable",date_modified="2024-09-07T12:00:00Z",
          kickoff_utc="2024-09-08T17:00:00Z",lead_hours=29.0,
          timing_class="modified_before_kickoff_unverified")])
    def test_unverified_by_default(self):
        q,s=build_queue(self.fixture())
        self.assertEqual(s["review_rows"],1)
        self.assertEqual(q.source_verification.iloc[0],"pending_independent_source")
        self.assertFalse(q.model_eligible.iloc[0])
        self.assertEqual(q.source_url.iloc[0],"")
    def test_duplicate_rejected(self):
        d=self.fixture();d=pd.concat([d,d],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Duplicate"):build_queue(d)
    def test_invalid_side_rejected(self):
        d=self.fixture();d.loc[0,"side"]="unknown"
        with self.assertRaisesRegex(ValueError,"Invalid side"):build_queue(d)
    def test_empty_queue(self):
        d=self.fixture().iloc[:0];q,s=build_queue(d)
        self.assertEqual(len(q),0)
        self.assertEqual(s["model_eligible"],0)
