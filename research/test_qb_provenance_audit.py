import unittest
import pandas as pd
from research.qb_provenance_audit import build_queue,validate_qb_evidence

class QBProvenanceTests(unittest.TestCase):
    def test_queue_is_unverified_by_default(self):
        q=build_queue(pd.DataFrame([{"game_id":"g1","gameday":"2025-10-01",
                                    "home_team":"DET","away_team":"GB"}]))
        self.assertEqual(len(q),2)
        self.assertFalse(q.eligible_for_model.any())
        self.assertTrue((q.verification=="unreviewed").all())
    def test_retrospective_label_not_enough(self):
        e=self.example();e.loc[0,"verification"]="retrospective"
        _,summary=validate_qb_evidence(e)
        self.assertEqual(summary["eligible"],0)
    def test_postkickoff_evidence_rejected(self):
        e=self.example();e.loc[0,"captured_at"]="2025-10-01T21:00:00Z"
        _,summary=validate_qb_evidence(e)
        self.assertEqual(summary["eligible"],0)
    def test_verified_timestamped_evidence_accepted(self):
        _,summary=validate_qb_evidence(self.example())
        self.assertEqual(summary["eligible"],1)
    def test_missing_source_rejected(self):
        e=self.example();e.loc[0,"source_url"]=""
        _,summary=validate_qb_evidence(e)
        self.assertEqual(summary["eligible"],0)
    def example(self):
        return pd.DataFrame([{"game_id":"g1","side":"home","qb_player_id":"00-001",
          "status":"expected_starter","captured_at":"2025-10-01T16:00:00Z",
          "kickoff_at":"2025-10-01T20:00:00Z","source":"archived report",
          "source_url":"https://example.org/report","verification":"verified_pregame"}])
