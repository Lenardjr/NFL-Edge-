import unittest
import pandas as pd
from research.injury_source_provenance import audit
class SourceProvenanceTests(unittest.TestCase):
    def test_no_automatic_eligibility(self):
        q,s=audit(pd.DataFrame([{"game_id":"g","season":2024,"week":18,"priority_rank":1,"absolute_gap":.4}]))
        self.assertFalse(q.model_eligible.iloc[0])
        self.assertFalse(q.source_verified_before_decision.iloc[0])
        self.assertEqual(s["verified_before_prediction"],0)
    def test_duplicate_rejected(self):
        q=pd.DataFrame([{"game_id":"g","season":2024,"week":18,"priority_rank":1,"absolute_gap":.4}]*2)
        with self.assertRaisesRegex(ValueError,"Invalid priority"):audit(q)
    def test_missing_fields_rejected(self):
        with self.assertRaisesRegex(ValueError,"Missing priority"):audit(pd.DataFrame({"game_id":["g"]}))
    def test_empty_allowed(self):
        q,s=audit(pd.DataFrame(columns=["game_id","season","week","priority_rank","absolute_gap"]))
        self.assertEqual(len(q),0)
        self.assertEqual(s["model_eligible"],0)
