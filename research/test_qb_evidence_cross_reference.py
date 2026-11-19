import unittest
import pandas as pd
from research.qb_evidence_cross_reference import cross_reference
class CrossReferenceTests(unittest.TestCase):
    def priority(self):
        return pd.DataFrame([{"game_id":"g","priority_rank":1,"absolute_gap":0.21,"season":2024,"week":18}])
    def evidence(self):
        return pd.DataFrame([{"game_id":"g","side":"home","qb_name":"Q","source_url":"https://example.com/article","published_date":"2025-01-01","game_date":"2025-01-05"}])
    def test_match_and_no_promotion(self):
        d,s=cross_reference(self.priority(),self.evidence())
        self.assertEqual(s["examples_matching_priority_games"],1)
        self.assertEqual(s["model_eligible"],0)
        self.assertFalse(d.model_eligible.iloc[0])
    def test_unmatched_retained(self):
        e=self.evidence();e.loc[0,"game_id"]="other"
        d,s=cross_reference(self.priority(),e)
        self.assertEqual(s["examples_outside_priority_games"],1)
    def test_publication_date_after_game_not_accepted(self):
        e=self.evidence();e.loc[0,"published_date"]="2025-01-06"
        d,s=cross_reference(self.priority(),e)
        self.assertEqual(s["publication_dates_before_game"],0)
    def test_duplicate_game_side_rejected(self):
        e=self.evidence()
        with self.assertRaisesRegex(ValueError,"Duplicate"):cross_reference(self.priority(),pd.concat([e,e]))
    def test_invalid_url_rejected(self):
        e=self.evidence();e.loc[0,"source_url"]="http://example.com"
        with self.assertRaisesRegex(ValueError,"Invalid evidence URL"):cross_reference(self.priority(),e)
