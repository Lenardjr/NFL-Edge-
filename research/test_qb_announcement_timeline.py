import unittest
import pandas as pd
from research.qb_announcement_timeline import timeline
class TimelineTests(unittest.TestCase):
    def example(self):
        e=pd.DataFrame([{"game_id":"2024_18_KC_DEN","side":"away","qb_name":"Carson Wentz","source_url":"https://example.com","published_date":"2025-01-01","game_date":"2025-01-05","in_priority_queue":True}])
        k=pd.DataFrame([{"game_id":"2024_18_KC_DEN","kickoff_utc":"2025-01-05T21:25:00+00:00"}])
        return e,k
    def test_dated_source_not_eligible(self):
        d,s=timeline(*self.example())
        self.assertTrue(d.published_date_before_kickoff_day.iloc[0])
        self.assertFalse(d.model_eligible.iloc[0])
        self.assertEqual(s["verified_market_quotes_at_decision"],0)
    def test_missing_kickoff(self):
        e,k=self.example();k.loc[0,"kickoff_utc"]=None
        d,s=timeline(e,k)
        self.assertEqual(s["kickoff_missing"],1)
    def test_duplicate_kickoff_rejected(self):
        e,k=self.example()
        with self.assertRaisesRegex(ValueError,"Duplicate"):timeline(e,pd.concat([k,k]))
    def test_same_day_not_assumed_early(self):
        e,k=self.example();e.loc[0,"published_date"]="2025-01-05"
        d,s=timeline(e,k)
        self.assertFalse(d.published_date_before_kickoff_day.iloc[0])
    def test_unmatched_evidence_retained(self):
        e,k=self.example();e.loc[0,"game_id"]="unknown"
        d,s=timeline(e,k)
        self.assertEqual(s["kickoff_missing"],1)
