import unittest
import pandas as pd
from research.qb_publisher_timestamp_review import review
class PublisherTests(unittest.TestCase):
    def case(self):
        return pd.DataFrame([{"game_id":"2023_18_PIT_BAL","side":"home","kickoff_utc":"2024-01-06T21:30:00Z","in_priority_queue":True}])
    def test_publisher_time_not_promoted(self):
        d,s=review(self.case())
        self.assertEqual(s["publisher_timestamps_found"],1)
        self.assertTrue(d.publisher_timestamp_before_kickoff.iloc[0])
        self.assertFalse(d.model_eligible.iloc[0])
    def test_time_zone_conversion(self):
        d,s=review(self.case())
        self.assertEqual(d.publisher_displayed_utc.iloc[0],"2024-01-03T20:42:00+00:00")
    def test_unmatched_kept(self):
        d=self.case();d.loc[0,"game_id"]="unknown"
        x,s=review(d)
        self.assertEqual(s["publisher_timestamps_found"],0)
    def test_duplicate_rejected(self):
        d=self.case()
        with self.assertRaisesRegex(ValueError,"Duplicate timeline"):review(pd.concat([d,d]))
    def test_no_market_quote_claim(self):
        d,s=review(self.case())
        self.assertEqual(s["verified_market_quotes"],0)
        self.assertEqual(s["independent_captures_verified"],0)
