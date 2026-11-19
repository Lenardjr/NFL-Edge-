import unittest
import pandas as pd
from research.historical_odds_request_plan import plan
class HistoricalOddsRequestPlanTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([{"game_id":"a","snapshot_window":"pre_1h",
           "requested_snapshot_utc":"2025-01-01T19:01:00+00:00","in_priority_queue":True}])
    def test_url_has_no_key(self):
        d,s=plan(self.rows());self.assertIn("americanfootball_nfl/odds?",d.iloc[0].request_url_without_key)
        self.assertNotIn("apiKey",d.iloc[0].request_url_without_key)
    def test_quota(self):
        d,s=plan(self.rows());self.assertEqual(s["estimated_credits_with_timestamp_deduplication"],10)
    def test_deduplication(self):
        d,s=plan(pd.concat([self.rows(),self.rows()]))
        self.assertEqual(s["estimated_credits_without_deduplication"],20)
        self.assertEqual(s["estimated_credits_with_timestamp_deduplication"],10)
    def test_no_network_or_quotes(self):
        d,s=plan(self.rows());self.assertEqual(s["network_calls_made"],0)
        self.assertEqual(s["quotes_retrieved"],0)
    def test_invalid_time(self):
        e=self.rows();e.loc[0,"requested_snapshot_utc"]="bad"
        with self.assertRaisesRegex(ValueError,"Invalid requested time"):plan(e)
