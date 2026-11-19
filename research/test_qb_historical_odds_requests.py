import unittest
import pandas as pd
from research.qb_historical_odds_requests import build_manifest
class HistoricalOddsRequestTests(unittest.TestCase):
    def case(self):
        return pd.DataFrame([{"game_id":"2024_18_KC_DEN","side":"away",
          "publisher_displayed_utc":"2025-01-01T19:01:00+00:00",
          "kickoff_utc":"2025-01-05T21:25:00+00:00","in_priority_queue":True}])
    def test_requests_not_quotes(self):
        d,s=build_manifest(self.case())
        self.assertEqual(len(d),4)
        self.assertEqual(s["verified_quotes"],0)
        self.assertFalse(d.model_eligible.any())
    def test_request_times(self):
        d,s=build_manifest(self.case())
        self.assertEqual(d.iloc[0].requested_snapshot_utc,"2024-12-31T19:01:00+00:00")
    def test_after_kickoff_excluded(self):
        e=self.case();e.loc[0,"kickoff_utc"]="2025-01-01T19:30:00+00:00"
        d,s=build_manifest(e)
        self.assertEqual(set(d.snapshot_window),{"pre_24h","pre_1h"})
    def test_no_valid_announcement(self):
        e=self.case();e.loc[0,"publisher_displayed_utc"]=None
        d,s=build_manifest(e)
        self.assertEqual(s["snapshot_requests"],0)
    def test_duplicate_rejected(self):
        e=self.case()
        with self.assertRaisesRegex(ValueError,"Duplicate"):build_manifest(pd.concat([e,e]))
