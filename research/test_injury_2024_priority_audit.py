import unittest
import pandas as pd
from research.injury_2024_priority_audit import analyze
class Injury2024PriorityTests(unittest.TestCase):
    def fixtures(self):
        d=pd.DataFrame([dict(season=2024,game_id="g1",team="DET",side="home",position="QB",date_modified="2024-09-07T17:00:00Z",kickoff_utc="2024-09-08T17:00:00Z",timing_class="modified_before_kickoff_unverified"),dict(season=2024,game_id="g2",team="GB",side="away",position="WR",date_modified="2024-09-08T18:00:00Z",kickoff_utc="2024-09-08T17:00:00Z",timing_class="modified_at_or_after_kickoff")])
        return d,pd.DataFrame({"game_id":["g1"]})
    def test_qb_extreme_priority(self):
        d,e=self.fixtures();q,s=analyze(d,e)
        self.assertEqual(s["qb_extreme_gap_rows"],1)
        self.assertEqual(s["lead_time_buckets"]["6_to_24h"],1)
        self.assertEqual(s["verified_pregame_rows"],0)
        self.assertEqual(q.game_id.iloc[0],"g1")
    def test_duplicate_extreme_rejected(self):
        d,e=self.fixtures();e=pd.concat([e,e],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Invalid extreme"):analyze(d,e)
    def test_missing_columns_rejected(self):
        _,e=self.fixtures()
        with self.assertRaisesRegex(ValueError,"Missing injury"):analyze(pd.DataFrame({"season":[2024]}),e)
    def test_missing_2024_rejected(self):
        d,e=self.fixtures();d["season"]=2025
        with self.assertRaisesRegex(ValueError,"No 2024"):analyze(d,e)
