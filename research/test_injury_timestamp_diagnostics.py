import unittest
import pandas as pd
from research.injury_timestamp_diagnostics import analyze,timestamp_format
class InjuryTimestampDiagnosticsTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(timestamp_format("2025-09-04T15:00:00Z"),"explicit_offset")
        self.assertEqual(timestamp_format("2025-09-04 15:00:00"),"naive_iso")
        self.assertEqual(timestamp_format("09/04/2025"),"naive_us")
        self.assertEqual(timestamp_format(None),"missing")
    def test_qb_counts_unverified(self):
        d=pd.DataFrame([{"season":2025,"date_modified":"2025-09-04T15:00:00Z",
           "timing_class":"modified_before_kickoff_unverified","game_id":"g1","team":"DET","week":1,"position":"QB"},
           {"season":2025,"date_modified":"2025-09-04 15:00:00",
           "timing_class":"timestamp_timezone_unverified","game_id":None,"team":"GB","week":1,"position":"WR"}])
        result=analyze(d)
        self.assertEqual(result["seasons"][0]["qb_rows"],1)
        self.assertEqual(result["verified_pregame_rows"],0)
    def test_empty_rejected(self):
        with self.assertRaisesRegex(ValueError,"Empty"):analyze(pd.DataFrame(columns=["season","date_modified","timing_class","game_id","team","week"]))
    def test_missing_columns_rejected(self):
        with self.assertRaisesRegex(ValueError,"Missing"):analyze(pd.DataFrame({"season":[2025]}))
