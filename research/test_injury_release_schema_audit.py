import unittest
import pandas as pd
from research.injury_release_schema_audit import inspect
class InjurySchemaTests(unittest.TestCase):
    def test_schema_and_no_pregame_claim(self):
        d=pd.DataFrame({"team":["DET","GB"],"week":[1,1],"report_date":["2025-09-04","2025-09-04"],"status":["Out","Questionable"]})
        r=inspect(2025,d)
        self.assertEqual(r["rows"],2)
        self.assertEqual(r["verified_pregame_records"],0)
        self.assertFalse(r["eligible_for_model"])
        self.assertEqual(r["date_columns"],["report_date"])
    def test_empty_rejected(self):
        with self.assertRaisesRegex(ValueError,"Empty"):inspect(2025,pd.DataFrame({"team":[]}))
    def test_no_date_does_not_invent_timestamp(self):
        r=inspect(2024,pd.DataFrame({"team":["DET"],"status":["Out"]}))
        self.assertEqual(r["date_columns"],[])
