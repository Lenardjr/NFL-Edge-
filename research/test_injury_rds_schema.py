import unittest
import pandas as pd
from research.injury_rds_schema import inspect
class RDSSchemaTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([{"game_id":"g","asset_found":True,"asset_url":"https://github.com/a/b.rds","uploaded_before_kickoff":True}])
    def test_schema_without_eligibility(self):
        d,s=inspect(self.fixture(),lambda u:b"mock",lambda b:pd.DataFrame({"player_name":["X"],"position":["QB"],"status":["Out"]}))
        self.assertEqual(s["schema_inspected"],1)
        self.assertEqual(s["qb_records_verified"],0)
        self.assertFalse(d.model_eligible.iloc[0])
        self.assertIn("position",d.columns_json.iloc[0])
    def test_invalid_rds_recorded(self):
        d,s=inspect(self.fixture(),lambda u:b"mock",lambda b: (_ for _ in ()).throw(ValueError("bad")))
        self.assertEqual(s["schema_errors"],1)
    def test_skip_unmatched(self):
        a=self.fixture();a.loc[0,"asset_found"]=False
        d,s=inspect(a,lambda u: (_ for _ in ()).throw(Exception("called")))
        self.assertEqual(s["schema_inspected"],0)
    def test_reject_duplicate(self):
        a=self.fixture()
        with self.assertRaisesRegex(ValueError,"Invalid game"):inspect(pd.concat([a,a]))
    def test_reject_non_frame(self):
        d,s=inspect(self.fixture(),lambda u:b"mock",lambda b:{"x":1})
        self.assertEqual(s["schema_errors"],1)
