import unittest
import pandas as pd
from research.injury_qb_rds_inventory import summarize
class QBInventoryTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([{"game_id":"g","asset_found":True,"asset_url":"https://github.com/o/r/file.rds","uploaded_before_kickoff":True}])
    def test_count_case_insensitive_qb(self):
        d,s=summarize(self.fixture(),lambda u:b"x",lambda b:pd.DataFrame({"position":["QB"," qb ","RB",None]}))
        self.assertEqual(s["qb_candidate_rows"],2)
        self.assertEqual(s["model_eligible"],0)
    def test_unmapped_not_assumed_zero_verified(self):
        d,s=summarize(self.fixture(),lambda u:b"x",lambda b:pd.DataFrame({"role":["QB"]}))
        self.assertEqual(s["unmapped_position_schema_files"],1)
        self.assertEqual(s["qb_records_pregame_verified"],0)
    def test_skips_unmatched(self):
        a=self.fixture();a.loc[0,"asset_found"]=False
        d,s=summarize(a,lambda u: (_ for _ in ()).throw(Exception("unexpected")))
        self.assertEqual(d.status.iloc[0],"skipped")
    def test_rejects_duplicate(self):
        a=self.fixture()
        with self.assertRaisesRegex(ValueError,"Duplicate"):summarize(pd.concat([a,a]))
    def test_download_error(self):
        d,s=summarize(self.fixture(),lambda u: (_ for _ in ()).throw(OSError("down")))
        self.assertEqual(s["read_errors"],1)
