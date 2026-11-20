import unittest
import pandas as pd
from research.injury_qb_rds_extract import extract
class QBExtractTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([{"game_id":"g","asset_found":True,"asset_url":"https://github.com/a/b.rds","uploaded_before_kickoff":True}])
    def test_extract_qb_not_rb(self):
        frame=pd.DataFrame({"position":["QB","RB"],"player_name":["Q","R"],"status":["Out","Active"]})
        d,a,s=extract(self.fixture(),lambda u:b"x",lambda b:frame)
        self.assertEqual(s["qb_candidate_rows"],1)
        self.assertEqual(d.player.iloc[0],"Q")
        self.assertFalse(d.model_eligible.iloc[0])
    def test_no_position_is_flagged(self):
        d,a,s=extract(self.fixture(),lambda u:b"x",lambda b:pd.DataFrame({"player":["Q"]}))
        self.assertEqual(s["files_with_unmapped_position"],1)
    def test_empty_qb_valid(self):
        d,a,s=extract(self.fixture(),lambda u:b"x",lambda b:pd.DataFrame({"pos":["RB"]}))
        self.assertEqual(len(d),0)
        self.assertEqual(s["files_with_extraction"],1)
    def test_invalid_host(self):
        a=self.fixture();a.loc[0,"asset_url"]="https://example.com"
        d,audit,s=extract(a)
        self.assertEqual(s["files_with_errors"],1)
    def test_duplicate_rejected(self):
        a=self.fixture()
        with self.assertRaisesRegex(ValueError,"Invalid game"):extract(pd.concat([a,a]))
