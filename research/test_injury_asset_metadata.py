import unittest
import pandas as pd
from research.injury_asset_metadata import inspect
class InjuryAssetMetadataTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([{"game_id":"2024_12_DET_IND","priority_rank":1,"season":2024,"week":12}])
    def test_found_is_not_verified(self):
        d,s=inspect(self.fixture(),{"assets":[{"name":"2024_12_DET_IND.rds","created_at":"2024-11-24T20:37:30Z","id":123,"browser_download_url":"https://github.com/example"}]})
        self.assertTrue(d.asset_found.iloc[0]);self.assertFalse(d.model_eligible.iloc[0])
        self.assertEqual(s["assets_found"],1)
    def test_missing_is_safe(self):
        d,s=inspect(self.fixture(),{"assets":[]})
        self.assertFalse(d.asset_found.iloc[0]);self.assertEqual(s["assets_missing"],1)
    def test_duplicate_asset_rejected(self):
        a={"name":"2024_12_DET_IND.rds"}
        with self.assertRaisesRegex(ValueError,"Duplicate"):inspect(self.fixture(),{"assets":[a,a]})
    def test_duplicate_games_rejected(self):
        q=pd.concat([self.fixture(),self.fixture()])
        with self.assertRaisesRegex(ValueError,"Invalid game"):inspect(q,{"assets":[]})
    def test_bad_upload_rejected(self):
        with self.assertRaisesRegex(ValueError,"Invalid upload"):inspect(self.fixture(),{"assets":[{"name":"2024_12_DET_IND.rds","created_at":"nonsense"}]})
