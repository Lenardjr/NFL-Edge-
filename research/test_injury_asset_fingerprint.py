import unittest
import pandas as pd
from research.injury_asset_fingerprint import fingerprint
class AssetFingerprintTests(unittest.TestCase):
    def fixtures(self):
        return pd.DataFrame([{"game_id":"g","asset_found":True,"asset_url":"https://github.com/org/repo/releases/download/tag/g.rds","asset_uploaded_at":"2024-01-01T00:00:00Z","uploaded_before_kickoff":True}])
    def test_download_hash_does_not_verify_content(self):
        d,s=fingerprint(self.fixtures(),lambda url:b"example")
        self.assertEqual(s["downloaded"],1)
        self.assertEqual(len(d.sha256.iloc[0]),64)
        self.assertFalse(d.model_eligible.iloc[0])
        self.assertFalse(d.content_parsed.iloc[0])
    def test_unmatched_not_downloaded(self):
        a=self.fixtures();a.loc[0,"asset_found"]=False
        d,s=fingerprint(a,lambda url: (_ for _ in ()).throw(Exception("unexpected")))
        self.assertEqual(s["downloaded"],0)
    def test_untrusted_host_rejected(self):
        a=self.fixtures();a.loc[0,"asset_url"]="https://example.com/file"
        d,_=fingerprint(a)
        self.assertEqual(d.download_status.iloc[0],"invalid_host")
    def test_fetch_failure_recorded(self):
        d,s=fingerprint(self.fixtures(),lambda url: (_ for _ in ()).throw(OSError("offline")))
        self.assertEqual(s["download_errors"],1)
    def test_duplicate_rejected(self):
        a=self.fixtures();a=pd.concat([a,a])
        with self.assertRaisesRegex(ValueError,"Invalid game"):fingerprint(a)
