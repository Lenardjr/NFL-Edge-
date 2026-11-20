import tempfile,unittest
from pathlib import Path
import pandas as pd
from research.next_slate_v3 import freeze_slate,export_slate

class NextSlateTests(unittest.TestCase):
    def setUp(self):
        self.now="2026-11-19T19:10:00Z"
        self.data=pd.DataFrame([
            dict(game_id="a",kickoff_utc="2026-11-20T01:15:00Z",home_team="HOU",away_team="IND",
                 home_win_probability=.71,input_fingerprint="a"*64,model_version="v3",
                 features_asof_utc="2026-11-19T19:04:44Z"),
            dict(game_id="b",kickoff_utc="2026-12-15T18:00:00Z",home_team="DET",away_team="GB",
                 home_win_probability=.60,input_fingerprint="b"*64,model_version="v3",
                 features_asof_utc="2026-11-19T19:04:44Z")])
    def test_next_week_only(self):
        self.assertEqual(freeze_slate(self.data,now_utc=self.now).game_id.tolist(),["a"])
    def test_reject_post_kickoff_snapshot(self):
        self.data.loc[0,"features_asof_utc"]="2026-11-20T02:00:00Z"
        with self.assertRaisesRegex(ValueError,"on or after kickoff"):
            freeze_slate(self.data,now_utc="2026-11-21T00:00:00Z")
    def test_reject_duplicate(self):
        self.data.loc[1,"game_id"]="a"
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            freeze_slate(self.data,now_utc=self.now)
    def test_export_no_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);src=p/"predictions.csv";self.data.to_csv(src,index=False)
            manifest=export_slate(prediction_csv=src,output_dir=p/"out",now_utc=self.now)
            self.assertEqual(manifest["selected_games"],1)
            self.assertFalse(manifest["prospective_ledger_written"])
            self.assertTrue((p/"out"/"next_slate_v3_manifest.json").exists())
