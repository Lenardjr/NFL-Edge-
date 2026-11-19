import tempfile,unittest
from datetime import datetime,timezone
from pathlib import Path
from research.live_prediction_bridge import record_live_inference
class LivePredictionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.kw=dict(game_id="2026_11_A_B",kickoff_utc="2026-11-22T18:00:00Z",
            model_version="live-v3",home_win_probability=.61,input_fingerprint="a"*64,
            features_asof_utc="2026-11-19T12:00:00Z",
            model_trained_through_utc="2026-11-18T00:00:00Z",
            now=datetime(2026,11,19,13,tzinfo=timezone.utc))
    def test_accept_valid(self):
        with tempfile.TemporaryDirectory() as d:
            r=record_live_inference(Path(d)/"ledger.csv",**self.kw)
            self.assertEqual(r["source_provenance"],"live_inference_unverified")
    def test_reject_historical_label(self):
        self.kw["model_version"]="v3"
        with self.assertRaises(ValueError):record_live_inference("unused.csv",**self.kw)
    def test_reject_future_features(self):
        self.kw["features_asof_utc"]="2026-11-20T12:00:00Z"
        with self.assertRaises(ValueError):record_live_inference("unused.csv",**self.kw)
    def test_reject_training_leak(self):
        self.kw["model_trained_through_utc"]="2026-11-19T13:00:00Z"
        with self.assertRaises(ValueError):record_live_inference("unused.csv",**self.kw)
    def test_reject_postkickoff(self):
        self.kw["now"]=datetime(2026,11,23,tzinfo=timezone.utc)
        with self.assertRaises(ValueError):record_live_inference("unused.csv",**self.kw)
