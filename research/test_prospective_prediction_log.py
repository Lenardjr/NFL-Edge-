import tempfile,unittest
from datetime import datetime,timezone,timedelta
from pathlib import Path
from research.prospective_prediction_log import prepare,append
class ProspectiveLogTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,11,19,tzinfo=timezone.utc)
        self.entry={"game_id":"2026_11_X_Y","kickoff_utc":"2026-11-22T18:00:00Z","model_version":"v3","home_win_probability":.62,"input_fingerprint":"a"*64}
    def test_valid(self):
        r=prepare(self.entry,self.now)
        self.assertEqual(len(r["prediction_fingerprint"]),64)
        self.assertEqual(r["independent_timestamp_verified"],"false")
    def test_reject_past(self):
        self.entry["kickoff_utc"]="2026-11-18T18:00:00Z"
        with self.assertRaises(ValueError):prepare(self.entry,self.now)
    def test_reject_invalid_prob(self):
        self.entry["home_win_probability"]=1.0
        with self.assertRaises(ValueError):prepare(self.entry,self.now)
    def test_reject_naive_kickoff(self):
        self.entry["kickoff_utc"]="2026-11-22T18:00:00"
        with self.assertRaises(ValueError):prepare(self.entry,self.now)
    def test_reject_invalid_fingerprint(self):
        self.entry["input_fingerprint"]="not-a-hash"
        with self.assertRaises(ValueError):prepare(self.entry,self.now)
    def test_immutable_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"ledger.csv";append(p,self.entry,self.now)
            with self.assertRaises(ValueError):append(p,self.entry,self.now)
            self.assertEqual(len(p.read_text().splitlines()),2)
