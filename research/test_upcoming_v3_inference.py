import unittest
import pandas as pd
from research.upcoming_v3_inference import predict_upcoming
class UpcomingV3InferenceTests(unittest.TestCase):
    def setUp(self):
        self.g=pd.DataFrame([{"game_id":"2026_11_A_B","season":2026,"week":11,
          "gameday":"2026-11-22","gametime":"13:00","home_team":"CHI","away_team":"NYJ",
          "home_score":None,"away_score":None,"kickoff_utc":"2026-11-22T18:00:00Z"}])
        self.epa=pd.DataFrame(columns=["game_id","team","offense_epa_per_play","offense_success_rate"])
        self.kw=dict(asof_utc="2026-11-19T12:00:00Z",now_utc="2026-11-19T13:00:00Z")
    def test_missing_historical_data_rejected(self):
        with self.assertRaisesRegex(ValueError,"Insufficient"):
            predict_upcoming(self.g,self.epa,**self.kw)
    def test_future_scores_rejected(self):
        self.g.loc[0,"home_score"]=24;self.g.loc[0,"away_score"]=14
        with self.assertRaisesRegex(ValueError,"final scores"):
            predict_upcoming(self.g,self.epa,**self.kw)
    def test_future_epa_rejected(self):
        self.epa=pd.DataFrame([{"game_id":"2026_11_A_B","team":"CHI",
          "offense_epa_per_play":.2,"offense_success_rate":.5}])
        with self.assertRaisesRegex(ValueError,"EPA includes"):
            predict_upcoming(self.g,self.epa,**self.kw)
    def test_future_snapshot_rejected(self):
        self.kw["asof_utc"]="2026-11-20T12:00:00Z"
        with self.assertRaisesRegex(ValueError,"later"):
            predict_upcoming(self.g,self.epa,**self.kw)
