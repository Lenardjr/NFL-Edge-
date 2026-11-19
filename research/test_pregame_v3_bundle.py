import unittest
import pandas as pd
from research.pregame_v3_bundle import filter_pregame_epa
class PregameBundleTests(unittest.TestCase):
    def setUp(self):
        self.g=pd.DataFrame([dict(game_id="old",gameday="2026-11-15",
            home_team="CHI",away_team="NYJ",home_score=20,away_score=10),
            dict(game_id="future",gameday="2026-11-22",
            home_team="CHI",away_team="NYJ",home_score=None,away_score=None)])
        self.e=pd.DataFrame([dict(game_id="old",team=t,
            offense_epa_per_play=.1,offense_success_rate=.5) for t in ("CHI","NYJ")])
    def check(self):
        return filter_pregame_epa(self.g,self.e,asof_utc="2026-11-19T17:00:00Z")
    def test_completed_game_allowed(self):
        self.assertEqual(len(self.check()),2)
    def test_future_epa_rejected(self):
        self.e.loc[0,"game_id"]="future"
        with self.assertRaisesRegex(ValueError,"future games"):
            self.check()
    def test_partial_team_game_rejected(self):
        self.e=self.e.iloc[:1]
        with self.assertRaisesRegex(ValueError,"missing team"):
            self.check()
    def test_unfinished_prior_game_rejected(self):
        self.g.loc[0,"home_score"]=None
        with self.assertRaisesRegex(ValueError,"uncompleted"):
            self.check()
