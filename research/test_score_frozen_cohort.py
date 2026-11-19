import unittest
import pandas as pd
from research.score_frozen_cohort import score_cohort
class CohortScoringTests(unittest.TestCase):
    def setUp(self):
        self.p=pd.DataFrame([dict(game_id="a",home_team="DET",away_team="CHI",
            kickoff_utc="2026-11-26T18:00:00Z",features_asof_utc="2026-11-19T19:12:53Z",
            home_win_probability=.75)])
        self.r=pd.DataFrame([dict(game_id="a",home_team="DET",away_team="CHI",
            home_score=21,away_score=17,status="final")])
    def test_score_verified_final(self):
        _,s=score_cohort(self.p,self.r)
        self.assertEqual(s["scored_games"],1)
        self.assertAlmostEqual(s["brier_score"],.0625)
    def test_no_final_no_score(self):
        self.r.loc[0,"status"]="scheduled"
        _,s=score_cohort(self.p,self.r)
        self.assertEqual(s["scored_games"],0)
        self.assertIsNone(s["brier_score"])
    def test_matchup_mismatch_rejected(self):
        self.r.loc[0,"home_team"]="GB"
        with self.assertRaisesRegex(ValueError,"mismatch"):
            score_cohort(self.p,self.r)
    def test_post_kickoff_prediction_rejected(self):
        self.p.loc[0,"features_asof_utc"]="2026-11-27T00:00:00Z"
        with self.assertRaisesRegex(ValueError,"Non-pregame"):
            score_cohort(self.p,self.r)
