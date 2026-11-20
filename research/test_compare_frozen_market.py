import unittest
import pandas as pd
from research.compare_frozen_market import compare
class PairedTests(unittest.TestCase):
 def setUp(self):
  self.p=pd.DataFrame([{"game_id":"g","home_win_probability":.8,"kickoff_utc":"2026-11-22T18:00:00Z"}])
  self.r=pd.DataFrame([{"game_id":"g","status":"final","home_score":24,"away_score":10}])
  self.q=pd.DataFrame([dict(game_id="g",sportsbook=str(i),home_no_vig_probability=.7,
    retrieved_at_utc="2026-11-20T18:00:00Z",market_last_update_utc="2026-11-20T17:55:00Z") for i in range(3)])
 def test_paired_brier(self):
  games,s=compare(self.p,self.r,self.q)
  self.assertEqual(s["paired_games_with_fresh_pregame_market"],1)
  self.assertAlmostEqual(s["model_brier_on_paired"],.04)
  self.assertAlmostEqual(s["market_brier_on_paired"],.09)
  self.assertFalse(s["market_outperformance_verified"])
 def test_no_postkickoff_market(self):
  self.q["retrieved_at_utc"]="2026-11-22T19:00:00Z"
  _,s=compare(self.p,self.r,self.q)
  self.assertEqual(s["paired_games_with_fresh_pregame_market"],0)
 def test_no_unfinished_games(self):
  self.r["status"]="scheduled"
  _,s=compare(self.p,self.r,self.q)
  self.assertEqual(s["completed_scored_games"],0)
