import unittest
import pandas as pd
from research.snapshot_nfl_odds import implied,parse_snapshot
class OddsTests(unittest.TestCase):
 def setUp(self):
  self.c=pd.DataFrame([dict(game_id="g",kickoff_utc="2026-11-22T18:00:00Z",home_team="KC",away_team="ARI",home_win_probability=.8)])
  self.e=[{"id":"e","commence_time":"2026-11-22T18:00:00Z","home_team":"Kansas City Chiefs","away_team":"Arizona Cardinals","bookmakers":[{"key":"book","markets":[{"key":"h2h","outcomes":[{"name":"Kansas City Chiefs","price":-200},{"name":"Arizona Cardinals","price":170}]}]}]}]
 def test_probabilities(self):
  self.assertAlmostEqual(implied(-200),2/3)
  self.assertAlmostEqual(implied(170),100/270)
 def test_match(self):
  x=parse_snapshot(self.c,self.e,"2026-11-20T18:00:00Z")
  self.assertEqual(len(x),1)
  self.assertAlmostEqual(x.iloc[0].home_ev_per_unit,.2)
 def test_no_postkickoff_quotes(self):
  self.assertEqual(len(parse_snapshot(self.c,self.e,"2026-11-22T18:00:00Z")),0)
 def test_mismatch(self):
  self.e[0]["away_team"]="Buffalo Bills"
  self.assertEqual(len(parse_snapshot(self.c,self.e,"2026-11-20T18:00:00Z")),0)
