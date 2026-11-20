import unittest
import pandas as pd
from research.market_disagreement_report import summarize
class ReportTests(unittest.TestCase):
 def test_consensus_and_staleness(self):
  base={"game_id":"g","home_team":"KC","away_team":"ARI",
    "kickoff_utc":"2026-11-22T18:00:00Z","retrieved_at_utc":"2026-11-20T18:00:00Z",
    "model_home_win_probability":.85,"home_ev_per_unit":.1,"away_ev_per_unit":-.1}
  rows=[]
  for i,v in enumerate([.70,.72,.74,.1]):
   rows.append(dict(base,sportsbook=str(i),home_no_vig_probability=v,
    market_last_update_utc="2026-11-20T17:50:00Z" if i<3 else "2026-11-19T17:50:00Z"))
  x=summarize(pd.DataFrame(rows)).iloc[0]
  self.assertEqual(x.fresh_books,3)
  self.assertAlmostEqual(x.market_consensus_home_no_vig,.72)
  self.assertEqual(x.review_flag,"MONITOR")
 def test_no_fresh_quotes(self):
  r=pd.DataFrame([{"game_id":"g","home_team":"KC","away_team":"ARI",
    "kickoff_utc":"2026-11-22T18:00:00Z","retrieved_at_utc":"2026-11-20T18:00:00Z",
    "market_last_update_utc":"2026-11-19T18:00:00Z","sportsbook":"x",
    "model_home_win_probability":.8,"home_no_vig_probability":.7,
    "home_ev_per_unit":.1,"away_ev_per_unit":-.1}])
  x=summarize(r).iloc[0]
  self.assertEqual(x.review_flag,"INSUFFICIENT_FRESH_QUOTES")
  self.assertTrue(pd.isna(x.market_consensus_home_no_vig))
