import unittest
import pandas as pd
from research.collect_cohort_scores import parse_events,match_results
class ScoreboardTests(unittest.TestCase):
 def setUp(self):
  self.p=pd.DataFrame([dict(game_id="a",home_team="HOU",away_team="IND",kickoff_utc="2026-11-20T01:15:00Z")])
  self.event={"id":"123","date":"2026-11-20T01:15:00Z","competitions":[{"date":"2026-11-20T01:15:00Z","status":{"type":{"completed":True}},"competitors":[{"homeAway":"home","team":{"abbreviation":"HOU"},"score":"24"},{"homeAway":"away","team":{"abbreviation":"IND"},"score":"17"}]}]}
 def test_completed(self):
  e=parse_events({"events":[self.event]})
  r=match_results(self.p,e,now_utc="2026-11-21T00:00:00Z")
  self.assertEqual(r.iloc[0].status,"final")
  self.assertEqual(r.iloc[0].home_score,24)
 def test_live_never_scores(self):
  self.event["competitions"][0]["status"]["type"]["completed"]=False
  r=match_results(self.p,parse_events({"events":[self.event]}),now_utc="2026-11-21T00:00:00Z")
  self.assertEqual(r.iloc[0].status,"unverified")
  self.assertTrue(pd.isna(r.iloc[0].home_score))
 def test_wrong_team_rejected(self):
  self.event["competitions"][0]["competitors"][1]["team"]["abbreviation"]="NYJ"
  r=match_results(self.p,parse_events({"events":[self.event]}),now_utc="2026-11-21T00:00:00Z")
  self.assertEqual(r.iloc[0].status,"unverified")
 def test_ambiguous_rejected(self):
  e=parse_events({"events":[self.event]})
  with self.assertRaisesRegex(ValueError,"Ambiguous"):
   match_results(self.p,e+e,now_utc="2026-11-21T00:00:00Z")
