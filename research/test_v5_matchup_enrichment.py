import unittest
import pandas as pd
from research.v5_matchup_enrichment import enrich

class MatchupEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.s=pd.DataFrame({"game_id":["g1","g2","g3"],"season":[2023]*3,
          "week":[1,1,2],"gameday":["2023-09-10","2023-09-10","2023-09-17"],
          "home_team":["A","C","A"],"away_team":["B","D","C"]})
        self.c=pd.DataFrame({"game_id":["g3"],"season":[2023],"absolute_gap":[.25]})
    def test_matchup_and_rest(self):
        d=enrich(self.c,self.s)
        self.assertEqual(d.home_team.iloc[0],"A")
        self.assertEqual(d.away_team.iloc[0],"C")
        self.assertEqual(d.home_rest_days.iloc[0],6)
        self.assertEqual(d.away_rest_days.iloc[0],6)
        self.assertEqual(d.rest_advantage_home.iloc[0],0)
    def test_unknown_first_game_rest(self):
        d=enrich(pd.DataFrame({"game_id":["g1"],"season":[2023]}),self.s)
        self.assertTrue(pd.isna(d.home_rest_days.iloc[0]))
    def test_missing_case_rejected(self):
        with self.assertRaises(ValueError):
            enrich(pd.DataFrame({"game_id":["missing"],"season":[2023]}),self.s)
    def test_duplicate_schedule_rejected(self):
        with self.assertRaises(ValueError):enrich(self.c,pd.concat([self.s,self.s.iloc[:1]]))
    def test_season_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            enrich(pd.DataFrame({"game_id":["g3"],"season":[2024]}),self.s)
