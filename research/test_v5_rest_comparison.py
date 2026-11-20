import unittest
import pandas as pd
from research.v5_rest_comparison import analyze

class RestComparisonTests(unittest.TestCase):
    def setUp(self):
        self.s=pd.DataFrame({"game_id":["a","b","c","d"],"season":[2023]*4,
          "week":[1,1,2,2],"gameday":["2023-09-10","2023-09-10","2023-09-17","2023-09-17"],
          "home_team":["A","C","A","B"],"away_team":["B","D","C","D"]})
        self.p=pd.DataFrame({"game_id":["a","b","c","d"],"season":[2023]*4,
          "week":[1,1,2,2],"home_win":[1,0,1,0],
          "v3_prob":[.7,.4,.8,.3],"market_prob":[.5,.45,.55,.3]})
    def test_all_games_and_cohorts(self):
        r=analyze(self.p,self.s)
        self.assertEqual(r["total_games"],4)
        self.assertEqual(sum(g["games"] for g in r["gap_groups"]),4)
        self.assertEqual(sum(g["games"] for g in r["gap_by_rest"]),4)
        self.assertEqual(sum(g["games"] for g in r["gap_by_short_rest"]),4)
    def test_outcome_blind_cohort(self):
        a=analyze(self.p,self.s)
        q=self.p.copy();q.home_win=1-q.home_win
        b=analyze(q,self.s)
        self.assertEqual([(x["gap_group"],x["games"]) for x in a["gap_groups"]],
                         [(x["gap_group"],x["games"]) for x in b["gap_groups"]])
    def test_missing_schedule_fails(self):
        with self.assertRaises(ValueError):analyze(self.p,self.s.iloc[:3])
    def test_duplicate_prediction_fails(self):
        with self.assertRaises(ValueError):analyze(pd.concat([self.p,self.p.iloc[:1]]),self.s)
