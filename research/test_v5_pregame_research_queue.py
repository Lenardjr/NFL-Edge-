import unittest
import pandas as pd
from research.v5_pregame_research_queue import make_queue

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.d=pd.DataFrame({"game_id":["a","b","c","d"],"season":[2023,2023,2023,2024],
            "week":[1,2,3,1],"gameday":["2023-09-10"]*3+["2024-09-10"],
            "home_team":["A"]*4,"away_team":["B"]*4,
            "v3_prob":[.8,.9,.6,.2],"market_prob":[.5,.6,.5,.5],
            "absolute_gap":[.3,.3,.1,.3],"home_rest_days":[6]*4,
            "away_rest_days":[6]*4})
    def test_season_balanced_and_ties(self):
        q,s=make_queue(self.d,1)
        self.assertEqual(q.game_id.tolist(),["a","d"])
        self.assertEqual(s["eligible_extreme_games"],3)
        self.assertEqual(s["queued"],2)
    def test_no_outcome_dependence(self):
        d=self.d.copy();d["home_win"]=[0,1,0,1]
        a,_=make_queue(d)
        d.home_win=1-d.home_win
        b,_=make_queue(d)
        self.assertEqual(a.game_id.tolist(),b.game_id.tolist())
    def test_empty_sources(self):
        q,_=make_queue(self.d)
        self.assertTrue((q.source_verification_status=="not_started").all())
        self.assertTrue((q.qb_home_source=="").all())
    def test_inconsistent_gap_rejected(self):
        d=self.d.copy();d.loc[0,"absolute_gap"]=.7
        with self.assertRaises(ValueError):make_queue(d)
