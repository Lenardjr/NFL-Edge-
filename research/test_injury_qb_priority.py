import unittest
import pandas as pd
from research.injury_qb_priority import prioritize
class QBPriorityTests(unittest.TestCase):
    def fixtures(self):
        q=pd.DataFrame([dict(game_id="a",team="DET",side="home",player="QB1",date_modified="2024-01-01Z",kickoff_utc="2024-01-02Z",source_verification="pending_independent_source"),dict(game_id="b",team="GB",side="away",player="QB2",date_modified="2024-01-01Z",kickoff_utc="2024-01-02Z",source_verification="pending_independent_source")])
        c=pd.DataFrame([dict(game_id="a",season=2024,week=18,v3_prob=.9,market_prob=.5,home_win=1),dict(game_id="b",season=2024,week=1,v3_prob=.6,market_prob=.5,home_win=0)])
        return q,c
    def test_rank_uses_gap_not_outcome(self):
        q,c=self.fixtures();d,s=prioritize(q,c)
        self.assertEqual(d.game_id.tolist(),["a","b"])
        self.assertEqual(s["week18_games"],1)
        c.home_win=1-c.home_win
        self.assertTrue(d.equals(prioritize(q,c)[0]))
    def test_limit(self):
        q,c=self.fixtures();d,_=prioritize(q,c,1)
        self.assertEqual(d.game_id.tolist(),["a"])
    def test_unmatched_rejected(self):
        q,c=self.fixtures();q.loc[0,"game_id"]="missing"
        with self.assertRaisesRegex(ValueError,"absent"):prioritize(q,c)
    def test_duplicate_case_rejected(self):
        q,c=self.fixtures();c=pd.concat([c,c],ignore_index=True)
        with self.assertRaisesRegex(ValueError,"Invalid case"):prioritize(q,c)
    def test_invalid_prob_rejected(self):
        q,c=self.fixtures();c.loc[0,"market_prob"]=1.5
        with self.assertRaisesRegex(ValueError,"Invalid probability"):prioritize(q,c)
