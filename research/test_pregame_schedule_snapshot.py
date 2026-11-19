import unittest
import pandas as pd
from research.pregame_schedule_snapshot import convert_schedule
class ScheduleSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.g=pd.DataFrame([dict(game_id="2026_11_NYJ_CHI",season=2026,week=11,
           gameday="2026-11-22",gametime="13:00",home_team="CHI",away_team="NYJ",
           home_score=None,away_score=None)])
        self.kw=dict(captured_at_utc="2026-11-19T17:00:00Z",asof_utc="2026-11-19T17:00:00Z")
    def test_dst_winter_kickoff(self):
        x=convert_schedule(self.g,**self.kw)
        self.assertEqual(x.kickoff_utc.iloc[0],"2026-11-22T18:00:00+00:00")
    def test_dst_summer_kickoff(self):
        self.g.loc[0,"gameday"]="2026-09-20"
        x=convert_schedule(self.g,**self.kw)
        self.assertEqual(x.kickoff_utc.iloc[0],"2026-09-20T17:00:00+00:00")
    def test_reject_backdating(self):
        self.kw["asof_utc"]="2026-11-18T17:00:00Z"
        with self.assertRaisesRegex(ValueError,"backdate"):
            convert_schedule(self.g,**self.kw)
    def test_reject_missing_time(self):
        self.g.loc[0,"gametime"]=None
        with self.assertRaisesRegex(ValueError,"Missing gametime"):
            convert_schedule(self.g,**self.kw)
    def test_suppress_same_day_score(self):
        self.g.loc[0,"gameday"]="2026-11-19"
        self.g.loc[0,"home_score"]=24;self.g.loc[0,"away_score"]=10
        x=convert_schedule(self.g,**self.kw)
        self.assertTrue(pd.isna(x.home_score.iloc[0]))
