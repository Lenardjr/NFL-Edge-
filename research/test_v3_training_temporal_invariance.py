"""Integration tests: V3 fit and predictions must ignore future results and EPA."""
import unittest
import numpy as np
import pandas as pd
from research.run_nfl_v3 import evaluate

def synthetic_history():
    games=[];epa=[]
    teams=[("DET","GB"),("CHI","MIN")]
    idx=0
    for season,days in [(2021,160),(2022,6)]:
        for day in range(days):
            date=pd.Timestamp(f"{season}-01-01")+pd.Timedelta(days=day)
            for pair,(home,away) in enumerate(teams):
                # Rotate home and away to avoid degenerate feature patterns.
                if (day+pair)%2:home,away=away,home
                gid=f"{season}_{day}_{pair}"
                home_score=14+(idx*7)%24
                away_score=13+(idx*11)%25
                games.append(dict(game_id=gid,gameday=date.strftime("%Y-%m-%d"),
                                  season=season,week=day+1,home_team=home,away_team=away,
                                  home_score=home_score,away_score=away_score,
                                  home_moneyline=-125,away_moneyline=110))
                for j,team in enumerate((home,away)):
                    epa.append(dict(game_id=gid,team=team,
                                    offense_epa_per_play=((idx+j)%13-6)/20,
                                    offense_success_rate=.35+((idx+j)%10)/40))
                idx+=1
    return pd.DataFrame(games),pd.DataFrame(epa)

class TrainingTemporalInvarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games,cls.epa=synthetic_history()
        _,cls.reference=evaluate(cls.games,cls.epa,first_test_season=2022)
    def test_future_results_and_epa_cannot_change_earlier_predictions(self):
        g=self.games.copy();e=self.epa.copy()
        future=g[(g.season==2022)&(g.week==6)].game_id.tolist()
        g.loc[g.game_id.isin(future),"home_score"]=120
        g.loc[g.game_id.isin(future),"away_score"]=0
        e.loc[e.game_id.isin(future),"offense_epa_per_play"]=25
        e.loc[e.game_id.isin(future),"offense_success_rate"]=1
        _,modified=evaluate(g,e,first_test_season=2022)
        cols=["game_id","baseline_prob","v3_prob","market_prob","home_win"]
        before=self.reference[self.reference.week<6][cols].sort_values("game_id").reset_index(drop=True)
        after=modified[modified.week<6][cols].sort_values("game_id").reset_index(drop=True)
        self.assertGreater(len(before),0)
        pd.testing.assert_frame_equal(before,after,check_exact=True)
    def test_same_week_future_results_cannot_change_earlier_game_predictions(self):
        g=self.games.copy();e=self.epa.copy()
        future=g[(g.season==2022)&(g.week==5)].game_id.tolist()
        g.loc[g.game_id.isin(future),"home_score"]=110
        g.loc[g.game_id.isin(future),"away_score"]=0
        e.loc[e.game_id.isin(future),"offense_epa_per_play"]=-30
        e.loc[e.game_id.isin(future),"offense_success_rate"]=0
        _,modified=evaluate(g,e,first_test_season=2022)
        cols=["game_id","baseline_prob","v3_prob"]
        a=self.reference[self.reference.week<=4][cols].sort_values("game_id").reset_index(drop=True)
        b=modified[modified.week<=4][cols].sort_values("game_id").reset_index(drop=True)
        self.assertGreater(len(a),0)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
