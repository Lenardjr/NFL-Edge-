import unittest
import pandas as pd
from research.opponent_features import opponent_adjusted_features,V2_FEATURES

def game(date,home,away,hs,aws,season=2022):
    return dict(gameday=date,season=season,home_team=home,away_team=away,
                home_score=hs,away_score=aws)

class OpponentFeaturesTests(unittest.TestCase):
    def test_same_day_results_do_not_leak(self):
        games=pd.DataFrame([
            game("2022-09-01","A","B",21,14),
            game("2022-09-08","A","C",31,7),
            game("2022-09-08","B","D",10,3),
            game("2022-09-15","A","D",24,14)])
        first=opponent_adjusted_features(games)
        altered=games.copy()
        altered.loc[1,"home_score"]=90
        altered.loc[2,"home_score"]=80
        second=opponent_adjusted_features(altered)
        cols=["home_offense_adj","away_offense_adj","home_defense_adj","away_defense_adj","adj_net_diff"]
        for col in cols:
            self.assertEqual(first.loc[1,col],second.loc[1,col])
            self.assertEqual(first.loc[2,col],second.loc[2,col])
        self.assertTrue(set(V2_FEATURES).issubset(first.columns))

    def test_incomplete_game_does_not_update(self):
        games=pd.DataFrame([
            game("2022-09-01","A","B",None,None),
            game("2022-09-08","A","B",10,7)])
        f=opponent_adjusted_features(games)
        self.assertEqual(f.loc[1,"home_offense_adj"],0.0)

if __name__=="__main__":unittest.main()
