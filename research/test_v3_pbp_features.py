import unittest
import pandas as pd
from research.v3_pbp_features import aggregate_games

class V3PbpTests(unittest.TestCase):
    def test_aggregates_and_excludes_special_teams(self):
        data=pd.DataFrame([
          dict(game_id="g",season=2025,week=1,posteam="DET",defteam="GB",play_type="pass",epa=1.0),
          dict(game_id="g",season=2025,week=1,posteam="DET",defteam="GB",play_type="run",epa=-.5),
          dict(game_id="g",season=2025,week=1,posteam="DET",defteam="GB",play_type="punt",epa=20),
          dict(game_id="g",season=2025,week=1,posteam="GB",defteam="DET",play_type="pass",epa=0)])
        out=aggregate_games(data)
        det=out.set_index("team").loc["DET"]
        self.assertEqual(det.eligible_plays,2)
        self.assertAlmostEqual(det.offense_epa_per_play,.25)
        self.assertAlmostEqual(det.offense_success_rate,.5)
    def test_missing_column_rejected(self):
        with self.assertRaises(ValueError):aggregate_games(pd.DataFrame({"game_id":["g"]}))
if __name__=="__main__":unittest.main()
