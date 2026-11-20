import unittest
import pandas as pd
from research.run_nfl_v3 import evaluate

class V3RunnerTests(unittest.TestCase):
    def test_insufficient_history_rejected(self):
        games=pd.DataFrame([dict(game_id="2025_01_GB_DET",gameday="2025-09-01",
            season=2025,week=1,home_team="DET",away_team="GB",
            home_score=20,away_score=17,home_moneyline=-110,away_moneyline=100)])
        epa=pd.DataFrame(columns=["game_id","team","offense_epa_per_play","offense_success_rate"])
        with self.assertRaises(ValueError):
            evaluate(games,epa)
if __name__=="__main__":unittest.main()
