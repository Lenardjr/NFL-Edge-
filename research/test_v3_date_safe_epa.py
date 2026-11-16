import unittest
import pandas as pd
from research.v3_date_safe_epa import attach_prior_epa

class V3LeakageTests(unittest.TestCase):
    def setUp(self):
        self.games=pd.DataFrame([
            dict(game_id="a",gameday="2025-09-01",season=2025,home_team="DET",away_team="GB"),
            dict(game_id="b",gameday="2025-09-08",season=2025,home_team="DET",away_team="CHI"),
            dict(game_id="c",gameday="2025-09-08",season=2025,home_team="GB",away_team="MIN"),
            dict(game_id="d",gameday="2025-09-15",season=2025,home_team="DET",away_team="GB"),
            dict(game_id="e",gameday="2026-09-01",season=2026,home_team="DET",away_team="GB")])
        self.epa=pd.DataFrame([
            dict(game_id="a",team="DET",offense_epa_per_play=.2,offense_success_rate=.6),
            dict(game_id="a",team="GB",offense_epa_per_play=-.1,offense_success_rate=.3),
            dict(game_id="b",team="DET",offense_epa_per_play=2,offense_success_rate=.9),
            dict(game_id="c",team="GB",offense_epa_per_play=3,offense_success_rate=.8)])
    def test_prior_only_and_season_reset(self):
        out=attach_prior_epa(self.games,self.epa).set_index("game_id")
        self.assertEqual(out.loc["a","home_epa_prior5"],0)
        self.assertAlmostEqual(out.loc["b","home_epa_prior5"],.2)
        self.assertAlmostEqual(out.loc["c","home_epa_prior5"],-.1)
        self.assertAlmostEqual(out.loc["d","home_epa_prior5"],1.1)
        self.assertEqual(out.loc["e","home_epa_prior5"],0)
    def test_same_day_changes_cannot_leak(self):
        original=attach_prior_epa(self.games,self.epa).set_index("game_id")
        modified=self.epa.copy()
        modified.loc[modified.game_id=="b","offense_epa_per_play"]=100
        other=attach_prior_epa(self.games,modified).set_index("game_id")
        self.assertEqual(original.loc["c","home_epa_prior5"],other.loc["c","home_epa_prior5"])
    def test_unknown_game_rejected(self):
        extra=self.epa.copy();extra.loc[0,"game_id"]="unknown"
        with self.assertRaises(ValueError):attach_prior_epa(self.games,extra)
if __name__=="__main__":unittest.main()
