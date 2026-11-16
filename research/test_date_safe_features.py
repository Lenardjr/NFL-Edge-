"""Regression tests for no same-day leakage in research NFL features."""
import unittest
import pandas as pd
from research.date_safe_features import date_safe_features


def fixtures():
    return pd.DataFrame([
        dict(game_id="g1",season=2025,gameday="2025-09-01",home_team="A",away_team="B",home_score=21,away_score=14),
        dict(game_id="g2",season=2025,gameday="2025-09-01",home_team="A",away_team="C",home_score=40,away_score=0),
        dict(game_id="g3",season=2025,gameday="2025-09-08",home_team="A",away_team="B",home_score=17,away_score=10),
    ])


class DateSafeTests(unittest.TestCase):
    def test_same_day_features_equal_and_reorder_invariant(self):
        a = date_safe_features(fixtures()).set_index("game_id")
        b = date_safe_features(fixtures().iloc[::-1]).set_index("game_id")
        self.assertEqual(a.loc["g1","home_games_prior"], 0)
        self.assertEqual(a.loc["g2","home_games_prior"], 0)
        self.assertEqual(a.loc["g3","home_games_prior"], 2)
        for game_id in ("g1","g2","g3"):
            self.assertAlmostEqual(a.loc[game_id,"elo_diff"], b.loc[game_id,"elo_diff"])

    def test_same_day_result_cannot_change_pregame_features(self):
        g = fixtures()
        baseline = date_safe_features(g).set_index("game_id")
        g.loc[g.game_id=="g1","home_score"] = 99
        changed = date_safe_features(g).set_index("game_id")
        self.assertAlmostEqual(baseline.loc["g2","elo_diff"],changed.loc["g2","elo_diff"])
        self.assertAlmostEqual(baseline.loc["g2","home_pf5"],changed.loc["g2","home_pf5"])

    def test_unplayed_games_do_not_update_state(self):
        g=fixtures()
        g.loc[g.game_id=="g2",["home_score","away_score"]] = float("nan")
        out=date_safe_features(g).set_index("game_id")
        self.assertEqual(out.loc["g3","home_games_prior"],1)


if __name__ == "__main__":
    unittest.main()
