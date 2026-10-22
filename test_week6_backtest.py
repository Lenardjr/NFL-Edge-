import numpy as np
import pandas as pd

from week6_backtest import _hide_target_week_and_future_scores, _preweek_mask


def _sample_games():
    return pd.DataFrame(
        {
            "season": [2025, 2026, 2026, 2026, 2027],
            "week": [18, 4, 5, 6, 1],
            "home_score": [21.0, 24.0, 27.0, 30.0, 33.0],
            "away_score": [17.0, 20.0, 23.0, 26.0, 29.0],
        }
    )


def test_preweek_mask_excludes_target_week_and_future():
    games = _sample_games()
    mask = _preweek_mask(games, 2026, 5)
    assert mask.tolist() == [True, True, False, False, False]


def test_hide_target_week_and_future_scores_preserves_only_preweek_results():
    games = _sample_games()
    replay = _hide_target_week_and_future_scores(games, 2026, 5)

    assert replay.loc[0, "home_score"] == 21.0
    assert replay.loc[0, "away_score"] == 17.0
    assert replay.loc[1, "home_score"] == 24.0
    assert replay.loc[1, "away_score"] == 20.0

    assert replay.loc[2:, ["home_score", "away_score"]].isna().all().all()


def test_score_hiding_does_not_mutate_source_frame():
    games = _sample_games()
    original = games.copy(deep=True)
    _ = _hide_target_week_and_future_scores(games, 2026, 5)
    pd.testing.assert_frame_equal(games, original)
