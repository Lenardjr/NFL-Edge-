import numpy as np
import pandas as pd

from market_residual_challenger import blocked_cv, prepare_frame


def sample_history():
    rows = []
    for season in (2023, 2024):
        for i in range(20):
            spread = float((i % 7) - 3)
            gap = float((i - 10) / 3)
            projected = spread + gap
            residual = 0.5 * gap + (1.0 if i % 2 == 0 else -1.0)
            actual = spread + residual
            rows.append({
                "season_eval": season,
                "projected_margin": projected,
                "actual_margin": actual,
                "market_spread": spread,
            })
    return pd.DataFrame(rows)


def test_prepare_frame_builds_residual_target_and_gap():
    frame = prepare_frame(sample_history())
    assert {"residual", "model_market_gap", "cover_target"}.issubset(frame.columns)
    assert np.isclose(frame.iloc[0].model_market_gap, frame.iloc[0].projected_margin - frame.iloc[0].market_spread)


def test_blocked_cv_uses_both_development_seasons():
    cv = blocked_cv(prepare_frame(sample_history()), alphas=(0.1, 1.0))
    assert len(cv) == 2
    assert (cv.games == 40).all()
    assert cv.auc.between(0, 1).all()


def test_prepare_frame_excludes_pushes():
    history = sample_history()
    history.loc[0, "actual_margin"] = history.loc[0, "market_spread"]
    frame = prepare_frame(history)
    assert len(frame) == len(history) - 1
