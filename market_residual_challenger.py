"""Development-only market-residual challenger for NFL Edge ATS.

The target is actual_margin - market_spread. Candidate selection uses blocked
2023/2024 cross-validation only. 2025 and 2026 are intentionally never loaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from game_betting_multiseason import load_history

DEV_SEASONS = (2023, 2024)
FEATURES = ("model_market_gap", "market_spread")
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def prepare_frame(history: pd.DataFrame) -> pd.DataFrame:
    required = {"season_eval", "projected_margin", "actual_margin", "market_spread"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"history missing required columns: {sorted(missing)}")
    frame = history.copy()
    frame["residual"] = frame["actual_margin"].astype(float) - frame["market_spread"].astype(float)
    frame["model_market_gap"] = frame["projected_margin"].astype(float) - frame["market_spread"].astype(float)
    frame["cover_target"] = np.where(frame["residual"] > 0, 1.0, np.where(frame["residual"] < 0, 0.0, np.nan))
    return frame.dropna(subset=["cover_target", *FEATURES]).copy()


def _fit(train: pd.DataFrame, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(train[list(FEATURES)], train["residual"])
    return model


def _score(test: pd.DataFrame, predicted_residual: np.ndarray) -> dict:
    actual = test["residual"].to_numpy(dtype=float)
    y = test["cover_target"].astype(int).to_numpy()
    pred = np.asarray(predicted_residual, dtype=float)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    auc = float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")
    side = pred >= 0
    correct = np.where(side, y, 1 - y)
    return {"games": int(len(test)), "rmse": rmse, "mae": mae, "auc": auc, "side_win_rate": float(np.mean(correct))}


def blocked_cv(frame: pd.DataFrame, alphas=ALPHAS) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        fold_scores = []
        for train_season, test_season in ((2023, 2024), (2024, 2023)):
            train = frame[frame.season_eval.eq(train_season)]
            test = frame[frame.season_eval.eq(test_season)]
            if train.empty or test.empty:
                raise ValueError("Both 2023 and 2024 are required for blocked CV")
            model = _fit(train, alpha)
            fold_scores.append(_score(test, model.predict(test[list(FEATURES)])))
        rows.append({
            "alpha": float(alpha),
            "games": sum(s["games"] for s in fold_scores),
            "rmse": float(np.mean([s["rmse"] for s in fold_scores])),
            "mae": float(np.mean([s["mae"] for s in fold_scores])),
            "auc": float(np.mean([s["auc"] for s in fold_scores])),
            "side_win_rate": float(np.mean([s["side_win_rate"] for s in fold_scores])),
        })
    return pd.DataFrame(rows).sort_values(["rmse", "mae", "alpha"]).reset_index(drop=True)


def run_challenger(simulations: int = 5000) -> dict:
    history = load_history(seasons=DEV_SEASONS, simulations=simulations)
    if not history.empty and history.season_eval.max() > max(DEV_SEASONS):
        raise RuntimeError("Non-development season entered market residual challenger")
    frame = prepare_frame(history)
    cv = blocked_cv(frame)
    winner = cv.iloc[0]
    frozen = _fit(frame, float(winner["alpha"]))
    return {
        "method": "2023_2024_BLOCKED_CV_RESIDUAL_CHALLENGER__2025_2026_UNTOUCHED",
        "features": FEATURES,
        "cv": cv,
        "selected_alpha": float(winner["alpha"]),
        "intercept": float(frozen.intercept_),
        "coefficients": dict(zip(FEATURES, map(float, frozen.coef_))),
    }


if __name__ == "__main__":
    result = run_challenger()
    print("NFL EDGE MARKET-RESIDUAL CHALLENGER")
    print(result["method"])
    print("\nBLOCKED 2023/2024 CROSS-VALIDATION")
    print(result["cv"].to_string(index=False))
    print(f"\nSELECTED ALPHA: {result['selected_alpha']:.6f}")
    print(f"INTERCEPT: {result['intercept']:.6f}")
    print("COEFFICIENTS")
    for name, value in result["coefficients"].items():
        print(f"{name}: {value:.6f}")
    print("\n2025 VALIDATION: UNTOUCHED")
    print("2026 FINAL HOLDOUT: UNTOUCHED")
