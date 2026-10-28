"""Development-only diagnostics for NFL Edge probability signal strength.

This module deliberately uses 2023-2024 only. It does not load 2025 or the
protected 2026 holdout. Its purpose is to understand whether the raw model
probabilities contain useful ranking signal even though their magnitudes are
overconfident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from game_betting_multiseason import load_history
from probability_calibration import prepare_calibration_frame

DEV_SEASONS = (2023, 2024)


def expected_calibration_error(probabilities: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(target, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(p, edges[1:-1], right=True)
    total = len(p)
    if total == 0:
        return float("nan")
    ece = 0.0
    for idx in range(bins):
        mask = bucket == idx
        if not np.any(mask):
            continue
        ece += (mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def discrimination_metrics(frame: pd.DataFrame) -> dict:
    y = frame["target"].astype(int).to_numpy()
    p = frame["raw_probability"].to_numpy()
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, p))
    return {
        "games": int(len(frame)),
        "home_cover_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "probability_std": float(np.std(p)),
        "auc": auc,
        "ece": expected_calibration_error(p, y),
    }


def decile_table(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["decile"] = pd.qcut(ranked["raw_probability"], q=10, labels=False, duplicates="drop")
    table = (
        ranked.groupby("decile", observed=True)
        .agg(
            games=("target", "size"),
            min_probability=("raw_probability", "min"),
            mean_probability=("raw_probability", "mean"),
            max_probability=("raw_probability", "max"),
            realized_home_cover=("target", "mean"),
        )
        .reset_index()
    )
    return table


def confidence_table(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["confidence"] = (ranked["raw_probability"] - 0.5).abs()
    ranked["model_side_correct"] = np.where(
        ranked["raw_probability"] >= 0.5,
        ranked["target"],
        1.0 - ranked["target"],
    )
    ranked["confidence_decile"] = pd.qcut(ranked["confidence"], q=10, labels=False, duplicates="drop")
    return (
        ranked.groupby("confidence_decile", observed=True)
        .agg(
            games=("target", "size"),
            mean_confidence=("confidence", "mean"),
            model_side_win_rate=("model_side_correct", "mean"),
        )
        .reset_index()
    )


def run_signal_diagnostics(simulations: int = 5000) -> dict:
    history = load_history(seasons=DEV_SEASONS, simulations=simulations)
    if not history.empty and history.season_eval.max() > max(DEV_SEASONS):
        raise RuntimeError("Non-development season entered signal diagnostics")
    frame = prepare_calibration_frame(history)
    return {
        "method": "2023_2024_SIGNAL_DIAGNOSTICS_ONLY__2025_2026_UNTOUCHED",
        "metrics": discrimination_metrics(frame),
        "probability_deciles": decile_table(frame),
        "confidence_deciles": confidence_table(frame),
    }


if __name__ == "__main__":
    result = run_signal_diagnostics()
    print("NFL EDGE CALIBRATION SIGNAL DIAGNOSTICS")
    print(result["method"])
    print("\nMETRICS")
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nRAW PROBABILITY DECILES")
    print(result["probability_deciles"].to_string(index=False))
    print("\nABSOLUTE-CONFIDENCE DECILES")
    print(result["confidence_deciles"].to_string(index=False))
    print("\n2025 VALIDATION: UNTOUCHED BY THIS DIAGNOSTIC")
    print("2026 FINAL HOLDOUT: UNTOUCHED")
