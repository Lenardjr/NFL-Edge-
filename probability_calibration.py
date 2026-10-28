"""Leakage-safe calibration diagnostics for NFL Edge spread probabilities.

Fit calibration on 2023-2024 development only, then evaluate the frozen mapping
on 2025. The protected 2026 holdout is never loaded.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from game_betting_multiseason import load_history

DEV_SEASONS = (2023, 2024)
VALIDATION_SEASON = 2025
ALLOWED_SEASONS = DEV_SEASONS + (VALIDATION_SEASON,)
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _actual_home_cover(history: pd.DataFrame) -> pd.Series:
    delta = history["actual_margin"].astype(float) - history["market_spread"].astype(float)
    # Pushes have no binary calibration target and are excluded from scoring.
    out = pd.Series(np.nan, index=history.index, dtype=float)
    out.loc[delta > 0] = 1.0
    out.loc[delta < 0] = 0.0
    return out


def prepare_calibration_frame(history: pd.DataFrame) -> pd.DataFrame:
    required = {"season_eval", "home_cover_probability", "market_spread", "actual_margin"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"history missing required columns: {sorted(missing)}")
    frame = history.copy()
    frame["raw_probability"] = frame["home_cover_probability"].astype(float).clip(EPS, 1.0 - EPS)
    frame["target"] = _actual_home_cover(frame)
    return frame.dropna(subset=["target"]).copy()


def fit_logit_calibrator(development: pd.DataFrame) -> LogisticRegression:
    if development.empty:
        raise ValueError("development calibration sample is empty")
    x = _logit(development["raw_probability"].to_numpy()).reshape(-1, 1)
    y = development["target"].astype(int).to_numpy()
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(x, y)
    return model


def apply_calibrator(model: LogisticRegression, probabilities: pd.Series) -> np.ndarray:
    x = _logit(probabilities.to_numpy()).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def brier_score(probabilities: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(target)) ** 2))


def log_loss_score(probabilities: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(target, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def reliability_table(probabilities: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(target, dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    bucket = pd.cut(p, bins=edges, include_lowest=True, right=True)
    frame = pd.DataFrame({"probability": p, "target": y, "bucket": bucket})
    grouped = frame.groupby("bucket", observed=False)
    table = grouped.agg(n=("target", "size"), predicted=("probability", "mean"), realized=("target", "mean")).reset_index()
    table["calibration_error"] = table["realized"] - table["predicted"]
    return table


def score_sample(sample: pd.DataFrame, calibrated: np.ndarray) -> dict:
    raw = sample["raw_probability"].to_numpy()
    y = sample["target"].to_numpy()
    return {
        "games": int(len(sample)),
        "raw_brier": brier_score(raw, y),
        "calibrated_brier": brier_score(calibrated, y),
        "raw_log_loss": log_loss_score(raw, y),
        "calibrated_log_loss": log_loss_score(calibrated, y),
        "raw_reliability": reliability_table(raw, y),
        "calibrated_reliability": reliability_table(calibrated, y),
    }


def run_probability_calibration(simulations: int = 5000) -> dict:
    history = load_history(seasons=ALLOWED_SEASONS, simulations=simulations)
    if not history.empty and history.season_eval.max() > VALIDATION_SEASON:
        raise RuntimeError("Protected 2026 holdout entered probability calibration")
    frame = prepare_calibration_frame(history)
    development = frame[frame.season_eval.isin(DEV_SEASONS)].copy()
    validation = frame[frame.season_eval.eq(VALIDATION_SEASON)].copy()
    calibrator = fit_logit_calibrator(development)
    dev_calibrated = apply_calibrator(calibrator, development["raw_probability"])
    val_calibrated = apply_calibrator(calibrator, validation["raw_probability"])
    return {
        "method": "CALIBRATE_2023_2024__FREEZE__VALIDATE_2025__2026_UNTOUCHED",
        "intercept": float(calibrator.intercept_[0]),
        "slope": float(calibrator.coef_[0][0]),
        "development": score_sample(development, dev_calibrated),
        "validation": score_sample(validation, val_calibrated),
    }


if __name__ == "__main__":
    result = run_probability_calibration()
    print("NFL EDGE PROBABILITY CALIBRATION")
    print(result["method"])
    print(f"Frozen logit calibrator: intercept={result['intercept']:.6f} slope={result['slope']:.6f}")
    for label in ("development", "validation"):
        r = result[label]
        print(f"\n{label.upper()} ({r['games']} decided games)")
        print(f"Brier raw={r['raw_brier']:.6f} calibrated={r['calibrated_brier']:.6f}")
        print(f"LogLoss raw={r['raw_log_loss']:.6f} calibrated={r['calibrated_log_loss']:.6f}")
        print("RAW RELIABILITY")
        print(r["raw_reliability"].to_string(index=False))
        print("CALIBRATED RELIABILITY")
        print(r["calibrated_reliability"].to_string(index=False))
    print("\n2026 FINAL HOLDOUT: UNTOUCHED")
