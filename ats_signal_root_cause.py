"""Root-cause diagnostics for NFL Edge ATS signal.

Development-only: 2023-2024. This intentionally does not load 2025 or 2026.
It separates margin-model ranking from Monte Carlo cover-probability ranking and
checks whether residual dispersion is masking or exaggerating ATS signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from game_betting_multiseason import load_history

DEV_SEASONS = (2023, 2024)


def _auc(target: pd.Series, score: pd.Series) -> float:
    mask = target.notna() & score.notna()
    y = target.loc[mask].astype(int).to_numpy()
    s = score.loc[mask].astype(float).to_numpy()
    return float(roc_auc_score(y, s)) if len(y) and len(np.unique(y)) == 2 else float("nan")


def prepare_frame(history: pd.DataFrame) -> pd.DataFrame:
    required = {"season_eval", "projected_margin", "actual_margin", "market_spread", "home_cover_probability"}
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"history missing required columns: {sorted(missing)}")
    frame = history.copy()
    frame["ats_delta"] = frame["actual_margin"].astype(float) - frame["market_spread"].astype(float)
    frame["target"] = np.where(frame["ats_delta"] > 0, 1.0, np.where(frame["ats_delta"] < 0, 0.0, np.nan))
    frame["model_market_gap"] = frame["projected_margin"].astype(float) - frame["market_spread"].astype(float)
    frame["abs_gap"] = frame["model_market_gap"].abs()
    frame["probability_edge"] = frame["home_cover_probability"].astype(float) - 0.5
    return frame.dropna(subset=["target"]).copy()


def gap_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["gap_decile"] = pd.qcut(d["model_market_gap"], q=10, labels=False, duplicates="drop")
    return d.groupby("gap_decile", observed=True).agg(
        games=("target", "size"),
        mean_gap=("model_market_gap", "mean"),
        mean_cover_probability=("home_cover_probability", "mean"),
        realized_home_cover=("target", "mean"),
    ).reset_index()


def confidence_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["model_side_correct"] = np.where(d["model_market_gap"] >= 0, d["target"], 1.0 - d["target"])
    d["confidence_decile"] = pd.qcut(d["abs_gap"], q=10, labels=False, duplicates="drop")
    return d.groupby("confidence_decile", observed=True).agg(
        games=("target", "size"),
        mean_abs_gap=("abs_gap", "mean"),
        model_side_win_rate=("model_side_correct", "mean"),
    ).reset_index()


def diagnostics(frame: pd.DataFrame) -> dict:
    margin_error = frame["actual_margin"].astype(float) - frame["projected_margin"].astype(float)
    return {
        "games": int(len(frame)),
        "home_cover_rate": float(frame["target"].mean()),
        "margin_vs_market_auc": _auc(frame["target"], frame["model_market_gap"]),
        "cover_probability_auc": _auc(frame["target"], frame["home_cover_probability"]),
        "gap_probability_correlation": float(frame[["model_market_gap", "probability_edge"]].corr().iloc[0, 1]),
        "projected_margin_mae": float(np.mean(np.abs(margin_error))),
        "projected_margin_rmse": float(np.sqrt(np.mean(margin_error ** 2))),
        "margin_residual_std": float(np.std(margin_error)),
        "mean_abs_model_market_gap": float(frame["abs_gap"].mean()),
    }


def run_root_cause(simulations: int = 5000) -> dict:
    history = load_history(seasons=DEV_SEASONS, simulations=simulations)
    if not history.empty and history.season_eval.max() > max(DEV_SEASONS):
        raise RuntimeError("Non-development season entered ATS root-cause diagnostics")
    frame = prepare_frame(history)
    return {
        "method": "2023_2024_ATS_ROOT_CAUSE_ONLY__2025_2026_UNTOUCHED",
        "metrics": diagnostics(frame),
        "gap_deciles": gap_deciles(frame),
        "confidence_deciles": confidence_deciles(frame),
    }


if __name__ == "__main__":
    result = run_root_cause()
    print("NFL EDGE ATS SIGNAL ROOT-CAUSE DIAGNOSTICS")
    print(result["method"])
    print("\nMETRICS")
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nMODEL-MARKET GAP DECILES")
    print(result["gap_deciles"].to_string(index=False))
    print("\nABSOLUTE GAP CONFIDENCE DECILES")
    print(result["confidence_deciles"].to_string(index=False))
    print("\n2025 VALIDATION: UNTOUCHED")
    print("2026 FINAL HOLDOUT: UNTOUCHED")
