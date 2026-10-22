"""Walk-forward validation for NFL Edge player fantasy projections."""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_engine import fantasy_points_ppr, normalize_player_schema, project_all_players_as_of
from player_validation import empirical_interval_errors, reliability_tier


def backtest_fantasy_projections(
    stats: pd.DataFrame,
    start_season: int = 2022,
    end_season: int = 2026,
    start_week: int = 4,
    end_week: int = 18,
    lookback: int = 8,
    min_games: int = 3,
) -> pd.DataFrame:
    """Chronological player-level PPR backtest with no target-week leakage."""
    d = normalize_player_schema(stats)
    id_col = next((c for c in ("player_id", "gsis_id", "player_display_name", "player_name") if c in d.columns), None)
    if id_col is None or d.empty:
        return pd.DataFrame()
    d = d.copy()
    d["actual_ppr"] = fantasy_points_ppr(d)
    rows = []
    for season in range(int(start_season), int(end_season) + 1):
        season_rows = d[d["season"] == season]
        if season_rows.empty:
            continue
        available_weeks = sorted(pd.to_numeric(season_rows["week"], errors="coerce").dropna().astype(int).unique())
        for week in [w for w in available_weeks if start_week <= w <= end_week]:
            board = project_all_players_as_of(d, season, week, lookback=lookback, min_games=min_games)
            if board.empty:
                continue
            actual = season_rows[season_rows["week"] == week].copy()
            actual = actual[[id_col, "actual_ppr"]].dropna(subset=[id_col]).groupby(id_col, as_index=False)["actual_ppr"].sum()
            predictions = board[["player_id", "player_name", "position", "team", "projected_ppr", "floor", "ceiling", "confidence", "games_used"]].copy()
            predictions["player_id"] = predictions["player_id"].astype(str)
            actual[id_col] = actual[id_col].astype(str)
            merged = predictions.merge(actual, left_on="player_id", right_on=id_col, how="inner")
            if merged.empty:
                continue
            merged["season"] = season
            merged["week"] = week
            merged["error"] = merged["projected_ppr"] - merged["actual_ppr"]
            merged["abs_error"] = merged["error"].abs()
            merged["inside_interval"] = (merged["actual_ppr"] >= merged["floor"]) & (merged["actual_ppr"] <= merged["ceiling"])
            rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fantasy_backtest_summary(results: pd.DataFrame) -> dict:
    """Aggregate validation metrics and a conservative reliability label."""
    if results is None or results.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan, "nmae": np.nan, "interval_coverage": np.nan, "tier": "RESEARCH"}
    metrics = empirical_interval_errors(results["projected_ppr"], results["actual_ppr"])
    coverage = float(pd.to_numeric(results["inside_interval"], errors="coerce").mean()) if "inside_interval" in results else np.nan
    metrics["interval_coverage"] = coverage
    metrics["tier"] = reliability_tier(metrics["n"], metrics["correlation"], metrics["nmae"])
    return metrics


def fantasy_backtest_by_position(results: pd.DataFrame) -> pd.DataFrame:
    if results is None or results.empty or "position" not in results.columns:
        return pd.DataFrame()
    rows = []
    for position, group in results.groupby("position", dropna=True):
        summary = fantasy_backtest_summary(group)
        rows.append({"position": position, **summary})
    return pd.DataFrame(rows).sort_values(["mae", "n"], ascending=[True, False]).reset_index(drop=True) if rows else pd.DataFrame()
