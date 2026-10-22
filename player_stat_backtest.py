"""Leakage-safe backtest for NFL Edge Player Production Challenger."""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_engine import normalize_player_schema
from player_stat_engine import STAT_COLUMNS, project_player_stats


def backtest_player_stats(stats: pd.DataFrame, season: int, start_week: int, end_week: int, lookback: int = 8, min_games: int = 3) -> pd.DataFrame:
    d = normalize_player_schema(stats)
    id_col = next((c for c in ("player_id", "gsis_id", "player_display_name", "player_name") if c in d.columns), None)
    if id_col is None:
        return pd.DataFrame()
    rows = []
    for week in range(int(start_week), int(end_week) + 1):
        history = d[(d["season"] < season) | ((d["season"] == season) & (d["week"] < week))]
        actuals = d[(d["season"] == season) & (d["week"] == week)]
        histories = {pid: g for pid, g in history.groupby(id_col, dropna=True)}
        for _, actual in actuals.iterrows():
            pid = actual.get(id_col)
            h = histories.get(pid)
            if h is None or len(h) < min_games:
                continue
            p = project_player_stats(h, lookback=lookback)
            if not p:
                continue
            base = {"season": season, "week": week, "player_id": str(pid), "player_name": p["player_name"], "team": p["team"], "position": p["position"]}
            for stat in STAT_COLUMNS:
                if stat not in actual.index or pd.isna(actual.get(stat)):
                    continue
                projected = float(p["stats"][stat]["projection"])
                observed = float(actual.get(stat))
                rows.append({**base, "stat": stat, "projected": projected, "actual": observed, "error": projected - observed, "abs_error": abs(projected - observed), "floor": p["stats"][stat]["floor"], "ceiling": p["stats"][stat]["ceiling"], "covered": p["stats"][stat]["floor"] <= observed <= p["stats"][stat]["ceiling"]})
    return pd.DataFrame(rows)


def summarize_player_stats(results: pd.DataFrame) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame(columns=["stat", "n", "mae", "rmse", "bias", "correlation", "coverage"])
    rows = []
    for stat, g in results.groupby("stat"):
        corr = g["projected"].corr(g["actual"]) if len(g) > 1 else np.nan
        rows.append({"stat": stat, "n": int(len(g)), "mae": float(g["abs_error"].mean()), "rmse": float(np.sqrt(np.mean(g["error"] ** 2))), "bias": float(g["error"].mean()), "correlation": float(corr) if pd.notna(corr) else np.nan, "coverage": float(g["covered"].mean())})
    return pd.DataFrame(rows).sort_values("stat").reset_index(drop=True)
