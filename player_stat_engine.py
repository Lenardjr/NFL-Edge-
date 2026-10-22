"""NFL Edge Player Production Challenger v0.1.

Leakage-safe experimental layer for predicting the underlying box-score stats
that drive fantasy points and future player props. This does not replace the
frozen Player Engine 1.0 champion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from player_engine import normalize_player_schema, player_history_as_of, _weighted_mean

STAT_COLUMNS = (
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
    "receiving_yards", "receiving_tds",
)


def _project_stat(history: pd.DataFrame, stat: str, lookback: int = 8) -> dict:
    if stat not in history.columns:
        return {"projection": 0.0, "floor": 0.0, "ceiling": 0.0, "games": 0}
    values = pd.to_numeric(history[stat], errors="coerce").dropna().tail(max(3, lookback)).to_numpy(dtype=float)
    if not len(values):
        return {"projection": 0.0, "floor": 0.0, "ceiling": 0.0, "games": 0}
    projection = _weighted_mean(values)
    volatility = float(np.std(values, ddof=0)) if len(values) > 1 else 0.0
    floor = max(0.0, projection - 0.85 * volatility)
    ceiling = max(floor, projection + 1.15 * volatility)
    return {"projection": round(projection, 2), "floor": round(floor, 2), "ceiling": round(ceiling, 2), "games": int(len(values))}


def project_player_stats(history: pd.DataFrame, lookback: int = 8) -> dict | None:
    if history is None or history.empty:
        return None
    d = normalize_player_schema(history)
    sort_cols = [c for c in ("season", "week") if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols)
    recent = d.tail(max(3, lookback))
    row = recent.iloc[-1]
    name = row.get("player_display_name", row.get("player_name", row.get("name", "Unknown")))
    pid = row.get("player_id", row.get("gsis_id", str(name)))
    return {
        "player_id": str(pid),
        "player_name": str(name),
        "team": None if pd.isna(row.get("recent_team")) else str(row.get("recent_team")),
        "position": None if pd.isna(row.get("position")) else str(row.get("position")),
        "stats": {stat: _project_stat(recent, stat, lookback=lookback) for stat in STAT_COLUMNS},
    }


def project_all_player_stats_as_of(stats: pd.DataFrame, season: int, week: int, lookback: int = 8, min_games: int = 3) -> list[dict]:
    history = player_history_as_of(stats, season=season, week=week)
    id_col = next((c for c in ("player_id", "gsis_id", "player_display_name", "player_name") if c in history.columns), None)
    if id_col is None:
        return []
    rows = []
    for _, group in history.groupby(id_col, dropna=True):
        if len(group) < min_games:
            continue
        projected = project_player_stats(group, lookback=lookback)
        if projected:
            rows.append(projected)
    return rows


def flatten_projections(projections: list[dict]) -> pd.DataFrame:
    rows = []
    for p in projections:
        row = {k: p.get(k) for k in ("player_id", "player_name", "team", "position")}
        for stat, values in p["stats"].items():
            row[f"{stat}_proj"] = values["projection"]
            row[f"{stat}_floor"] = values["floor"]
            row[f"{stat}_ceiling"] = values["ceiling"]
        rows.append(row)
    return pd.DataFrame(rows)
