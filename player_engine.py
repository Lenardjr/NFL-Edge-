"""NFL Edge Player Projection Engine v1.

Builds transparent player projections from nflverse weekly data while tolerating
historical/new team-column schema changes. Designed to be imported by the app
and reused by future prop/fantasy backtests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd

TEAM_COLUMNS = ("recent_team", "team", "team_abbr", "club_code")


def normalize_player_schema(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "recent_team" not in d.columns:
        source = next((c for c in TEAM_COLUMNS[1:] if c in d.columns), None)
        d["recent_team"] = d[source] if source else pd.NA
    for c in ("season", "week"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _weighted_mean(values: Iterable[float], decay: float = 0.82) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return 0.0
    weights = decay ** np.arange(len(a) - 1, -1, -1)
    return float(np.average(a, weights=weights))


def _safe_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def fantasy_points_ppr(df: pd.DataFrame) -> pd.Series:
    return (
        _safe_series(df, "passing_yards") * 0.04
        + _safe_series(df, "passing_tds") * 4
        - _safe_series(df, "interceptions") * 2
        + _safe_series(df, "rushing_yards") * 0.10
        + _safe_series(df, "rushing_tds") * 6
        + _safe_series(df, "receptions")
        + _safe_series(df, "receiving_yards") * 0.10
        + _safe_series(df, "receiving_tds") * 6
        + _safe_series(df, "fumbles_lost") * -2
    )


@dataclass
class PlayerProjection:
    player_id: str
    player_name: str
    team: str | None
    games_used: int
    projected_ppr: float
    floor: float
    ceiling: float
    recent_avg: float
    volatility: float
    trend: float
    confidence: float

    def to_dict(self):
        return asdict(self)


def project_player(history: pd.DataFrame, lookback: int = 8) -> PlayerProjection | None:
    if history is None or history.empty:
        return None
    d = normalize_player_schema(history)
    sort_cols = [c for c in ("season", "week") if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols)
    d = d.tail(max(3, lookback)).copy()
    d["ppr"] = fantasy_points_ppr(d)
    points = d["ppr"].astype(float).to_numpy()
    projected = _weighted_mean(points)
    recent_avg = float(np.mean(points[-3:])) if len(points) else 0.0
    volatility = float(np.std(points, ddof=0)) if len(points) > 1 else 0.0
    trend = float(recent_avg - np.mean(points[:-3])) if len(points) > 3 else 0.0
    # Conservative empirical interval; Monte Carlo layer can replace this later.
    floor = max(0.0, projected - 0.85 * volatility)
    ceiling = max(floor, projected + 1.15 * volatility)
    sample_score = min(1.0, len(points) / max(lookback, 1))
    stability = 1.0 / (1.0 + volatility / max(projected, 4.0))
    confidence = float(np.clip(0.55 * sample_score + 0.45 * stability, 0.0, 1.0))
    row = d.iloc[-1]
    name = row.get("player_display_name", row.get("player_name", row.get("name", "Unknown")))
    pid = row.get("player_id", row.get("gsis_id", str(name)))
    team = row.get("recent_team")
    return PlayerProjection(
        player_id=str(pid), player_name=str(name),
        team=None if pd.isna(team) else str(team), games_used=len(points),
        projected_ppr=round(projected, 2), floor=round(floor, 2),
        ceiling=round(ceiling, 2), recent_avg=round(recent_avg, 2),
        volatility=round(volatility, 2), trend=round(trend, 2),
        confidence=round(confidence, 3),
    )


def project_all_players(stats: pd.DataFrame, lookback: int = 8, min_games: int = 3) -> pd.DataFrame:
    d = normalize_player_schema(stats)
    id_col = next((c for c in ("player_id", "gsis_id", "player_display_name", "player_name") if c in d.columns), None)
    if id_col is None:
        return pd.DataFrame()
    rows = []
    for _, g in d.groupby(id_col, dropna=True):
        if len(g) < min_games:
            continue
        p = project_player(g, lookback=lookback)
        if p:
            rows.append(p.to_dict())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["projected_ppr", "confidence"], ascending=False).reset_index(drop=True)
