"""NFL Edge player projection validation helpers.

Pure helpers used to make the player pipeline resilient to nflverse schema changes
and to score projection quality without coupling validation logic to Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_ALIASES = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV"}
TEAM_COLUMNS = ("recent_team", "team", "team_abbr", "club_code")


def normalize_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of nflverse weekly player stats."""
    if df is None:
        return pd.DataFrame()
    d = df.copy()
    if "recent_team" not in d.columns:
        source = next((c for c in TEAM_COLUMNS[1:] if c in d.columns), None)
        d["recent_team"] = d[source] if source else pd.NA
    d["recent_team"] = d["recent_team"].map(
        lambda value: TEAM_ALIASES.get(str(value), str(value)) if pd.notna(value) else value
    )
    for c in ("season", "week"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def validate_player_feed(df: pd.DataFrame) -> dict:
    """Small health report for the weekly player feed."""
    d = normalize_player_stats(df)
    required = {"season", "week", "recent_team"}
    player_key = next((c for c in ("player_id", "gsis_id", "player_display_name", "player_name") if c in d.columns), None)
    missing = sorted(required.difference(d.columns))
    if player_key is None:
        missing.append("player identifier")
    seasons = sorted(pd.to_numeric(d.get("season", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    return {
        "ok": bool(not missing and not d.empty),
        "rows": int(len(d)),
        "missing": missing,
        "player_key": player_key,
        "seasons": seasons,
        "teams": int(d["recent_team"].dropna().nunique()) if "recent_team" in d else 0,
    }


def empirical_interval_errors(predicted, actual) -> dict:
    """Core deterministic error metrics shared by championship reporting."""
    p = pd.to_numeric(pd.Series(predicted), errors="coerce")
    a = pd.to_numeric(pd.Series(actual), errors="coerce")
    ok = p.notna() & a.notna()
    p, a = p[ok].to_numpy(float), a[ok].to_numpy(float)
    if not len(p):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan, "nmae": np.nan}
    err = p - a
    mae = float(np.mean(np.abs(err)))
    denom = float(np.mean(np.abs(a)))
    corr = float(np.corrcoef(p, a)[0, 1]) if len(p) > 1 and np.std(p) > 0 and np.std(a) > 0 else np.nan
    return {
        "n": int(len(p)),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "correlation": corr,
        "nmae": mae / denom if denom > 0 else np.nan,
    }


def reliability_tier(n: int, correlation: float, nmae: float) -> str:
    """Conservative projection-quality tier; never represents betting EV."""
    if np.isfinite(correlation) and np.isfinite(nmae):
        if n >= 250 and correlation >= 0.55 and nmae <= 0.55:
            return "STRONG"
        if n >= 150 and correlation >= 0.40 and nmae <= 0.75:
            return "USABLE"
    return "RESEARCH"
