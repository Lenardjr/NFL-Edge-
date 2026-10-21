"""Runtime compatibility shim for nflverse weekly player-stat schema changes.

This keeps the production nfl_edge.py file untouched while normalizing only the
nflverse stats_player weekly CSV feed. Remove this shim after the normalization
is folded directly into the player loader.
"""
from __future__ import annotations

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

if pd is not None:
    _nfl_edge_original_read_csv = pd.read_csv

    def _nfl_edge_read_csv(*args, **kwargs):
        frame = _nfl_edge_original_read_csv(*args, **kwargs)
        source = str(args[0] if args else kwargs.get("filepath_or_buffer", ""))
        if "stats_player_week_" not in source:
            return frame

        if "recent_team" not in frame.columns:
            for candidate in ("team", "team_abbr", "club_code"):
                if candidate in frame.columns:
                    frame = frame.copy()
                    frame["recent_team"] = frame[candidate]
                    break

        if "recent_team" not in frame.columns:
            frame = frame.copy()
            frame["recent_team"] = pd.NA

        aliases = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV"}
        frame["recent_team"] = frame["recent_team"].map(
            lambda value: aliases.get(str(value), str(value)) if pd.notna(value) else value
        )
        return frame

    pd.read_csv = _nfl_edge_read_csv
