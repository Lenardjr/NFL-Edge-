"""Week 6 championship for the experimental Player Production Engine.

Replays 2026 Week 6 with only pre-Week-6 player history, then reports accuracy
for each underlying box-score statistic. Research only; does not promote a model.
"""
from __future__ import annotations

import pandas as pd

from player_engine import normalize_player_schema
from player_stat_backtest import backtest_player_stats, summarize_player_stats

PLAYER_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
SEASON = 2026
WEEK = 6
LOOKBACK = 8
MIN_GAMES = 3


def load_stats(start_season: int = 2025, end_season: int = 2026) -> pd.DataFrame:
    frames = []
    for season in range(start_season, end_season + 1):
        frames.append(pd.read_csv(PLAYER_URL_TEMPLATE.format(season=season), low_memory=False))
    return normalize_player_schema(pd.concat(frames, ignore_index=True))


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = load_stats()
    results = backtest_player_stats(stats, season=SEASON, start_week=WEEK, end_week=WEEK, lookback=LOOKBACK, min_games=MIN_GAMES)
    summary = summarize_player_stats(results)
    if results.empty or summary.empty:
        raise RuntimeError("Player Production championship returned no results")
    if not bool((results["week"] == WEEK).all()):
        raise AssertionError("Championship contains rows outside target week")
    return results, summary


if __name__ == "__main__":
    results, summary = run()
    print("NFL EDGE PLAYER PRODUCTION CHALLENGER — 2026 WEEK 6")
    print("Validation mode: STRICT PRE-WEEK-6 PLAYER HISTORY")
    print(f"Stat observations: {len(results):,}")
    print(summary.to_string(index=False))
