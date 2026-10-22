"""Combined Week 6 retrospective championship for frozen NFL Edge 1.0 engines."""
from __future__ import annotations

import pandas as pd

from player_backtest import backtest_fantasy_projections, fantasy_backtest_by_position, fantasy_backtest_summary
from player_engine import normalize_player_schema
from week6_backtest import run_week_retrospective, summarize_week

PLAYER_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"


def load_player_stats(start_season: int = 2025, end_season: int = 2026) -> pd.DataFrame:
    frames = []
    for season in range(int(start_season), int(end_season) + 1):
        try:
            frames.append(pd.read_csv(PLAYER_URL_TEMPLATE.format(season=season), low_memory=False))
        except Exception:
            continue
    return normalize_player_schema(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def run_week6_championship(simulations: int = 20000, seed: int = 42) -> dict:
    """Run frozen Game Engine + Player Engine against the completed 2026 Week 6 slate."""
    games = run_week_retrospective(2026, 6, simulations=simulations, seed=seed)
    game_summary = summarize_week(games)

    stats = load_player_stats(2025, 2026)
    players = backtest_fantasy_projections(
        stats,
        start_season=2026,
        end_season=2026,
        start_week=6,
        end_week=6,
        lookback=8,
        min_games=3,
    ) if not stats.empty else pd.DataFrame()
    player_summary = fantasy_backtest_summary(players)
    by_position = fantasy_backtest_by_position(players)

    # Diagnostic leaderboards make misses actionable instead of hiding them in averages.
    biggest_player_misses = pd.DataFrame()
    best_player_hits = pd.DataFrame()
    if not players.empty:
        cols = [c for c in ["player_name", "position", "team", "projected_ppr", "actual_ppr", "abs_error", "floor", "ceiling", "inside_interval"] if c in players.columns]
        biggest_player_misses = players.sort_values("abs_error", ascending=False)[cols].head(20).reset_index(drop=True)
        best_player_hits = players.sort_values("abs_error", ascending=True)[cols].head(20).reset_index(drop=True)

    biggest_game_misses = pd.DataFrame()
    if not games.empty:
        biggest_game_misses = games.assign(combined_error=games["margin_error"] + games["total_error"]).sort_values("combined_error", ascending=False).head(10).reset_index(drop=True)

    return {
        "game_results": games,
        "game_summary": game_summary,
        "player_results": players,
        "player_summary": player_summary,
        "player_by_position": by_position,
        "biggest_player_misses": biggest_player_misses,
        "best_player_hits": best_player_hits,
        "biggest_game_misses": biggest_game_misses,
    }


if __name__ == "__main__":
    championship = run_week6_championship()
    print("GAME SUMMARY")
    print(championship["game_summary"])
    print("\nPLAYER SUMMARY")
    print(championship["player_summary"])
    print("\nBY POSITION")
    print(championship["player_by_position"].to_string(index=False))
    print("\nGAME RESULTS")
    print(championship["game_results"].to_string(index=False))
    print("\nBIGGEST PLAYER MISSES")
    print(championship["biggest_player_misses"].to_string(index=False))
