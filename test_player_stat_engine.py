import pandas as pd

from player_stat_engine import project_player_stats
from player_stat_backtest import backtest_player_stats, summarize_player_stats


def sample_stats():
    rows=[]
    for week, yards, attempts in [(1,200,30),(2,220,32),(3,240,34),(4,260,36),(5,280,38),(6,300,40)]:
        rows.append({"season":2026,"week":week,"player_id":"qb1","player_display_name":"Test QB","recent_team":"TST","position":"QB","passing_yards":yards,"attempts":attempts,"passing_tds":2,"interceptions":1})
    return pd.DataFrame(rows)


def test_projection_uses_only_supplied_history():
    d=sample_stats()
    p=project_player_stats(d[d.week < 6],lookback=5)
    assert p["stats"]["passing_yards"]["projection"] < 280
    assert p["stats"]["passing_yards"]["games"] == 5


def test_backtest_is_strictly_pre_target_week():
    d=sample_stats()
    results=backtest_player_stats(d,season=2026,start_week=6,end_week=6,lookback=5,min_games=3)
    yards=results[results.stat=="passing_yards"].iloc[0]
    assert yards.actual == 300
    assert yards.projected < 280


def test_stat_summary_reports_accuracy_metrics():
    results=backtest_player_stats(sample_stats(),season=2026,start_week=6,end_week=6,lookback=5,min_games=3)
    summary=summarize_player_stats(results)
    yards=summary[summary.stat=="passing_yards"].iloc[0]
    assert yards.n == 1
    assert yards.mae > 0
    assert 0 <= yards.coverage <= 1
