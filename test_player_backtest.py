import pandas as pd

from player_backtest import backtest_fantasy_projections, fantasy_backtest_summary, fantasy_backtest_by_position


def sample_stats():
    rows=[]
    for season in (2025, 2026):
        for week in range(1, 7):
            for pid, name, pos, base in [("p1","Receiver One","WR",10),("p2","Runner Two","RB",8)]:
                rows.append({
                    "season":season,"week":week,"player_id":pid,"player_display_name":name,
                    "position":pos,"team_abbr":"DET","receptions":(base+week)/5 if pos=="WR" else 1,
                    "receiving_yards":(base+week)*4 if pos=="WR" else 5,
                    "receiving_tds":0,"rushing_yards":(base+week)*4 if pos=="RB" else 0,
                    "rushing_tds":0,"passing_yards":0,"passing_tds":0,"interceptions":0,"fumbles_lost":0,
                })
    return pd.DataFrame(rows)


def test_walk_forward_backtest_produces_target_week_rows():
    results=backtest_fantasy_projections(sample_stats(),2026,2026,start_week=4,end_week=6,lookback=6,min_games=3)
    assert not results.empty
    assert set(results.week)=={4,5,6}
    assert set(results.player_name)=={"Receiver One","Runner Two"}
    assert (results.games_used <= 6).all()


def test_summary_and_position_tables_are_finite():
    results=backtest_fantasy_projections(sample_stats(),2026,2026,start_week=4,end_week=6,lookback=6,min_games=3)
    summary=fantasy_backtest_summary(results)
    assert summary["n"]==len(results)
    assert summary["mae"] >= 0
    assert 0 <= summary["interval_coverage"] <= 1
    assert summary["tier"] in {"STRONG","USABLE","RESEARCH"}
    by_pos=fantasy_backtest_by_position(results)
    assert set(by_pos.position)=={"RB","WR"}
