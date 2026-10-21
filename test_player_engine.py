import pandas as pd

from player_engine import normalize_player_schema, fantasy_points_ppr, project_player, project_all_players


def sample_stats(team_column="team"):
    rows=[]
    for week,pts in [(1,10),(2,12),(3,14),(4,18),(5,20),(6,22)]:
        row={
            "season":2026,"week":week,"player_id":"p1","player_display_name":"Test Receiver",
            "position":"WR",team_column:"DET","receptions":pts/4,"receiving_yards":pts*5,
            "receiving_tds":0,"rushing_yards":0,"rushing_tds":0,"passing_yards":0,
            "passing_tds":0,"interceptions":0,"fumbles_lost":0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_schema_normalizes_team_alias_column():
    d=normalize_player_schema(sample_stats("team_abbr"))
    assert "recent_team" in d.columns
    assert set(d.recent_team)=={"DET"}


def test_projection_has_ordered_range_and_confidence():
    p=project_player(sample_stats())
    assert p is not None
    assert p.floor <= p.projected_ppr <= p.ceiling
    assert 0 <= p.confidence <= 1
    assert p.trend > 0


def test_projection_board_returns_player():
    board=project_all_players(sample_stats(),lookback=6,min_games=3)
    assert len(board)==1
    assert board.iloc[0].player_name=="Test Receiver"
    assert board.iloc[0].team=="DET"


def test_ppr_scoring_math():
    d=pd.DataFrame([{"receptions":5,"receiving_yards":100,"receiving_tds":1}])
    assert fantasy_points_ppr(d).iloc[0]==21.0
