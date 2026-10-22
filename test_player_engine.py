import pandas as pd

from player_engine import (
    QB_MODEL_VERSION,
    normalize_player_schema,
    fantasy_points_ppr,
    project_player,
    project_all_players,
    player_history_as_of,
    project_all_players_as_of,
)


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


def sample_qb_stats():
    rows=[]
    games=[
        (1,205,1,1,12,0,0),(2,220,1,0,18,0,0),(3,235,1,1,15,0,0),
        (4,245,2,0,22,0,0),(5,260,2,1,28,0,0),(6,275,5,0,35,1,0),
    ]
    for week,py,ptd,ints,ry,rtd,fl in games:
        rows.append({
            "season":2026,"week":week,"player_id":"qb1","player_display_name":"Test Quarterback",
            "position":"QB","team":"DET","passing_yards":py,"passing_tds":ptd,
            "interceptions":ints,"rushing_yards":ry,"rushing_tds":rtd,"fumbles_lost":fl,
            "receptions":0,"receiving_yards":0,"receiving_tds":0,
        })
    return pd.DataFrame(rows)


def weighted_baseline(points, decay=0.82):
    weights=[decay ** i for i in range(len(points)-1,-1,-1)]
    return sum(p*w for p,w in zip(points,weights))/sum(weights)


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
    assert p.position == "WR"


def test_projection_board_returns_player():
    board=project_all_players(sample_stats(),lookback=6,min_games=3)
    assert len(board)==1
    assert board.iloc[0].player_name=="Test Receiver"
    assert board.iloc[0].team=="DET"
    assert board.iloc[0].position=="WR"


def test_ppr_scoring_math():
    d=pd.DataFrame([{"receptions":5,"receiving_yards":100,"receiving_tds":1}])
    assert fantasy_points_ppr(d).iloc[0]==21.0


def test_as_of_history_excludes_target_and_future_weeks():
    d=sample_stats()
    history=player_history_as_of(d,2026,4)
    assert history.week.tolist()==[1,2,3]


def test_as_of_history_keeps_prior_season_for_early_weeks():
    prior=sample_stats().assign(season=2025)
    current=sample_stats().iloc[:2].copy()
    all_stats=pd.concat([prior,current],ignore_index=True)
    history=player_history_as_of(all_stats,2026,2)
    assert len(history)==7
    assert set(history.season)=={2025,2026}
    assert history[history.season==2026].week.tolist()==[1]


def test_as_of_board_is_leakage_safe():
    d=sample_stats()
    board=project_all_players_as_of(d,2026,5,lookback=8,min_games=3)
    assert len(board)==1
    assert int(board.iloc[0].games_used)==4


def test_qb_champion_has_explicit_model_version():
    assert QB_MODEL_VERSION == "qb-legacy-champion-v1"


def test_qb_champion_matches_frozen_weighted_baseline():
    d=sample_qb_stats()
    p=project_player(d,lookback=6)
    legacy=weighted_baseline(fantasy_points_ppr(d).tolist())
    assert p is not None
    assert p.position=="QB"
    assert abs(p.projected_ppr-legacy) < 0.01


def test_qb_champion_shrinks_recent_touchdown_spike():
    d=sample_qb_stats()
    p=project_player(d,lookback=6)
    recent_game=float(fantasy_points_ppr(d.tail(1)).iloc[0])
    assert p is not None
    assert p.projected_ppr < recent_game
    assert p.floor <= p.projected_ppr <= p.ceiling


def test_qb_as_of_projection_excludes_target_week_spike():
    d=sample_qb_stats()
    board=project_all_players_as_of(d,2026,6,lookback=6,min_games=3)
    direct=project_player(d[d.week < 6],lookback=6)
    assert len(board)==1
    assert direct is not None
    assert board.iloc[0].position=="QB"
    assert board.iloc[0].projected_ppr==direct.projected_ppr
    assert int(board.iloc[0].games_used)==5
