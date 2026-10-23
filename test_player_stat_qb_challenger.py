import pandas as pd

from player_stat_qb_challenger import project_qb_stats


def sample_qb():
    rows=[]
    for week,att,comp,yards,td,ints in [(1,30,19,210,1,1),(2,32,21,230,2,0),(3,34,23,250,2,1),(4,36,24,270,2,1),(5,38,26,290,3,0),(6,50,40,500,5,0)]:
        rows.append({"season":2026,"week":week,"player_id":"qb1","player_display_name":"Test QB","recent_team":"TST","position":"QB","attempts":att,"completions":comp,"passing_yards":yards,"passing_tds":td,"interceptions":ints})
    return pd.DataFrame(rows)


def test_qb_challenger_uses_only_supplied_history():
    d=sample_qb()
    p=project_qb_stats(d[d.week < 6],lookback=5)
    assert p["model_version"]=="qb-volume-efficiency-v0.2"
    assert p["stats"]["attempts"]["projection"] < 38
    assert p["stats"]["passing_yards"]["projection"] < 290


def test_qb_challenger_components_are_coherent():
    p=project_qb_stats(sample_qb().query("week < 6"),lookback=5)
    att=p["stats"]["attempts"]["projection"]
    comp=p["stats"]["completions"]["projection"]
    yards=p["stats"]["passing_yards"]["projection"]
    assert 0 <= comp <= att
    assert yards >= comp


def test_non_qb_is_rejected():
    d=sample_qb();d["position"]="RB"
    assert project_qb_stats(d) is None
