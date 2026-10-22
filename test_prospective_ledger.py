import json

from prospective_ledger import ledger_rows, summarize


def test_ledger_scores_companion_result_without_mutating_prediction(tmp_path):
    prediction = {
        "ledger_id": "TEST-001", "season": 2026, "week": 7,
        "away_team": "AWY", "home_team": "HME", "kickoff_et": "2026-10-22T20:15:00-04:00",
        "engine_1_0": {"projected_home_margin": 1.0, "official_pick": "HME"},
        "market_snapshot": {"home_margin": 3.0},
        "engine_1_1_b_frozen_challenger": {"projected_home_margin": 1.6, "winner_pick": "HME"},
    }
    p = tmp_path / "game.json"
    p.write_text(json.dumps(prediction))
    before = p.read_text()
    (tmp_path / "game_result.json").write_text(json.dumps({"actual_away_score": 20, "actual_home_score": 24}))
    ledger = ledger_rows([p])
    assert p.read_text() == before
    assert ledger.loc[0, "status"] == "FINAL"
    assert ledger.loc[0, "actual_margin"] == 4.0
    assert ledger.loc[0, "engine_1_0_margin_error"] == 3.0
    assert ledger.loc[0, "engine_1_1_b_margin_error"] == 2.4
    assert bool(ledger.loc[0, "engine_1_0_winner_correct"])


def test_summary_handles_pending_and_final(tmp_path):
    base = {
        "season": 2026, "week": 7, "away_team": "A", "home_team": "H",
        "kickoff_et": "2026-10-22T20:15:00-04:00",
        "engine_1_0": {"projected_home_margin": 1.0, "official_pick": "H"},
        "market_snapshot": {"home_margin": 2.0},
        "engine_1_1_b_frozen_challenger": {"projected_home_margin": 1.3, "winner_pick": "H"},
    }
    p1 = tmp_path / "one.json"
    p1.write_text(json.dumps({**base, "ledger_id": "ONE"}))
    p2 = tmp_path / "two.json"
    p2.write_text(json.dumps({**base, "ledger_id": "TWO"}))
    (tmp_path / "one_result.json").write_text(json.dumps({"actual_away_score": 17, "actual_home_score": 20}))
    summary = summarize(ledger_rows([p1, p2]))
    assert summary["registered_games"] == 2
    assert summary["completed_games"] == 1
    assert summary["awaiting_results"] == 1
    assert summary["engine_1_0_winner_accuracy"] == 1.0
