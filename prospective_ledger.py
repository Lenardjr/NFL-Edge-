"""Prospective NFL Edge record ledger.

Reads frozen prediction/result companion artifacts and produces an auditable
running record. Prediction artifacts are never rewritten after kickoff.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

PROSPECTIVE_DIR = Path("prospective")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_predictions(root: Path = PROSPECTIVE_DIR) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*.json") if not p.name.endswith("_result.json"))


def result_path(prediction_path: Path) -> Path:
    return prediction_path.with_name(f"{prediction_path.stem}_result.json")


def ledger_rows(paths: Iterable[Path] | None = None) -> pd.DataFrame:
    prediction_paths = list(paths) if paths is not None else discover_predictions()
    rows: list[dict] = []
    for path in prediction_paths:
        p = _load(path)
        e10 = p["engine_1_0"]
        e11 = p["engine_1_1_b_frozen_challenger"]
        row = {
            "ledger_id": p["ledger_id"],
            "season": p["season"],
            "week": p["week"],
            "away_team": p["away_team"],
            "home_team": p["home_team"],
            "kickoff_et": p["kickoff_et"],
            "engine_1_0_margin": float(e10["projected_home_margin"]),
            "engine_1_1_b_margin": float(e11["projected_home_margin"]),
            "engine_1_0_pick": e10["official_pick"],
            "engine_1_1_b_pick": e11["winner_pick"],
            "market_home_margin": float(p["market_snapshot"]["home_margin"]),
            "status": "AWAITING_RESULT",
        }
        rpath = result_path(path)
        if rpath.exists():
            r = _load(rpath)
            away_score = float(r["actual_away_score"])
            home_score = float(r["actual_home_score"])
            actual_margin = home_score - away_score
            actual_winner = p["home_team"] if actual_margin > 0 else p["away_team"]
            row.update({
                "status": "FINAL",
                "actual_away_score": away_score,
                "actual_home_score": home_score,
                "actual_margin": actual_margin,
                "actual_winner": actual_winner,
                "engine_1_0_margin_error": abs(row["engine_1_0_margin"] - actual_margin),
                "engine_1_1_b_margin_error": abs(row["engine_1_1_b_margin"] - actual_margin),
                "engine_1_0_winner_correct": row["engine_1_0_pick"] == actual_winner,
                "engine_1_1_b_winner_correct": row["engine_1_1_b_pick"] == actual_winner,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {"registered_games": 0, "completed_games": 0}
    final = ledger[ledger["status"] == "FINAL"].copy()
    summary = {
        "registered_games": int(len(ledger)),
        "completed_games": int(len(final)),
        "awaiting_results": int((ledger["status"] != "FINAL").sum()),
    }
    if final.empty:
        return summary
    summary.update({
        "engine_1_0_winner_accuracy": float(final["engine_1_0_winner_correct"].mean()),
        "engine_1_1_b_winner_accuracy": float(final["engine_1_1_b_winner_correct"].mean()),
        "engine_1_0_margin_mae": float(final["engine_1_0_margin_error"].mean()),
        "engine_1_1_b_margin_mae": float(final["engine_1_1_b_margin_error"].mean()),
        "engine_1_1_b_margin_mae_delta": float(final["engine_1_1_b_margin_error"].mean() - final["engine_1_0_margin_error"].mean()),
    })
    return summary


if __name__ == "__main__":
    ledger = ledger_rows()
    print("PROSPECTIVE LEDGER")
    print(ledger.to_string(index=False) if not ledger.empty else "No registered games")
    print("\nSUMMARY")
    print(summarize(ledger))
