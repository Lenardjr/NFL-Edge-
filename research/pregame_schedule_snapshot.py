"""Conservative NFLverse pregame schedule snapshot.

NFLverse gametime is treated as US Eastern wall time. Future games with
unknown kickoff times are excluded, never assigned guessed timestamps.
Completed historical games may lack kickoff time and remain trainable.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def convert_schedule(raw: pd.DataFrame, *, captured_at_utc, asof_utc):
    captured = pd.Timestamp(captured_at_utc)
    asof = pd.Timestamp(asof_utc)
    if captured.tzinfo is None or asof.tzinfo is None:
        raise ValueError("UTC-aware timestamps required")
    captured = captured.tz_convert("UTC")
    asof = asof.tz_convert("UTC")
    if asof > captured:
        raise ValueError("As-of cannot be later than capture")
    if asof.date() != captured.date():
        raise ValueError("Cannot backdate a current schedule snapshot")

    required = {"game_id", "season", "week", "gameday", "gametime",
                "home_team", "away_team", "home_score", "away_score"}
    if required - set(raw):
        raise ValueError("Missing schedule columns: " + str(sorted(required - set(raw))))
    g = raw.copy()
    if g.game_id.isna().any() or g.game_id.duplicated().any():
        raise ValueError("Duplicate or missing game IDs")
    g["gameday"] = pd.to_datetime(g.gameday, errors="raise")
    g["home_score"] = pd.to_numeric(g.home_score, errors="coerce")
    g["away_score"] = pd.to_numeric(g.away_score, errors="coerce")
    if (g.home_score.notna() != g.away_score.notna()).any():
        raise ValueError("Incomplete score pair")

    def kickoff(row):
        if pd.isna(row.gametime) or not str(row.gametime).strip():
            return None
        local = pd.Timestamp(
            row.gameday.strftime("%Y-%m-%d") + " " + str(row.gametime)
        ).tz_localize("America/New_York", ambiguous="raise", nonexistent="raise")
        return local.tz_convert("UTC").isoformat()

    g["kickoff_utc"] = g.apply(kickoff, axis=1)
    # Without verified completion timestamps, do not use any same-day scores.
    g.loc[g.gameday.dt.date >= asof.date(), ["home_score", "away_score"]] = float("nan")
    # Preserve older scored games for training; do not predict TBD kickoffs.
    keep = g.kickoff_utc.notna() | (g.home_score.notna() & g.away_score.notna())
    return g.loc[keep].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-csv", required=True)
    parser.add_argument("--output-dir", default="research_outputs")
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    raw = pd.read_csv(args.schedule_csv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    games = convert_schedule(raw, captured_at_utc=now, asof_utc=now)
    games.to_csv(out / "nfl_pregame_schedule_snapshot.csv", index=False)
    (out / "nfl_pregame_schedule_snapshot_metadata.json").write_text(
        json.dumps({"captured_at_utc": now, "asof_utc": now,
                    "source_file": str(args.schedule_csv), "rows": len(games),
                    "source_publication_verified": False}, indent=2)
    )
    print(f"Prepared {len(games)} schedule rows")


if __name__ == "__main__":
    main()
