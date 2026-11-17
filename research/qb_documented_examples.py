"""Documented historical pregame QB announcements (research evidence only).

These dated publications precede the game date, but do not constitute
verified archived capture timestamps or executable market snapshots.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

EVIDENCE=[
 dict(game_id="2024_18_KC_DEN",side="away",qb_name="Carson Wentz",
      replaced_qb="Patrick Mahomes",game_date="2025-01-05",
      published_date="2025-01-01",source="CBS Sports / RotoWire",
      source_url="https://www.cbssports.com/fantasy/football/news/chiefs-carson-wentz-set-to-start-sunday/",
      finding="Andy Reid confirmed Wentz starting; Mahomes resting"),
 dict(game_id="2023_18_PIT_BAL",side="home",qb_name="Tyler Huntley",
      replaced_qb="Lamar Jackson",game_date="2024-01-06",
      published_date="2024-01-03",source="Baltimore Ravens official",
      source_url="https://www.baltimoreravens.com/news/lamar-jackson-won-t-play-vs-steelers",
      finding="John Harbaugh announced Huntley starting; Jackson resting"),
 dict(game_id="2024_18_SEA_LA",side="home",qb_name="Jimmy Garoppolo",
      replaced_qb="Matthew Stafford",game_date="2025-01-05",
      published_date="2025-01-01",source="Los Angeles Rams official",
      source_url="https://www.therams.com/news/rams-jimmy-garoppolo-to-start-at-quarterback-week-18-seahawks",
      finding="Sean McVay announced Garoppolo starting; Stafford resting")
]

def validate(rows):
    df=pd.DataFrame(rows)
    if df.duplicated(["game_id","side"]).any():raise ValueError("Duplicate QB game-side")
    if (pd.to_datetime(df.published_date)>=pd.to_datetime(df.game_date)).any():
        raise ValueError("Publication date must precede game date")
    if not df.source_url.str.startswith("https://").all():raise ValueError("Invalid source URL")
    df["evidence_status"]="contemporary_pregame_publication_date_verified"
    df["archive_capture_verified"]=False
    df["eligible_for_model"]=False
    return df

def main():
    out=Path("research_outputs");out.mkdir(exist_ok=True)
    df=validate(EVIDENCE)
    df.to_csv(out/"nfl_qb_documented_pregame_examples.csv",index=False)
    summary={"documented_examples":len(df),"verified_publication_before_game_date":len(df),
      "archive_capture_verified":0,"model_eligible":0,
      "limitation":"Published dates are pregame, but capture timestamps, kickoff alignment and broad sample coverage are not verified."}
    (out/"nfl_qb_documented_pregame_examples_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
