"""Research-only audit of candidate historical data sources.

This records documented availability, not purchased access, observed coverage,
source authentication, or feature eligibility.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

SOURCES=[
 {"id":"the_odds_api","domain":"market","url":"https://the-odds-api.com/liveapi/guides/v4/",
  "historical_start":"2020-06-06","cadence":"5-minute snapshots from September 2022",
  "access":"paid historical endpoint","time_evidence":"snapshot timestamp supplied by provider",
  "limitations":"Need paid access, per-book event matching, source/quote audit and executable-price assumptions",
  "available_in_project":False,"pregame_verified":False},
 {"id":"nflverse_injuries","domain":"injury","url":"https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html",
  "historical_start":"2009","cadence":"reported daily update at 07:00 UTC during season",
  "access":"public historical release","time_evidence":"update cadence alone is not a row-level archival capture timestamp",
  "limitations":"Historical injury fields must be inspected; cannot infer a specific pregame snapshot from current file",
  "available_in_project":False,"pregame_verified":False},
 {"id":"nflverse_depth_charts","domain":"quarterback","url":"https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html",
  "historical_start":"not asserted","cadence":"reported daily update; from 2025 appended timestamped updates",
  "access":"public release","time_evidence":"timestamped updates described for 2025 onward, not independently verified here",
  "limitations":"Week labels or current releases do not establish exact pregame availability in earlier seasons",
  "available_in_project":False,"pregame_verified":False}
]
def audit(sources):
    ids=[s["id"] for s in sources]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate source IDs")
    for s in sources:
        if not str(s.get("url","")).startswith("https://"):raise ValueError("HTTPS source URL required")
        if s.get("pregame_verified") and not s.get("available_in_project"):
            raise ValueError("Cannot verify data absent from project")
    return {"sources":sources,"candidate_sources":len(sources),
        "integrated_sources":sum(bool(s["available_in_project"]) for s in sources),
        "verified_pregame_sources":sum(bool(s["pregame_verified"]) for s in sources),
        "next_steps":["Confirm provider coverage and licensing before purchasing",
          "Inspect raw row schemas and timestamps on a small historical sample",
          "Join to NFL game IDs and verify team, kickoff and price orientation",
          "Freeze pregame snapshots before any model training or ROI analysis"],
        "decision":"Source discovery only; no verified new model features or executable prices"}
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="research_outputs/nfl_data_source_readiness.json")
    a=p.parse_args();report=audit(SOURCES)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
