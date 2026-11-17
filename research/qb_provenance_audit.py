"""Quarterback availability provenance gate (research-only).

Retrospective QB IDs, postgame participation and depth charts are not proof
of what was publicly known before kickoff. No unverified starter is accepted.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from research.market_validation import validate_snapshot

FIELDS=["game_id","side","qb_player_id","status","captured_at","kickoff_at",
        "source","source_url","verification"]
STATUSES={"confirmed_starter","expected_starter","out","questionable","doubtful"}

def validate_qb_evidence(evidence:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    missing=set(FIELDS)-set(evidence)
    if missing:raise ValueError("Missing QB evidence fields: "+str(sorted(missing)))
    e=evidence.copy()
    if e.duplicated(["game_id","side","captured_at","source_url"]).any():
        raise ValueError("Duplicate QB evidence rows")
    accepted=[]
    for _,r in e.iterrows():
        try:
            if r.side not in ("home","away"):raise ValueError("invalid side")
            if r.status not in STATUSES:raise ValueError("invalid status")
            for k in ("game_id","qb_player_id","source_url"):
                if pd.isna(r[k]) or not str(r[k]).strip():raise ValueError("missing "+k)
            if not str(r.source_url).startswith(("https://","http://")):
                raise ValueError("source URL required")
            validate_snapshot(captured_at=str(r.captured_at),kickoff_at=str(r.kickoff_at),
                              sportsbook="QB source audit",source=str(r.source))
            # Source verification is an explicit independent human/archival review,
            # never inferred from a historical game log or depth-chart snapshot.
            if r.verification!="verified_pregame":
                raise ValueError("pregame source not independently verified")
            accepted.append(True)
        except (ValueError,TypeError):
            accepted.append(False)
    e["eligible_for_pregame_research"]=accepted
    return e,{"rows":len(e),"eligible":sum(accepted),"excluded":len(e)-sum(accepted),
              "note":"Eligibility validates declared evidence metadata, not source authenticity; independent source review required."}

def build_queue(cases:pd.DataFrame)->pd.DataFrame:
    required={"game_id","gameday","home_team","away_team"}
    if required-set(cases):raise ValueError("Missing matchup fields")
    if cases.game_id.duplicated().any():raise ValueError("Duplicate game IDs")
    rows=[]
    for _,g in cases.iterrows():
        for side in ("home","away"):
            rows.append({"game_id":str(g.game_id),"gameday":str(g.gameday),
                "team":str(g[f"{side}_team"]),"side":side,
                "historical_qb_id":"","pregame_expected_qb_id":"","status":"",
                "captured_at":"","kickoff_at":"","source":"","source_url":"",
                "verification":"unreviewed","eligible_for_model":False})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cases",default="research_outputs/nfl_v5_extreme_cases_with_matchups.csv")
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    cases=pd.read_csv(a.cases)
    q=build_queue(cases)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    q.to_csv(out/"nfl_qb_pregame_verification_queue.csv",index=False)
    summary={"games":len(cases),"team_game_reviews":len(q),
             "verified_pregame":0,"unverified":len(q),
             "warning":"Do not train on retrospective starting-QB labels without pregame source verification."}
    (out/"nfl_qb_provenance_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
