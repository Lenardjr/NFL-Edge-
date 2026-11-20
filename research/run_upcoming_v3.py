"""Run frozen, pregame V3 research inference and optionally append to the ledger.

Input schedule MUST contain a real kickoff_utc field supplied by the snapshot
producer. Never infer UTC kickoff from a date or from gametime without timezone.
No network access, betting action, or production mutation occurs here.
"""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from research.upcoming_v3_inference import predict_upcoming,log_upcoming

def run(*,games_csv,epa_csv,asof_utc,output_dir,ledger=None,now_utc=None,minimum_train=300):
    now=now_utc or datetime.now(timezone.utc).isoformat()
    games=pd.read_csv(games_csv)
    epa=pd.read_csv(epa_csv)
    predictions=predict_upcoming(games,epa,asof_utc=asof_utc,now_utc=now,minimum_train=minimum_train)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    prediction_path=out/"nfl_upcoming_v3_research_predictions.csv"
    predictions.to_csv(prediction_path,index=False)
    recorded=log_upcoming(predictions,ledger,now_utc=now) if ledger else []
    summary={"predictions":len(predictions),"logged":len(recorded),
             "snapshot_asof_utc":asof_utc,"prediction_generated_at_utc":now,
             "source_authenticity_verified":False,
             "status":"research only; no validated edge, executable odds or betting"}
    (out/"nfl_upcoming_v3_research_summary.json").write_text(json.dumps(summary,indent=2))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--games-csv",required=True)
    p.add_argument("--epa-csv",required=True)
    p.add_argument("--asof-utc",required=True)
    p.add_argument("--output-dir",default="research_outputs")
    p.add_argument("--ledger",default=None,help="Opt-in; append predictions only if frozen inputs are authentic")
    p.add_argument("--minimum-train",type=int,default=300)
    a=p.parse_args()
    print(json.dumps(run(games_csv=a.games_csv,epa_csv=a.epa_csv,asof_utc=a.asof_utc,
        output_dir=a.output_dir,ledger=a.ledger,minimum_train=a.minimum_train),indent=2))
if __name__=="__main__":main()
