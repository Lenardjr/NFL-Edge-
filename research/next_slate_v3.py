"""Create a reviewable near-term slate from a previously captured V3 prediction CSV.

Never re-label an old snapshot as newly generated. This command does not write
to the prospective ledger, place wagers, or verify sportsbook prices.
"""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

REQUIRED={"game_id","kickoff_utc","home_team","away_team","home_win_probability",
          "input_fingerprint","model_version","features_asof_utc"}

def freeze_slate(predictions,*,now_utc,horizon_days=7):
    now=pd.Timestamp(now_utc)
    if now.tzinfo is None:raise ValueError("Timezone-aware now_utc required")
    now=now.tz_convert("UTC")
    if not 0<horizon_days<=14:raise ValueError("Horizon must be 1-14 days")
    if REQUIRED-set(predictions):raise ValueError("Missing prediction fields")
    p=predictions.copy()
    if p.game_id.isna().any() or p.game_id.duplicated().any():
        raise ValueError("Duplicate or missing game IDs")
    p["kickoff_utc"]=pd.to_datetime(p.kickoff_utc,utc=True,errors="raise")
    p["features_asof_utc"]=pd.to_datetime(p.features_asof_utc,utc=True,errors="raise")
    p["home_win_probability"]=pd.to_numeric(p.home_win_probability,errors="raise")
    if not p.home_win_probability.between(0,1,inclusive="neither").all():
        raise ValueError("Invalid probabilities")
    if (p.features_asof_utc>=p.kickoff_utc).any():
        raise ValueError("Prediction snapshot on or after kickoff")
    if (p.features_asof_utc>now).any():
        raise ValueError("Prediction snapshot dated in future")
    end=now+pd.Timedelta(days=horizon_days)
    slate=p.loc[(p.kickoff_utc>now)&(p.kickoff_utc<=end)].copy()
    slate=slate.sort_values(["kickoff_utc","game_id"]).reset_index(drop=True)
    return slate

def export_slate(*,prediction_csv,output_dir,now_utc=None,horizon_days=7):
    now=now_utc or datetime.now(timezone.utc).isoformat()
    path=Path(prediction_csv)
    data=path.read_bytes()
    slate=freeze_slate(pd.read_csv(path),now_utc=now,horizon_days=horizon_days)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    slate_path=out/"next_slate_v3_research.csv"
    slate.to_csv(slate_path,index=False)
    manifest={"prepared_at_utc":pd.Timestamp(now).isoformat(),
              "source_prediction_file_sha256":hashlib.sha256(data).hexdigest(),
              "slate_sha256":hashlib.sha256(slate_path.read_bytes()).hexdigest(),
              "horizon_days":horizon_days,"selected_games":len(slate),
              "kickoff_source_independently_verified":False,
              "source_publication_verified":False,"market_quotes_verified":False,
              "prospective_ledger_written":False,
              "note":"Predictions retain original features_asof_utc; no backdating or new inference."}
    (out/"next_slate_v3_manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions-csv",required=True)
    p.add_argument("--output-dir",default="research_outputs/next_slate")
    p.add_argument("--horizon-days",type=int,default=7)
    a=p.parse_args()
    print(json.dumps(export_slate(prediction_csv=a.predictions_csv,
        output_dir=a.output_dir,horizon_days=a.horizon_days),indent=2))
if __name__=="__main__":main()
