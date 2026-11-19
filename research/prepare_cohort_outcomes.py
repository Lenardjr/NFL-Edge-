"""Prepare a blank, non-scoring outcomes template for the fixed Nov 19 cohort."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import pandas as pd

def prepare_template(cohort_csv,output_dir):
    source=Path(cohort_csv);data=source.read_bytes()
    p=pd.read_csv(source)
    required={"game_id","home_team","away_team","kickoff_utc","home_win_probability"}
    if required-set(p):raise ValueError("Missing cohort fields")
    if p.game_id.isna().any() or p.game_id.duplicated().any():raise ValueError("Duplicate game IDs")
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    t=p[["game_id","home_team","away_team"]].copy()
    t["home_score"]=pd.NA;t["away_score"]=pd.NA;t["status"]="unverified"
    t.to_csv(out/"cohort_results_template.csv",index=False)
    metadata={"cohort_sha256":hashlib.sha256(data).hexdigest(),
              "cohort_games":len(p),"results_verified":False,
              "source_run_id":44595866608,"source_artifact_id":14566315811,
              "note":"Do not fill final scores without verified outcome source; original predictions are unchanged"}
    (out/"cohort_template_manifest.json").write_text(json.dumps(metadata,indent=2))
    return metadata
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--cohort",required=True);a.add_argument("--output-dir",required=True)
    args=a.parse_args()
    print(json.dumps(prepare_template(args.cohort,args.output_dir),indent=2))
if __name__=="__main__":main()
