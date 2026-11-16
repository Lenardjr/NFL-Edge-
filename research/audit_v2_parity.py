"""Audit V2 paired backtest against the independently exported V1 baseline.

This guards against silent changes to the original model or mismatched games.
A failed parity check invalidates V2 comparisons until investigated.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def audit(baseline,v2,tolerance=1e-8):
    required_base={"game_id","home_win","model_home_prob","season"}
    required_v2={"game_id","home_win","baseline_prob","v2_prob","season"}
    if required_base-set(baseline):raise ValueError("Baseline missing required fields")
    if required_v2-set(v2):raise ValueError("V2 missing required fields")
    if baseline.game_id.duplicated().any() or v2.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs")
    joined=baseline.merge(v2,on="game_id",how="outer",indicator=True,suffixes=("_v1","_v2"),validate="one_to_one")
    only_v1=int((joined._merge=="left_only").sum())
    only_v2=int((joined._merge=="right_only").sum())
    both=joined[joined._merge=="both"].copy()
    if both.empty:raise ValueError("No matching games")
    max_delta=float(np.max(np.abs(both.model_home_prob-both.baseline_prob)))
    outcome_mismatch=int((both.home_win_v1!=both.home_win_v2).sum())
    season_mismatch=int((both.season_v1!=both.season_v2).sum())
    # If both experiments cover the same universe, the baseline predictions
    # should match up to floating-point tolerance.
    passed=(only_v1==0 and only_v2==0 and outcome_mismatch==0
            and season_mismatch==0 and max_delta<=tolerance)
    return {"passed":bool(passed),"baseline_games":len(baseline),"v2_games":len(v2),
        "matched_games":len(both),"only_in_baseline":only_v1,"only_in_v2":only_v2,
        "outcome_mismatches":outcome_mismatch,"season_mismatches":season_mismatch,
        "max_abs_baseline_probability_difference":max_delta,
        "tolerance":tolerance,
        "decision":"paired V2 comparison valid" if passed else "STOP: investigate mismatch before interpreting V2 scores"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",required=True)
    ap.add_argument("--v2",required=True)
    ap.add_argument("--output",default="research_outputs/nfl_v2_parity_audit.json")
    args=ap.parse_args()
    result=audit(pd.read_csv(args.baseline),pd.read_csv(args.v2))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result["passed"]:raise SystemExit(1)
if __name__=="__main__":main()
