"""Score a frozen prediction cohort against independently supplied final scores.

Never modify the frozen predictions. No odds, ROI, or profitability claim is
made unless separately verified market data is supplied in another workflow.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def score_cohort(predictions:pd.DataFrame,results:pd.DataFrame):
    required={"game_id","home_team","away_team","home_win_probability","kickoff_utc","features_asof_utc"}
    if required-set(predictions):raise ValueError("Missing cohort fields")
    if {"game_id","home_score","away_score"}-set(results):raise ValueError("Missing results fields")
    p=predictions.copy();r=results.copy()
    if p.game_id.isna().any() or p.game_id.duplicated().any():raise ValueError("Duplicate cohort games")
    if r.game_id.isna().any() or r.game_id.duplicated().any():raise ValueError("Duplicate result games")
    p["kickoff_utc"]=pd.to_datetime(p.kickoff_utc,utc=True,errors="raise")
    p["features_asof_utc"]=pd.to_datetime(p.features_asof_utc,utc=True,errors="raise")
    if (p.features_asof_utc>=p.kickoff_utc).any():raise ValueError("Non-pregame cohort timestamp")
    p["home_win_probability"]=pd.to_numeric(p.home_win_probability,errors="raise")
    if not p.home_win_probability.between(0,1,inclusive="neither").all():
        raise ValueError("Invalid probabilities")
    r["home_score"]=pd.to_numeric(r.home_score,errors="coerce")
    r["away_score"]=pd.to_numeric(r.away_score,errors="coerce")
    if (r.home_score.notna()!=r.away_score.notna()).any():raise ValueError("Partial score")
    if "home_team" in r and "away_team" in r:
        merged=p.merge(r,on="game_id",how="left",validate="one_to_one",suffixes=("","_result"),indicator=True)
        matched=merged._merge=="both"
        if ((merged.loc[matched,"home_team"]!=merged.loc[matched,"home_team_result"])|
            (merged.loc[matched,"away_team"]!=merged.loc[matched,"away_team_result"])).any():
            raise ValueError("Results matchup mismatch")
    else:
        merged=p.merge(r,on="game_id",how="left",validate="one_to_one",indicator=True)
    if "status" in merged:
        # Only explicit final/closed results may be scored.
        final=merged.status.astype(str).str.lower().isin(["final","closed"])
    else:
        # Without independently verified final status, treat all as unverified.
        final=pd.Series(False,index=merged.index)
    complete=merged.home_score.notna()&merged.away_score.notna()&final
    ties=complete&(merged.home_score==merged.away_score)
    scored=complete&~ties
    merged["evaluation_status"]=np.select(
        [scored,ties,complete,merged._merge=="left_only"],
        ["scored","tie_excluded","unverified_final","awaiting_result"],
        default="awaiting_verified_final")
    merged["home_win_actual"]=np.where(scored,(merged.home_score>merged.away_score).astype(int),np.nan)
    merged["brier"]=np.where(scored,(merged.home_win_probability-merged.home_win_actual)**2,np.nan)
    merged["correct_pick"]=np.where(scored,((merged.home_win_probability>.5)==(merged.home_win_actual==1)).astype(int),np.nan)
    summary={"cohort_games":len(merged),"scored_games":int(scored.sum()),
             "ties_excluded":int(ties.sum()),"awaiting_verified_results":int((~complete).sum()),
             "brier_score":float(merged.loc[scored,"brier"].mean()) if scored.any() else None,
             "pick_accuracy":float(merged.loc[scored,"correct_pick"].mean()) if scored.any() else None,
             "market_comparison_available":False,"profitability_verified":False}
    return merged.drop(columns=["_merge"]),summary

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--cohort",required=True)
    a.add_argument("--results",required=True)
    a.add_argument("--output-dir",required=True)
    args=a.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    scored,summary=score_cohort(pd.read_csv(args.cohort),pd.read_csv(args.results))
    scored.to_csv(out/"cohort_outcomes.csv",index=False)
    (out/"cohort_evaluation.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
