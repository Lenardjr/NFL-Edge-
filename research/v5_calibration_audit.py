"""Research-only forecast calibration audit; no fitted model or betting claims.

Fixed pregame-defined bins and season-level breakdowns. Retrospective slices
must not be used as a deployed selection rule without prospective validation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

PROB_EDGES=np.linspace(0,1,11)
SIGNED_EDGES=[-np.inf,-.20,-.10,-.05,-.025,0,.025,.05,.10,.20,np.inf]

def audit(df:pd.DataFrame)->dict:
    required={"game_id","season","home_win","v3_prob","market_prob"}
    if required-set(df):raise ValueError("Missing columns: "+str(sorted(required-set(df))))
    if df.game_id.isna().any() or df.game_id.duplicated().any():raise ValueError("Missing or duplicate game IDs")
    if df[list(required)].isna().any().any():raise ValueError("Missing paired observations")
    if not df.home_win.isin([0,1]).all():raise ValueError("Invalid outcomes")
    for c in ("v3_prob","market_prob"):
        if not np.isfinite(df[c].to_numpy(dtype=float)).all() or not df[c].between(0,1).all():
            raise ValueError("Invalid probabilities")
    d=df.copy()
    d["signed_gap"]=d.v3_prob-d.market_prob
    d["signed_gap_bin"]=pd.cut(d.signed_gap,SIGNED_EDGES,include_lowest=True,right=False)
    def row(g,source):
        p=g[source].to_numpy(dtype=float);y=g.home_win.to_numpy(dtype=float)
        return {"games":len(g),"mean_predicted":float(p.mean()),
                "observed_win_rate":float(y.mean()),
                "calibration_error":float(p.mean()-y.mean()),
                "brier":float(np.mean((p-y)**2))}
    result={"games":len(d),"probability_reliability":{},"signed_gap_bins":[],
            "season_signed_gap":[],
            "notes":["Fixed bins; retrospective diagnostic, not a trained forecast or bet filter",
                     "Season and bin comparisons are descriptive; many small samples",
                     "Market odds lack verified pregame capture times and executable quotes"]}
    for source in ("v3_prob","market_prob"):
        b=pd.cut(d[source],PROB_EDGES,include_lowest=True,right=False)
        result["probability_reliability"][source]=[
            {"bin":str(k),**row(g,source)} for k,g in d.groupby(b,observed=True,sort=True)]
    for k,g in d.groupby("signed_gap_bin",observed=True,sort=True):
        result["signed_gap_bins"].append({"bin":str(k),"games":len(g),
            "v3":row(g,"v3_prob"),"market":row(g,"market_prob"),
            "v3_minus_market_brier":row(g,"v3_prob")["brier"]-row(g,"market_prob")["brier"]})
    for (season,k),g in d.groupby(["season","signed_gap_bin"],observed=True,sort=True):
        result["season_signed_gap"].append({"season":int(season),"bin":str(k),"games":len(g),
            "v3_minus_market_brier":row(g,"v3_prob")["brier"]-row(g,"market_prob")["brier"]})
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="research_outputs/nfl_v3_predictions.csv")
    ap.add_argument("--output",default="research_outputs/nfl_v5_calibration_audit.json")
    a=ap.parse_args()
    r=audit(pd.read_csv(a.predictions))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2))
    print(json.dumps({"games":r["games"],"signed_gap_bins":r["signed_gap_bins"],
                      "notes":r["notes"]},indent=2))
if __name__=="__main__":main()
