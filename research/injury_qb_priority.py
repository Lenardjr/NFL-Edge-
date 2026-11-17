"""Rank QB source-verification cases using pre-outcome model/market disagreement only.

This queue is a research triage tool, not a betting signal or verified pregame data.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def prioritize(queue:pd.DataFrame,cases:pd.DataFrame,limit:int=20):
    qcols={"game_id","team","side","player","date_modified","kickoff_utc","source_verification"}
    ccols={"game_id","season","week","v3_prob","market_prob"}
    if qcols-set(queue):raise ValueError("Missing QB queue columns")
    if ccols-set(cases):raise ValueError("Missing extreme case columns")
    if limit<1:raise ValueError("Invalid limit")
    if cases.game_id.isna().any() or cases.game_id.duplicated().any():
        raise ValueError("Invalid case game IDs")
    if queue.empty:
        return pd.DataFrame(columns=["priority_rank","game_id","season","week","absolute_gap","qb_reports","qb_teams","week18","verification_status"]),{"priority_games":0,"priority_qb_rows":0,"verified_pregame":0}
    if queue.game_id.isna().any():raise ValueError("Missing QB game ID")
    c=cases[["game_id","season","week","v3_prob","market_prob"]].copy()
    for name in ["v3_prob","market_prob"]:
        values=pd.to_numeric(c[name],errors="coerce")
        if not np.isfinite(values).all() or not values.between(0,1).all():
            raise ValueError("Invalid probability")
        c[name]=values
    c["absolute_gap"]=(c.v3_prob-c.market_prob).abs()
    q=queue.merge(c,on="game_id",how="left",validate="many_to_one",indicator=True)
    if not q["_merge"].eq("both").all():raise ValueError("QB game absent from cases")
    g=q.groupby("game_id",sort=False).agg(season=("season","first"),week=("week","first"),
        absolute_gap=("absolute_gap","first"),qb_reports=("game_id","size"),
        qb_teams=("team","nunique")).reset_index()
    g["week18"]=g.week.eq(18)
    g=g.sort_values(["absolute_gap","game_id"],ascending=[False,True],kind="stable").head(limit).reset_index(drop=True)
    g.insert(0,"priority_rank",range(1,len(g)+1))
    g["verification_status"]="pending_independent_source"
    summary={"priority_games":len(g),"priority_qb_rows":int(q[q.game_id.isin(g.game_id)].shape[0]),
        "week18_games":int(g.week18.sum()),"verified_pregame":0,
        "ranking":"Descending absolute V3-market gap; deterministic game ID tiebreak; no outcomes or hindsight performance used",
        "warning":"Research queue only; historical injury source timestamps are not authenticated publication captures."}
    return g,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--queue",default="research_outputs/nfl_2024_qb_source_verification_queue.csv")
    p.add_argument("--cases",default="research_outputs/nfl_v5_extreme_case_review.csv")
    p.add_argument("--limit",type=int,default=20)
    p.add_argument("--output-dir",default="research_outputs")
    a=p.parse_args()
    prioritized,summary=prioritize(pd.read_csv(a.queue),pd.read_csv(a.cases),a.limit)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    prioritized.to_csv(out/"nfl_2024_qb_priority_games.csv",index=False)
    (out/"nfl_2024_qb_priority_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
