"""Immutable-style append-only paper prediction ledger validation.

This research module validates pregame snapshots and settlement. It does NOT
place wagers or retrieve live odds. Write snapshots to a version-controlled
or externally timestamped log; local CSV alone is not tamper proof.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED=["game_id","kickoff_utc","captured_at_utc","home_team","away_team",
          "model_version","model_home_prob","home_moneyline","away_moneyline",
          "sportsbook","odds_captured_at_utc"]

def validate_predictions(df):
    missing=set(REQUIRED)-set(df.columns)
    if missing:raise ValueError(f"Missing required fields: {sorted(missing)}")
    d=df.copy()
    for col in ("kickoff_utc","captured_at_utc","odds_captured_at_utc"):
        d[col]=pd.to_datetime(d[col],utc=True,errors="coerce")
        if d[col].isna().any():raise ValueError(f"Invalid {col}")
    if not (d.captured_at_utc<d.kickoff_utc).all():raise ValueError("Predictions must precede kickoff")
    if not (d.odds_captured_at_utc<=d.captured_at_utc).all():raise ValueError("Odds must exist at prediction capture")
    if not d.model_home_prob.between(0,1).all():raise ValueError("Invalid model probabilities")
    if not np.isfinite(pd.to_numeric(d.home_moneyline,errors="coerce")).all() or not np.isfinite(pd.to_numeric(d.away_moneyline,errors="coerce")).all():
        raise ValueError("Missing or invalid moneylines")
    if not ((d.home_moneyline.abs()>=100)&(d.away_moneyline.abs()>=100)).all():
        raise ValueError("Invalid American moneyline magnitudes")
    if d.duplicated(["game_id","model_version","sportsbook"]).any():
        raise ValueError("Duplicate snapshot: only one locked snapshot per game/model/book")
    for col in ("game_id","home_team","away_team","model_version","sportsbook"):
        if d[col].isna().any() or (d[col].astype(str).str.strip()=="").any():
            raise ValueError(f"Missing {col}")
    return d

def american_profit_per_unit(odds):
    odds=float(odds)
    if abs(odds)<100:raise ValueError("Invalid American odds")
    return odds/100 if odds>0 else 100/abs(odds)

def settle(predictions,results,threshold=0.0):
    p=validate_predictions(predictions)
    required={"game_id","home_win"}
    if not required.issubset(results):raise ValueError("Missing result columns")
    if results.game_id.duplicated().any():raise ValueError("Duplicate results")
    merged=p.merge(results[["game_id","home_win"]],on="game_id",how="left",validate="many_to_one")
    if not merged.home_win.dropna().isin([0,1]).all():raise ValueError("Invalid outcomes")
    # Selection is based only on the frozen model estimate and odds at capture.
    def implied(odds):
        return 100/(odds+100) if odds>0 else -odds/(-odds+100)
    picks=[]
    for _,r in merged.iterrows():
        ph=float(r.model_home_prob)
        h=float(r.home_moneyline);a=float(r.away_moneyline)
        ev_home=ph*american_profit_per_unit(h)-(1-ph)
        ev_away=(1-ph)*american_profit_per_unit(a)-ph
        if max(ev_home,ev_away)<=threshold:side="pass";profit=np.nan
        else:
            side="home" if ev_home>=ev_away else "away"
            if pd.isna(r.home_win):profit=np.nan
            else:
                won=(int(r.home_win)==1) if side=="home" else (int(r.home_win)==0)
                profit=american_profit_per_unit(h if side=="home" else a) if won else -1.0
        picks.append({"game_id":r.game_id,"model_version":r.model_version,"sportsbook":r.sportsbook,
            "selection":side,"expected_value_home":ev_home,"expected_value_away":ev_away,
            "profit_units":profit})
    out=pd.DataFrame(picks)
    completed=out[(out.selection!="pass")&out.profit_units.notna()]
    return out,{"completed_bets":len(completed),
        "total_profit_units":float(completed.profit_units.sum()),
        "flat_stake_roi":float(completed.profit_units.mean()) if len(completed) else None,
        "unsettled_or_passed_games":len(out)-len(completed),
        "note":"Paper-only hypothetical returns using supplied timestamped quotes; no commissions, limits, slippage or execution verification."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--results",required=True)
    ap.add_argument("--output-dir",default="research_outputs")
    ap.add_argument("--min-ev",type=float,default=0.0)
    a=ap.parse_args()
    bets,summary=settle(pd.read_csv(a.predictions),pd.read_csv(a.results),a.min_ev)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    bets.to_csv(out/"paper_settlements.csv",index=False)
    (out/"paper_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
