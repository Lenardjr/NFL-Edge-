"""Phase 1 model-integrity audit: schedule, EPA and historical quote checks.

Read-only diagnostic. Fail on structural inconsistencies, report provenance limits.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from game_engine import TEAM_ALIASES
from research.run_nfl_v3 import SCHEDULE_URL

def audit(games:pd.DataFrame,predictions:pd.DataFrame)->dict:
    required={"game_id","gameday","season","week","home_team","away_team",
              "home_score","away_score","home_moneyline","away_moneyline"}
    missing=required-set(games)
    if missing:raise ValueError("Missing schedule fields: "+str(sorted(missing)))
    pcols={"game_id","season","week","home_win","v3_prob","market_prob"}
    if pcols-set(predictions):raise ValueError("Missing prediction fields")
    if games.game_id.isna().any() or games.game_id.duplicated().any():
        raise ValueError("Duplicate or null schedule game IDs")
    if predictions.game_id.isna().any() or predictions.game_id.duplicated().any():
        raise ValueError("Duplicate or null prediction IDs")
    g=games.copy()
    g.gameday=pd.to_datetime(g.gameday,errors="coerce")
    if g.gameday.isna().any():raise ValueError("Invalid schedule dates")
    for side in ("home_team","away_team"):
        if g[side].isna().any():raise ValueError("Null team identifiers")
        g[side]=g[side].map(lambda t:TEAM_ALIASES.get(str(t),str(t)))
    if (g.home_team==g.away_team).any():raise ValueError("Home and away team identical")
    if g.duplicated(["gameday","home_team"]).any() or g.duplicated(["gameday","away_team"]).any():
        raise ValueError("Team scheduled twice as same side on one day")
    long=pd.concat([g[["game_id","gameday","home_team"]].rename(columns={"home_team":"team"}),
                    g[["game_id","gameday","away_team"]].rename(columns={"away_team":"team"})])
    if long.duplicated(["gameday","team"]).any():raise ValueError("Team appears twice on same date")
    g["home_score"]=pd.to_numeric(g.home_score,errors="coerce")
    g["away_score"]=pd.to_numeric(g.away_score,errors="coerce")
    if ((g.home_score.isna()) != (g.away_score.isna())).any():
        raise ValueError("Only one team has a final score")
    if (g[["home_score","away_score"]].dropna()<0).any().any():
        raise ValueError("Negative final score")
    merged=predictions.merge(g,on="game_id",how="left",validate="one_to_one",indicator=True,
                             suffixes=("_pred","_schedule"))
    if not (merged._merge=="both").all():raise ValueError("Predictions missing from schedule")
    if not (merged.season_pred==merged.season_schedule).all() or not (merged.week_pred==merged.week_schedule).all():
        raise ValueError("Season/week alignment mismatch")
    if merged[["home_score","away_score"]].isna().any().any():
        raise ValueError("Predictions on unfinished games")
    if (merged.home_score==merged.away_score).any():raise ValueError("Ties in binary predictions")
    if not (merged.home_win==(merged.home_score>merged.away_score).astype(int)).all():
        raise ValueError("Home-win outcome orientation mismatch")
    for col in ("v3_prob","market_prob"):
        values=pd.to_numeric(merged[col],errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
            raise ValueError("Invalid predicted probability: "+col)
    from research.market_validation import no_vig_two_way
    ml_home=pd.to_numeric(merged.home_moneyline,errors="coerce")
    ml_away=pd.to_numeric(merged.away_moneyline,errors="coerce")
    if ml_home.isna().any() or ml_away.isna().any():raise ValueError("Missing historical odds")
    if ((ml_home.abs()<100)|(ml_away.abs()<100)).any():raise ValueError("Invalid American odds")
    reconstructed=np.array([no_vig_two_way(int(h),int(a))[0] for h,a in zip(ml_home,ml_away)])
    if not np.allclose(reconstructed,merged.market_prob.to_numpy(dtype=float),atol=1e-10,rtol=0):
        raise ValueError("Market probability orientation/arithmetic mismatch")
    return {"status":"passed_structural_checks","schedule_games":len(g),
            "predicted_games":len(merged),
            "unique_seasons":sorted(int(s) for s in merged.season_pred.unique()),
            "checks":["unique game IDs","valid dates and teams","one game per team per day",
                      "completed games only","season/week match","home winner orientation",
                      "probability bounds","market no-vig orientation and arithmetic"],
            "limitations":["Schedule moneylines have no verified quote capture timestamps",
                           "Structural checks do not prove historical injury/QB availability",
                           "Training-data temporal invariance requires separate perturbation tests",
                           "No betting profitability inference"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--games",default=SCHEDULE_URL)
    ap.add_argument("--predictions",default="research_outputs/nfl_v3_predictions.csv")
    ap.add_argument("--output",default="research_outputs/nfl_integrity_audit.json")
    a=ap.parse_args()
    result=audit(pd.read_csv(a.games),pd.read_csv(a.predictions))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
