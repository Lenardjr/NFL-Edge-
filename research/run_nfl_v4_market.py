"""V4 exploratory market-relative correction of out-of-sample V3 probabilities.

The input V3 predictions are chronological held-out predictions. V4 trains on
earlier completed seasons only, never the evaluated season. Historical market
prices are NOT timestamped executable betting quotes.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from research.market_residual import run as residual_run

def evaluate(v3:pd.DataFrame):
    needed={"game_id","season","home_win","v3_prob","market_prob"}
    missing=needed-set(v3.columns)
    if missing:raise ValueError(f"Missing V3 columns: {sorted(missing)}")
    if v3.game_id.duplicated().any():raise ValueError("Duplicate game_id")
    if v3[list(needed)].isna().any().any():raise ValueError("Missing paired game data")
    if not v3.v3_prob.between(0,1).all() or not v3.market_prob.between(0,1).all():
        raise ValueError("Invalid probabilities")
    # Residual estimator uses season as strict training boundary; date sorts only.
    inp=v3.assign(gameday=v3.season.astype(str)+"-01-01",
                  model_home_prob=v3.v3_prob,
                  market_home_prob=v3.market_prob)
    summary,p=residual_run(inp)
    if p.empty:raise ValueError("Insufficient previous-season data for V4")
    summary["model"]="V4 market-logit plus V3-minus-market logit residual"
    summary["training_rule"]="Prior held-out seasons only; current season excluded"
    summary["evaluation_note"]="Comparison applies only to V4-evaluable seasons, not V3's full 1125-game sample"
    summary["caution"]="Exploratory historical market prices; no timestamped executable odds, ROI or production approval"
    return summary,p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",default="research_outputs/nfl_v3_predictions.csv")
    ap.add_argument("--output-dir",default="research_outputs")
    a=ap.parse_args()
    summary,p=evaluate(pd.read_csv(a.predictions))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"nfl_v4_market_summary.json").write_text(json.dumps(summary,indent=2))
    p.to_csv(out/"nfl_v4_market_predictions.csv",index=False)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
