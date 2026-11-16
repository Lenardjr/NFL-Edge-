"""Research-only diagnostic of model-vs-market errors.

Usage:
 python -m research.diagnose_disagreement --predictions research_outputs/nfl_baseline_predictions.csv
No wager prices or profit claims are derived from closing-line benchmarks.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd

def diagnose(data: pd.DataFrame) -> dict:
    required={"home_win","model_home_prob","market_home_prob"}
    missing=required-set(data.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if data[list(required)].isna().any().any(): raise ValueError("Missing predictions or outcomes")
    y=data.home_win.to_numpy(dtype=float)
    model=data.model_home_prob.to_numpy(dtype=float)
    market=data.market_home_prob.to_numpy(dtype=float)
    if len(data)==0 or not np.isin(y,[0,1]).all() or not np.isfinite(model).all() or not np.isfinite(market).all() or not ((model>=0)&(model<=1)&(market>=0)&(market<=1)).all():
        raise ValueError("Invalid binary outcomes or probabilities")
    agree=(model>=.5)==(market>=.5)
    def summarize(mask):
        n=int(mask.sum())
        return dict(games=n, model_brier=float(np.mean((model[mask]-y[mask])**2)) if n else None,
                    market_brier=float(np.mean((market[mask]-y[mask])**2)) if n else None)
    bins=[]
    for lo,hi in ((0,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,1.0000001)):
        mask=(model>=lo)&(model<hi)
        bins.append(dict(lower=lo,upper=min(hi,1),games=int(mask.sum()),
                         mean_model_probability=float(model[mask].mean()) if mask.any() else None,
                         observed_home_win_rate=float(y[mask].mean()) if mask.any() else None))
    # Cluster bootstrap by season when available; captures between-season variability.
    # This is exploratory, not a confidence interval for profitable betting.
    diffs=(model-y)**2-(market-y)**2
    rng=np.random.default_rng(19)
    draws=[]
    if "season" in data.columns:
        seasons=data.season.to_numpy()
        unique=np.unique(seasons)
        for _ in range(3000):
            selected=rng.choice(unique,len(unique),replace=True)
            idx=np.concatenate([np.flatnonzero(seasons==s) for s in selected])
            draws.append(float(diffs[idx].mean()))
        method="season-cluster bootstrap"
    else:
        for _ in range(3000):
            idx=rng.integers(0,len(data),len(data))
            draws.append(float(diffs[idx].mean()))
        method="game-level bootstrap"
    return dict(overall=summarize(np.ones(len(data),dtype=bool)),
                same_favorite=summarize(agree),opposite_favorites=summarize(~agree),
                calibration_bins=bins,
                model_minus_market_brier=float(diffs.mean()),
                brier_difference_95pct_exploratory_interval=[float(v) for v in np.quantile(draws,[.025,.975])],
                bootstrap_method=method,
                note="Historical market benchmark only; no verified executable prices or betting ROI.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",required=True)
    p.add_argument("--output",default="research_outputs/nfl_disagreement_diagnostics.json")
    args=p.parse_args()
    result=diagnose(pd.read_csv(args.predictions))
    from pathlib import Path
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
