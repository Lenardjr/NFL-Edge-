"""Paired, exploratory V3 assessment on the same held-out games.

Season-cluster resampling is a sensitivity check, not a formal significance
claim with only five seasons (one partial). No betting ROI is inferred.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def compare(df: pd.DataFrame, draws: int = 5000, seed: int = 2026) -> dict:
    required = {"game_id", "season", "home_win", "baseline_prob", "v3_prob", "market_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.game_id.duplicated().any():
        raise ValueError("Duplicate game_id")
    d = df.dropna(subset=list(required)).copy()
    if d.empty or not d.home_win.isin([0, 1]).all():
        raise ValueError("No valid paired games or invalid outcomes")
    for c in ("baseline_prob", "v3_prob", "market_prob"):
        if not d[c].between(0, 1).all():
            raise ValueError(f"Invalid probability: {c}")
    y = d.home_win.to_numpy(dtype=float)
    b = d.baseline_prob.to_numpy(dtype=float)
    v = d.v3_prob.to_numpy(dtype=float)
    m = d.market_prob.to_numpy(dtype=float)
    vb = (v-y)**2-(b-y)**2
    vm = (v-y)**2-(m-y)**2
    accuracy = ((v>=.5)==y).astype(float)-((b>=.5)==y).astype(float)
    years = d.season.to_numpy()
    unique = np.unique(years)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, 3))
    indices = [np.flatnonzero(years==s) for s in unique]
    for i in range(draws):
        chosen = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([indices[j] for j in chosen])
        boot[i] = [vb[idx].mean(), vm[idx].mean(), accuracy[idx].mean()]
    ci = np.quantile(boot, [.025, .975], axis=0)
    def stats(mask):
        n = int(mask.sum())
        if not n:
            return {"games": 0}
        return {"games": n,
                "v3_minus_baseline_brier": float(vb[mask].mean()),
                "v3_minus_market_brier": float(vm[mask].mean()),
                "v3_accuracy": float(((v[mask]>=.5)==y[mask]).mean()),
                "market_accuracy": float(((m[mask]>=.5)==y[mask]).mean())}
    disagreement = (v>=.5)!=(m>=.5)
    seasons = {str(s):stats(years==s) for s in unique}
    return {"paired_games": len(d), "overall": stats(np.ones(len(d), dtype=bool)),
            "v3_minus_baseline_accuracy": float(accuracy.mean()),
            "season_cluster_bootstrap_95pct": {
                "v3_minus_baseline_brier": ci[:,0].tolist(),
                "v3_minus_market_brier": ci[:,1].tolist(),
                "v3_minus_baseline_accuracy": ci[:,2].tolist()},
            "seasons": seasons,
            "v3_market_disagreement": {
                "disagree": stats(disagreement),
                "agree": stats(~disagreement),
                "v3_correct_market_wrong": int(np.sum(disagreement & ((v>=.5)==y))),
                "market_correct_v3_wrong": int(np.sum(disagreement & ((m>=.5)==y)))},
            "decision": "research only; no production promotion",
            "caveats": "Five season clusters, one partial; exploratory bootstrap is fragile. Market odds are historical, not timestamped executable quotes."}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="research_outputs/nfl_v3_predictions.csv")
    ap.add_argument("--output", default="research_outputs/nfl_v3_comparison.json")
    args = ap.parse_args()
    report = compare(pd.read_csv(args.predictions))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
