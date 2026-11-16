"""Reproducible NFL game-model backtest; never asserts betting profitability.

Usage:
 python -m research.run_nfl_baseline --games-csv data/games.csv --output-dir research_outputs
Or omit --games-csv to download nflverse schedules (requires internet).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from game_engine import FEATURES, pipe, pregame_features
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from research.market_validation import brier_score, log_loss, no_vig_two_way

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"

def run(games: pd.DataFrame, first_test_season: int = 2022):
    g = games.copy()
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    for c in ("home_score", "away_score", "home_moneyline", "away_moneyline"):
        if c in g: g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.sort_values(["gameday", "gametime", "game_id"], na_position="last").reset_index(drop=True)
    features = pregame_features(g)
    done = features.dropna(subset=["target_margin", "target_total"]).copy()
    done = done.sort_values(["gameday", "gametime", "game_id"], na_position="last")
    test = done[(done.season >= first_test_season) & (done.home_games_prior >= 2) & (done.away_games_prior >= 2)]
    rows = []
    for (season, week), week_games in test.groupby(["season", "week"], sort=True):
        # Strictly earlier calendar dates; never train on games from test date.
        cutoff = week_games.gameday.min()
        train = done[(done.gameday < cutoff) & (done.home_games_prior >= 2) & (done.away_games_prior >= 2)]
        if len(train) < 300: continue
        clf = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("model", LogisticRegression(C=.45, max_iter=2000))])
        clf.fit(train[FEATURES], (train.target_margin > 0).astype(int))
        pred = clf.predict_proba(week_games[FEATURES])[:, 1]
        for (_, game), p in zip(week_games.iterrows(), pred):
            if game.target_margin == 0: continue  # Two-way market tie handling requires book rules.
            record = dict(game_id=str(game.get("game_id", "")), season=int(season), week=int(week),
                          gameday=str(game.gameday.date()), home_team=game.home_team,
                          away_team=game.away_team, model_home_prob=float(p),
                          home_win=int(game.target_margin > 0))
            h, a = game.get("home_moneyline"), game.get("away_moneyline")
            if pd.notna(h) and pd.notna(a) and abs(h) >= 100 and abs(a) >= 100:
                # NFLverse schedule moneylines may be closing lines: benchmark only, NOT executable historical quotes.
                record["market_home_prob"] = no_vig_two_way(int(h), int(a))[0]
            else:
                record["market_home_prob"] = float("nan")
            rows.append(record)
    return pd.DataFrame(rows)

def summarize(predictions):
    if predictions.empty: return {"games": 0, "status": "insufficient history"}
    y = predictions.home_win.tolist()
    p = predictions.model_home_prob.tolist()
    out = {"games": len(predictions), "model_brier": brier_score(p,y),
           "model_log_loss": log_loss(p,y), "model_accuracy": sum((v >= .5)==bool(t) for v,t in zip(p,y))/len(y)}
    market = predictions.dropna(subset=["market_home_prob"])
    out["market_games"] = len(market)
    if len(market):
        out["market_brier"] = brier_score(market.market_home_prob.tolist(),market.home_win.tolist())
        out["market_log_loss"] = log_loss(market.market_home_prob.tolist(),market.home_win.tolist())
    out["status"] = "prediction benchmark only; not a betting ROI test"
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--games-csv", default=GAMES_URL)
    parser.add_argument("--output-dir", default="research_outputs")
    parser.add_argument("--first-test-season",type=int,default=2022)
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    predictions=run(pd.read_csv(args.games_csv),args.first_test_season)
    predictions.to_csv(output/"nfl_baseline_predictions.csv",index=False)
    import json
    summary=summarize(predictions)
    (output/"nfl_baseline_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
