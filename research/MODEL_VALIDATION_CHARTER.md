# Edge Intelligence — Model Validation Charter (2026-11-16)

Status: RESEARCH ONLY. No real-money wagering, no claims of profitability.

## Objective
Estimate outcome probabilities independently, compare against time-stamped executable sportsbook prices, and determine whether a repeatable positive expected-value signal survives forward testing.

## Current audit (static code review; no models executed)
- `nfl_edge.py` implements chronological NFL features, ridge margin/total, logistic win probability, and residual-based simulations.
- `walk_forward_backtest` refits by historical cutoff, but its ATS reporting is not a complete wager-level profitability test.
- `NFLEdgeModel.fit` calculates simulation residuals on its training observations; replace with out-of-fold, time-respecting residuals before interpreting simulated cover probabilities.
- `player_engine.py` produces recency-weighted fantasy-point projections; its heuristic confidence is NOT calibrated prop-hit probability.
- `player_backtest.py` inner-joins predictions to players with observed target-week records, risking selection bias for actionable pregame prop opportunities.
- `app.py` includes a hard-coded October 2026 game-night market snapshot and is not an automated odds capture pipeline.
- Historical price provenance, availability timestamps, sportsbook-specific bet grading, and a trade ledger have not yet been verified.

## Guardrails
1. Keep `main` and frozen model artifacts untouched while auditing.
2. Use point-in-time datasets. Every input requires an as-of timestamp no later than the selection timestamp.
3. Split train/validation/test chronologically, with a locked holdout. Never tune thresholds on the test set.
4. Compare model-only, no-vig market baseline, and market-plus-model combined baseline.
5. Record every candidate including PASS, odds source, time, book, selection, limits if known, model version, and eventual outcome.
6. Report Brier score, log loss, calibration by bins, closing-line value, bet count, ROI, drawdown, and uncertainty intervals.
7. Account for vig, pushes, voids, missing lines, price movement, correlated selections, and costs.
8. Paper trade prospectively before considering any real-money use. No profitability guarantee.

## First experiment
Market: NFL pregame moneyline, two-way only.
Unit: one timestamped game-market-price observation per book.
Baseline: no-vig consensus market probability.
Candidate: chronological football model, optionally combined with no-vig market probabilities.
Evaluation: rolling-origin backtest followed by forward-only paper ledger.
Promotion: only if calibrated on unseen games and a positive expected edge persists after conservative costs; otherwise retain PASS.
Next candidates: NBA moneyline, NHL regulation/including-OT moneyline (separate grading), then totals and player props.

## Outstanding work
- Inventory all source files and any historical odds datasets.
- Reproduce current NFL walk-forward results.
- Audit schedule sorting, same-day ordering, leakage and data availability.
- Create timestamped odds schema and ingestion tests.
- Replace in-sample simulation residuals with time-respecting out-of-fold estimates.
- Build a deterministic wager settlement engine and immutable paper ledger.
- Test data licensing and operating costs before adding more sports.
