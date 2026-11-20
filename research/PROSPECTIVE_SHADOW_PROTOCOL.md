# Edge Intelligence: prospective NFL shadow-test protocol (v1)

**Status:** Research only. No real-money wagering, automatic bet placement, or production promotion.

## 1. Preconditions

1. Confirm the latest `Research validation` workflow succeeds and archive `nfl_v2_summary.json`, `nfl_v2_comparison.json`, and `nfl_v2_decision.json`.
2. Inspect the V2-vs-V1 paired sample sizes, coverage, Brier, log loss, season results and confidence intervals. Do not treat a passing historical gate as proof of profitability.
3. Freeze a candidate model commit and configuration before recording prospective picks. If V2 fails, shadow-test V1 for monitoring only; do not promote V2.
4. Record all games in the planned evaluation universe, including passes, to avoid selective reporting.

## 2. Pregame data contract

Each snapshot must contain:
`game_id,kickoff_utc,captured_at_utc,home_team,away_team,model_version,model_home_prob,home_moneyline,away_moneyline,sportsbook,odds_captured_at_utc`.

Additional strongly recommended fields:
`model_commit,market_source_id,odds_quote_id,market_type,book_region,home_team_id,away_team_id,run_id,raw_quote_sha256,model_input_sha256`.

Capture the sportsbook quote and model prediction before kickoff. Odds timestamp must be no later than the prediction snapshot timestamp. Use a single predetermined cutoff (e.g. 60 minutes before scheduled kickoff), document postponements and changes, and mark missing or stale odds as missing rather than fabricating prices.

Store each snapshot as an append-only Git commit or an external timestamped record. A local CSV by itself is not tamper-resistant. Never overwrite an earlier snapshot; append a correction event with reason and timestamp instead. Separate outcome settlement from the frozen pregame ledger.

## 3. Pre-registered analysis

Primary predictive comparison: **paired Brier score** of the frozen model against the same-game no-vig market probability.

Secondary metrics: log loss, calibration by probability range, season-by-season differences, coverage, and model-vs-market disagreement.

Paper-bet analysis is separate and only valid for actual time-stamped available quotes. Use one-unit flat stakes, record the exact sportsbook and line, apply a *pre-registered* minimum EV threshold, and report all passes and unavailable quotes. Do not retrospectively choose thresholds or cherry-pick bookmakers.

No claim of positive expected returns until a sufficiently large **untouched prospective sample** exists, with uncertainty and realistic execution limitations considered. A convenient starting checkpoint is 250+ eligible games, but this is not a statistical power guarantee.

## 4. Promotion decision

A historical backtest gate only permits a prospective shadow test. No deployment or real-money claim follows automatically. Promotion requires stable future performance, adequate sample size, verified quote provenance, operational reproducibility, and an explicit independent review.

## 5. Known limitations

- NFLverse schedule moneylines may be closing odds; they are **not** verified executable quotes at prediction time.
- Opponent-adjusted features are a first prototype; they are not injury, quarterback, play-by-play EPA, or opponent-strength-of-schedule models.
- Season-cluster bootstrap intervals from only a few seasons are fragile.
- Repeated experimentation on the same 2022–2026 seasons risks model-selection overfitting; preserve future untouched games for final evaluation.
