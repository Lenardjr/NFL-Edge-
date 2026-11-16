# Research audit — 2026-11-16

## Confirmed locally
Six deterministic checks of the market arithmetic and timestamp/settlement logic passed in an isolated Python reproduction. This does **not** constitute a successful GitHub Actions run or historical-model execution.

## Important blockers before interpreting NFL baseline results
1. **Same-day leakage risk:** `game_engine.pregame_features` updates team state immediately after each completed row. On retrospectively completed schedule files, a later game on the same date may incorporate an earlier game result even if the intended prediction timestamp preceded that earlier game. The research runner must implement point-in-time feature generation with an explicit prediction timestamp, not assume historical completed results were available.
2. **Market data semantics:** nflverse schedule moneylines are benchmark prices, not a verified timestamped executable quote at a specified sportsbook. Do not compute strategy ROI from them.
3. **Reproducibility:** record input URL, content hash, retrieval time, data coverage, dependency versions and model revision.
4. **Missing/ambiguous data:** audit game IDs, team aliases, neutral-site designations, tied games, scheduled versus completed games, and missing moneylines.
5. **Data availability:** live NFL feeds and GitHub workflow execution still need independent verification. The baseline runner is committed but no historical output has been reviewed.
6. **Selection bias:** do not report only games where the model has odds; show full eligible count and exclusions.
7. **Forecast residuals:** production engine uses in-sample residuals for simulations, so those cover/total probabilities require independent recalibration.

## Next implementation gate
Introduce an as-of timestamp for each prediction; only ingest results with completion times strictly earlier than that timestamp. Where completion timestamps are unavailable, use conservative full-date cutoffs and batch game-state updates by date. Add regression tests with multiple same-day games and reordered input to demonstrate invariance. Only then publish historical model metrics.
