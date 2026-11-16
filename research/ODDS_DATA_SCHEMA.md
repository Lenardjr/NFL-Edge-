# Historical moneyline odds — required research fields

One record per book, game, selection, and quote timestamp.

| Field | Example | Requirement |
|---|---|---|
| event_id | NFL_2026_W11_AWAY_HOME | stable unique game ID |
| league | NFL | sport and league |
| market | moneyline_full_game | include overtime/settlement rules |
| selection | home | home/away |
| sportsbook | source_book | actual source identity |
| source | licensed_api_name | provenance / licensing |
| captured_at | 2026-11-16T18:00:00-05:00 | timezone-aware quote time |
| scheduled_start_at | 2026-11-16T20:00:00-05:00 | timezone-aware event start |
| american_odds | +120 | price at captured time |
| model_version | model_001 | frozen model identifier |
| model_probability | 0.52 | pregame probability |
| market_no_vig_probability | 0.48 | from paired same-book, same-time prices |
| decision | PASS | predeclared policy; record every candidate |
| result | null | append only after event final |

Keep raw source files immutable and store hashes. Never use closing odds as a substitute for a pregame quote unless the strategy explicitly selects at close. Reject missing, post-kickoff, stale, or unpaired prices. For ties, postponements, and voids apply the actual book's market settlement rules. Data availability at historical decision time must be verified; a current API value is not evidence of a past available price.
