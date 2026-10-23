# BetSightly Independent Project — Complete Technical Audit
**Audit:** 2026-10-23
**Source:** black-jamesbond/betsightly-backend-devil, baseline release/vps-2026-10-23 @ f2914b22245b0f22a0c5175313e8e6b24755b53d
**Scope:** independent VPS application, NOT betsightly.com
**Decision:** NO-GO for replacing the current VPS working tree.

## Scope, methods and limits
Reviewed the GitHub source tree, live-interface status supplied by the operator, code/CI tests, API/router, models, feature engineering, publication, fixture fetching, odds, settlement, security, scheduling, infrastructure, historical NBA CSV and public 2026 basketball provider documentation. Existing full CI has passed on the release; no newly merged VPS candidate has passed CI. This is a repository static/data audit, not penetration-test certification or production performance measurement. The local-only VPS commit fd0c19cd and 73 dirty paths are not accessible through the GitHub connection.

Repository inventory at audit: 438 tracked files, 159 Python files, 14 pytest modules, 72 tracked model artifacts, 172 tracked data JSON/CSV/Parquet files, and 1 CI workflow. No test_basketball module exists.

## P0 critical release blockers

**BS-001 — Three-way merge not resolved.** The VPS isolated reconciliation reported 3 conflicted files: sportybet_jackpot.py (3 hunks), dashboard/app.py (6), dashboard/templates/index.html (8). The live main commit fd0c19cd contains work absent from the release, plus 73 modified/untracked entries. The released repair commit must not replace the live worktree. Preserve all six live tabs, Rollover, Fixtures, Daily Report, basketball and jackpot features, and complete a local candidate merge plus integrated testing. The source merge excluded 23 model/data paths, including tuned calibration and Dixon-Coles artifacts; these require independent backup and migration.

**BS-002 — Basketball feature values are fabricated.** basketball/feature_engineering.py uses random values for form, home/away splits, shooting, pace, rest and head-to-head instead of derived pre-game historical facts. A model trained on these cannot claim evidence-based probabilities. Replace all placeholders with point-in-time database computations, and refuse publication when inputs are unavailable.

**BS-003 — Basketball training can silently use synthetic games and outcomes.** basketball/train_models.py creates sample training data if missing. basketball/train_basketball_models.py falls back to create_synthetic_data after provider failures. basketball/data_fetcher.py:download_historical_data generates a three-row demonstration dataset. basketball/prediction_service.py:train_models substitutes random HOME_WIN or TOTAL_PTS labels when absent. Stop implicit synthetic fallbacks in all production paths. Synthetic fixtures must be test-only and clearly identified.

**BS-004 — Basketball accuracy is not forward-tested and preprocessing leaks information.** basketball/models.py fits the scaler on the complete X before train_test_split for win/loss and totals, then uses random stratified splits and ordinary cross validation. Enforce chronological train/validation/test, fit preprocessing only on past data, evaluate Brier/log-loss/ECE and score by market/season.

**BS-005 — Basketball confidence scale bug and theoretical odds.** basketball/models.py emits max(probability) on scale 0..1; basketball/prediction_service.py compares those confidence values against 75. That can exclude all legitimate picks. The same service estimates odds using 1/(probability*0.95), and treats spread as 1.91. These are NOT live bookmaker quotes. Standardize probability scale, forbid unverified basketball tickets, and test line-specific market ID/price fidelity.

**BS-006 — Legacy basketball API is disconnected.** api/api.py comments out basketball_predictions router. basketball/data_fetcher.py uses LeagueGameFinder as if it were an upcoming schedule; it references HOME_TEAM_ID, VISITOR_TEAM_ID and GAME_STATUS_TEXT fields that do not exist in the actual stored LeagueGameFinder team-game CSV. It ignores the supplied prediction date. Source of truth for upcoming matches must be a real scoreboard/schedule endpoint. Reconcile newer VPS-only basketball_* modules before activating GitHub basketball routes.

**BS-007 — No verified basketball settlement and bookmaker contract.** The current basketball tiers target arbitrary combined odds and an unchanging 220-point total without exact sportsbook lines. Add official game IDs, overtime-inclusive full-game settlement, push treatment for integer totals/spreads, observed quoted odds, bookmaker identity and complete readback before enabling booking.

## P1 high-priority findings

**BS-008 — General football ML trainer also uses a random holdout.** scripts/retrain_models.py:train_and_save randomly splits X and y. Point-in-time work DOES exist in calibration_harness.py, fit_calibration.py and backtest_dixon_coles.py, and automatic model promotion is disabled in services/training_pipeline_service.py. Nonetheless the general trainer's displayed accuracy is not a walk-forward performance guarantee. Make temporal testing a release gate.

**BS-009 — The checked-in NBA historical file is stale.** data/basketball/nba_games_raw.csv contains 6,934 team-game rows / 3,467 two-team games from seasons 2022-23 and 2023-24. First game: 2022-10-18; last: 2024-04-14. All checked game IDs have one valid home/away pair. It does not cover 2024-25, 2025-26 or live 2026-27 competition. Refresh the data, record provenance, run season completeness and cutoff checks.

**BS-010 — Provider outage is confused with no fixtures.** basketball/data_fetcher.py returns an empty dataframe on exceptions, while basketball/prediction_service.py can report status=success with zero games following a fetch error. Distinguish NO_GAMES from SOURCE_ERROR, STALE and MODEL_UNAVAILABLE, and alert operators.

**BS-011 — Real SportyBet contracts not yet validated.** sportybet_betting.py and services/booking_code_readback.py rely on undocumented bookmaker endpoints. Offline tests verify assumptions only, not the actual network schema. services/bookmaker_verification.py currently covers total-goals market 18 / 1.5 and 2.5, not all game markets. Fail closed on unknown readback outcomes or missing current price; never label theoretical odds as bookable.

**BS-012 — Legacy and new schedulers may race.** GitHub has fixture, prediction, booking-monitor, settlement and optional staging-training timer installers. VPS-only working tree has daily/weekly units and weekly_fixtures.py, daily_fixture_predict.py, fixture_store.py and report_store.py. Select one job owner, lock, schedule and storage source per purpose before installing any new timer.

**BS-013 — Production ingress remains unverified.** User-supplied VPS Gunicorn bound to 0.0.0.0:5050 with an older local Flask source. The new Flask operator-secret guard is not deployed. main.py sets CORS origins to wildcard; utils/security.py uses a process-local rate limiter based in part on user-agent. Configure HTTPS/SSH admin-only entry, scoped tokens, reverse-proxy/Redis rate limiting, TLS, secrets rotation, logging and service users.

**BS-014 — Multiple databases and mutable caches need a migration contract.** The app has SQLAlchemy DB, fixtures SQLite, JSON model and prediction caches, separate basketball/Telegram storage, and new VPS-only report/fixture stores. Specify canonical provider IDs, fixture-date semantics, append-only publication, backup/restore, and correction/adjudication policy.

**BS-015 — Automated QA coverage is incomplete.** .github/workflows/backend-ci.yml runs pytest, compileall, JavaScript syntax and limited flake8 critical errors, plus shell syntax checks. No direct basketball test modules, provider-contract simulations for VPS-only code, typed schema contract checks, dependency vulnerability scanner, secrets scan, coverage threshold or production load checks. Add tests for basketball feed errors, real-vs-synthetic provenance, probabilities, overtime/push settlement, and newly reconciled dashboard routes.

**BS-016 — Deployment configuration drift.** Procfile and render.yaml launch FastAPI main:app; live VPS runs Flask dashboard/app.py. render.yaml installs differently pinned dependencies than requirements.txt; current requirements.txt does not explicitly pin nba_api or sportsdataverse. Create a dedicated reproducible VPS lock and service configuration. Verify Gunicorn/Flask/FastAPI start modes separately.

## P2 follow-up findings

**BS-017 — Basketball categorical team codes are unstable.** pd.Categorical(...).codes is recomputed per data batch; identical teams can have different integer codes at training and inference. Use canonical NBA IDs and a versioned persisted encoder.

**BS-018 — Basketball total threshold is fixed at 220.0.** Real sportsbook totals are game-specific. Model full totals distribution conditional on actual price/line, not arbitrary 220.

**BS-019 — Results and Telegram state may disagree.** Manual code result updates in telegram_bot.py/telegram_db.py are separate from durable new settlement tables; link all to immutable ticket+leg identities, record override provenance.

**BS-020 — Model/data are tracked directly in Git.** 72 model artifacts and 172 data files inflate the repository and complicate safe production promotion. Use immutable, checksummed releases; exclude local calibration and private dataset artifacts from careless Git cleanup.

**BS-021 — Readiness labels/documentation overstate what is deployed.** Basketball README describes fully integrated API and training despite disabled router and synthetic fallbacks. Health checks are not a substitute for live fixture, booking, model drift and data-freshness checks. Give each feature explicit states: source coded, CI tested, real provider validated, staging verified, production healthy.

## Functional acceptance matrix

| Unit | Key acceptance gate |
| --- | --- |
| Fixture feeds | Canonical IDs, correct timezone, provider freshness, no duplicate/withdrawn games |
| Football modelling | Deterministic run with pinned model/calibrator, historical cutoff, temporal holdout |
| Daily publication | No started events, correct identity, immutable initial publication and reruns |
| Betting tiers | Unique legs, live market/odds exact IDs, no silent partial codes |
| Booking and refresh | Genuine current bookmaker board, exact readback, expiry and price-change tests |
| Jackpot | Verified current 12-game round, live scoring, replayable settlement |
| Rollover | Day transitions, bookability, interruption/resume, terminal results |
| Basketball ingestion | Live schedule source, correct game/team ID and pre-tipoff stats |
| Basketball model | No synthetic inputs, time-split training, calibrated exact-line predictions |
| Basketball result | Final-score regulation/overtime market policy, push, duplicates |
| Dashboard and API | All live features preserved, no duplicate/missing route, secure operator actions |
| Security | Secrets, TLS, host and ingress controls, shared rate limiting, SCA/SAST |
| Training/ops | Resource budgets, single timer owner, failure alarms, restore and rollback |
| Deployment | Clean conflict-free candidate, integrated CI, separate-port smoke, restorable model/DB state |

## Where to obtain NBA and basketball data (checked 23 Oct 2026)

- **nba_api** (https://github.com/swar/nba_api) — free unofficial Python client for NBA.com boxscore, schedules/live feeds and historical stats. Subject to NBA.com access limits and usage rights.
- **SportsDataverse Python / hoopR** (https://github.com/sportsdataverse/sportsdataverse-py and https://hoopr.sportsdataverse.org) — community NBA/ESPN/NCAA/WNBA histories and loaders. Use as supplementary research/backfill, check distribution terms.
- **BALLDONTLIE NBA** (https://docs.balldontlie.io/) — free teams/players/games at 5 requests/min; paid NBA ALL-STAR approximately $9.99/month includes player stats/injuries; GOAT approximately $39.99/month adds advanced boxscore and betting odds. Sports priced separately.
- **API-SPORTS API-NBA** (https://api-sports.io/sports/nba) — free tier advertises 100 API requests/day; PRO about $15 with 7,500 daily requests. Good alternative for a single commercial NBA data feed.
- **API-SPORTS API-BASKETBALL** (https://api-sports.io/sports/basketball) — advertises 427 leagues/cups, including African competitions. Verify each league/season's actual stats, schedule and odds coverage; no assumption all 427 have full features.
- **TheOddsAPI** (https://theoddsapi.com/pricing) — free about 25 NBA moneyline requests/day; Pro about $29/month includes spreads/totals and 20,000 monthly calls; Business about $99/month adds props and historical data. These odds are not SportyBet share codes.
- **Sportradar** (https://developer.sportradar.com/basketball/reference/nba-overview) — official NBA provider and wider WNBA/NCAA/G-League feeds, enterprise licensing/pricing inquiry.

Check provider commercial display/redistribution rights separately from API access and reliability. Public prices and plan details may change.

### Recommended basketball rollout
1. NBA only first; introduce one provider adapter for canonical schedules/finals plus nba_api or SportsDataverse for historical enrichment.
2. Load 2024-25, 2025-26 and rolling 2026-27 historical game/player/team data into versioned tables, with origin, UTC observation timestamp and quality indicators.
3. Compute pre-tipoff rolling form, pace, offensive/defensive ratings, shooting, rest, travel and injury data; fail closed rather than making random numbers.
4. Train moneyline and over/under total distributions with chronological splits, calibration and an untouched season test. Keep real bookmaker quotes as a separate timestamped feed.
5. Only publish basketball tier slips when actual game, market, handicap, outcome, price and settlement rules can be independently validated. Never label a model-derived fair price as placeable.
6. Expand to WNBA, NCAA, EuroLeague or African leagues only after league-level coverage and model-validation gates.

## Engineering order and go/no-go decision

1. **No live switch** until the three VPS merge conflicts are resolved and locally tuned models are protected.
2. Recover newer VPS basketball pipeline for inspection and compare it to these legacy GitHub findings; do not assume it is either safe or unsafe without seeing it.
3. Add fail-closed NBA ingestion and testable real-data-only feature/target pipeline.
4. Add walk-forward training, real price verification, settlement and per-market performance reports.
5. Consolidate schedulers, backups, shared auth/rate limiting and production monitoring.
6. Run unified integrated CI on the reconciled candidate, verify private model/data backups, test parallel-port candidate, then deploy with a rollback plan.

**Audit documents identified gaps in the checked GitHub tree. It does not prove the absence or presence of fixes in the unmerged private VPS worktree.**
