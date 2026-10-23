# Standalone BetSightly — GitHub completion and VPS handoff

Repository: black-jamesbond/betsightly-backend-devil
Runtime target: independent dashboard on port 5050. NOT betsightly.com.

## Review stack and acceptance

The following PRs are **stacked**, so review/merge sequentially. They are
not deployments and have not accessed the running VPS.

1. PR #1 — CI, dashboard loading and Lagos date handling.
2. PR #2 — Sunday seven-day fixture ingestion + daily refresh + SQLite.
3. PR #3 — guarded daily model prediction staging and publication.
4. PR #4 — verified bookmaker identity, share codes and automatic renewal.
5. PR #5 — conservative final-score settlement, voids, accurate reports.
6. PR #6 — operational security, health, training staging, stale jackpot guard.

### Cross-batch integration fixes (release hardening)

- A successful fixture collection records the currently active source IDs in
  SQLite table `active_fixture_ids`. Historical `collected_fixtures` rows remain
  preserved, but new predictions read **only the active IDs** so a withdrawn
  fixture cannot reappear in a new daily prediction card.
- Empty/failed fixture imports preserve the last successful active IDs.
  Before enabling a prediction runner on an existing SQLite file, perform at
  least one successful weekly/daily import to populate this new table.
- Repeated settlement runs keep exactly one audited/unverified prediction
  record per fixture, even if upstream loses kickoff-time evidence.
- The bookmaker-code cache is written atomically under an interprocess lock;
  code retrieval requests from different Flask workers and the renewal timer
  cannot clobber one another. Quotes older than 15 minutes without a successful
  recheck are labelled `stale_quote` and their quick-open links are withheld.
- The legacy Flask dashboard result cache now uses the same 90-minute
  adjudication as the durable settlement service. LiveScore AET/penalties
  never silently overwrite regulation-time results; only explicit FT can be
  folded from the live score API into the old dashboard's predictions view.
- Offline end-to-end regression tests exercise collection, withdrawal,
  publication, 90-minute settlement and rerun idempotency.

### Automated checks

- Python unit tests under tests/
- Python compileall and critical undefined-name/syntax lint
- Node --check for Flask dashboard inline scripts (Jinja normalized)
- No dependence on a real betting account or wager placement in tests

**Passing CI means source code checks passed.** It does not prove uptime,
LiveScore or SportyBet response compatibility, booking-code readback, model
quality, financial performance or that the VPS timers have been installed.

## VPS deployment prerequisites

Do not deploy this repository onto the current VPS working directory until
its newer live code is inspected. The VPS UI has a different tab layout from
GitHub main (including Rollover, Fixtures and Daily Report); a blind checkout
would delete features.

1. Capture the running process and actual app path using the read-only
   deploy/audit_vps_baseline.sh (no credentials or secrets included).
2. Back up application code, model artifacts, cache, SQLite and database
   before any checkout or merge.
3. Diff the running server source against the release branch; reconcile
   the newer live UI/features, not overwrite them.
4. Validate independent LiveScore fixture IDs and UTC-to-Lagos dates.
5. Validate SportyBet event/market IDs and both share and GET-code readback.
   The API is undocumented; failures should be visible, not hidden.
6. Run a single supervised fixture import, prediction generation and
   settlement against a small current-day set; keep original published picks.
7. Create a 32+-character random BETSIGHTLY_DASHBOARD_ADMIN_TOKEN in the VPS
   secret manager (not in Git or chat); set API_KEY in production too.
8. Put administration behind HTTPS reverse proxy or an SSH localhost tunnel.
   The public HTTP :5050 listener must never receive operator credentials.
9. Only then install/enable scheduled systemd units:
   - weekly fixtures Sunday 01:15 WAT
   - daily fixtures 05:15 WAT
   - tomorrow preview 21:30 WAT
   - early card 00:30 WAT, refresh 05:45 WAT
   - booking share-code monitoring every 10 minutes
   - settlement every 3 hours for last 3 days
   - OPTIONAL heavy candidate training Sunday 03:30 WAT, with resource checks
10. Inspect GET /api/operations/health and settlement-report results.
11. Restart the actual service manager once; verify timers, jobs and
    published data survive a controlled restart.
12. Roll back safely if dashboard features, booking fidelity or
    prediction integrity regress.

## Guardrails and intentionally unsolved dependencies

- A code marked readback_verified has its market/outcome/event identities
  checked against an undocumented GET response; API shape must be verified
  on real SportyBet responses before claiming production success.
- Real jackpot round auto-discovery is NOT complete: no verified public
  Sporty 12 fixture XHR endpoint is known. Old snapshots are now rejected
  instead of being shown as current. The manual snapshot importer is validated.
- Live-only Rollover implementation cannot be reconciled from the older GitHub
  main tree. Source must be retrieved from the VPS first.
- Model promotion stays disabled until time-based out-of-sample backtesting
  and review. Staged training outputs are measured but the current trainer
  uses a random holdout, which is not enough to certify financial performance.
- Historical predictions without trustworthy pre-kickoff timestamps are
  explicitly kept out of audited model accuracy.
- Public port-5050 authentication and HTTPS termination depend on real
  reverse proxy/server configuration.
- Bot and ticket reports are outcome summaries, not proof of placed bets,
  realized profit, or SportyBet financial settlement.

## Read-only verification

    python -m pytest tests/ -q
    python scripts/check_dashboard_js.py
    python scripts/fixture_collection.py status
    python scripts/publish_daily_predictions.py status
    python scripts/settle_results.py --status

Keep this document as a release gate: finish source QA first, and do not
equate review-branch completion with production completion.
