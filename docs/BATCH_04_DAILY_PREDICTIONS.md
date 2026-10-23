# Batch 4 — Daily model predictions and guarded publication

**Scope: the independent Flask/VPS BetSightly instance on port 5050 only.**

## Publication contract

The fixture collector (Batch 3) first populates the SQLite store with
source event IDs, dates and statuses. Batch 4 then:

1. Reads saved fixtures for the requested UTC fixture day.
2. Stages a JSON fixture file in a temporary directory.
3. Invokes the existing model via predict_today.py using --fixture-file,
   --save, --no-cache and --require-history.
4. Validates exact event IDs and team names, the 1X2 pick, probabilities,
   quality labels and complete coverage for matches yet to kick off.
5. Preserves any existing published prediction unchanged, including fixtures
   already started or settled; adds only predictions for newly available
   fixtures before their kickoff.
6. Publishes to cache/apifootball/predictions_DATE.json with atomic replacement
   **only after successful checks**; archives any previous snapshot.
7. Writes run outcomes to SQLite table prediction_publish_runs; failures
   leave the last good public JSON intact.

A new prediction is tagged recommendation_eligible=true only when quality
is HIGH or MEDIUM and confidence is >=60%. Lower confidence predictions can
still be displayed by legacy clients, which do not enforce this flag. These
are model estimates, **not verified bookmaker odds or guaranteed outcomes**.
The bookmaker booking path needs separate validation in Batch 5.

Unlike the old daily_predict.py, this runner **never restarts the dashboard**.

## Commands

Run against collected fixtures using the project's Python environment:

    python -m pytest tests/test_daily_prediction_publisher.py -q
    python scripts/publish_daily_predictions.py run
    python scripts/publish_daily_predictions.py run --offset 1 --import-if-missing
    python scripts/publish_daily_predictions.py status

The preview command generates predictions for tomorrow using the saved fixture
store; if that day has no stored fixtures, it attempts a LiveScore import.

The old standalone predict_today.py retains its existing manual behavior unless
you explicitly supply --fixture-file and --no-cache.

## Proposed VPS schedule (Africa/Lagos)

| Job | Lagos time | Outcome |
| --- | --- | --- |
| Preview | 21:30 every day | Generate tomorrow's pre-match card |
| Early | 00:30 every day | Add new fixtures to today's published card |
| Refresh | 05:45 every day | Add any fixtures collected by the 05:15 fixture refresh |

Each job uses a shared file lock with the Batch 3 collector so source cache
updates cannot race against prediction runs. Timers are persistent after server
outages. Review machine capacity: the existing model loads substantial
historical data and may require many minutes. Never promise response times
before VPS benchmarking.

**DO NOT INSTALL until reconciling the live VPS code, permissions, Python
environment and storage mount.** This standalone installer does not change
Flask's running service but enables scheduled writes to its cache.

    sudo bash deploy/install_prediction_timers.sh /absolute/repo/path /absolute/python/path VPS_SERVICE_USER
    sudo systemctl list-timers 'betsightly-predict-*'
    sudo journalctl -u betsightly-predict-early.service -n 70 --no-pager

To test safely before enabling -- run the parser and offline tests, inspect the
stored fixture count, back up existing predictions cache, then execute one
manually supervised model run. Use the current live process's code revision.

## Remaining

- Verify that the actual port-5050 process reads these same filesystem paths.
- Validate real provider IDs and timezone boundary between UTC fixture dates and
  Lagos-local user dates, especially fixtures around midnight.
- Confirm live import, model performance and calibration on the actual VPS.
- Add publication job status to the newer live dashboard after identifying its
  deployed source files.
- Batch 5: exact SportyBet fixture/odds/code validation and tier generation.
- Batch 6: reliable settlement and immutable historical performance metrics.

Code and tests being green does not imply the VPS scheduler is installed or
that daily production predictions have already been delivered.
