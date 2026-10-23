# Batch 3 — Sunday fixtures and daily refresh (independent VPS app)

This pipeline is for the standalone dashboard hosted on port 5050, **not betsightly.com**.

## What this adds

- Sunday collection of the current Sunday-through-Saturday (7 UTC fixture dates).
- Daily refresh of the current date in **Africa/Lagos**.
- Canonical LiveScore match IDs stored idempotently in a local SQLite database at data/fixtures/fixture_store.sqlite3.
- Status and kickoff updates without duplicating matches.
- Finished fixtures are not downgraded by older source responses.
- Per-day job results (including failures and rejection counts) in fixture_import_runs.
- Failed/empty source responses never erase old fixture data or cached match cards.
- Optional atomic JSON publication to cache/apifootball/events_YYYY-MM-DD.json.
- Offline regression tests, requiring no paid API credentials.

**Safety:** The live VPS frontend does not match the older GitHub dashboard layout.
Reconcile that live code before merging/deploying. This collector does not write
shared dashboard cache files unless you pass --publish-cache.

## CLI checks

Run from the repository root with the project's Python:

    python -m pytest tests/test_weekly_fixture_collector.py -q
    python scripts/fixture_collection.py weekly --start 2026-10-25
    python scripts/fixture_collection.py status --start 2026-10-25

To refresh today's fixtures and publish the compatible cache:

    python scripts/fixture_collection.py daily --publish-cache

A failed date returns nonzero exit status, leaves previous fixtures intact and
is recorded in the import log. The other days in a weekly collection continue.

The collector uses fixtures_livescore.fetch_day(day, use_cache=False), so it
attempts a fresh pull even if old raw LiveScore data exists.
Match source times stay in UTC; job scheduling uses Africa/Lagos local time.

## Optional installation on the independent Linux VPS

**Do not install until the live revision, repo working directory, service user,
and persistent storage have been verified.** The service user needs write
permissions to data/fixtures and cache/apifootball.

    sudo bash deploy/install_fixture_timers.sh /absolute/repo/path /absolute/python/path VPS_SERVICE_USER

Installed timers:

| Job | Lagos time | Behavior |
| --- | --- | --- |
| Weekly | Sundays 01:15 | Current Sunday-Saturday, 7 dates |
| Daily | Every day 05:15 | Refresh today's date |

Both have Persistent=true to recover missed runs. A Monday catch-up weekly run
starts on the immediately preceding Sunday, not the following week's Sunday.
An exclusive flock lock prevents overlap with daily refresh.

Inspect or manually test after supervised installation:

    sudo systemctl list-timers 'betsightly-fixtures-*'
    sudo systemctl start betsightly-fixtures-daily.service
    sudo journalctl -u betsightly-fixtures-daily.service -n 70 --no-pager

## Backup and limitations

Back up the SQLite fixture database using SQLite's online backup mechanism when
live. Do not delete preexisting prediction JSON caches or model artifacts.
GitHub Actions will **not** install timers or update the VPS automatically.

This is an isolated fixture collection/storage phase. It does not run models,
generate predictions or settle results; those are separate batches.
Before production use confirm LiveScore's UTC/day boundaries and make sure
the live dashboard and prediction runner accept the same event IDs/schema.
