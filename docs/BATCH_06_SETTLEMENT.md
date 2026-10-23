# Batch 6 — Automated settlement and trustworthy results

**Scope:** independent black-jamesbond repository / port-5050 dashboard only.

## What this implements

The scheduled settlement runner reads freshly fetched LiveScore fixtures,
previously published predictions, bot picks and frozen accumulator snapshots,
then records the outcomes in SQLite rather than editing historical picks.

- Idempotent settlement by canonical source fixture IDs.
- Only confirmed 90-minute FT results count. Extra-time/penalty results
  without separately recorded regulation-time scores remain pending.
- Cancelled, abandoned, postponed and awarded results are VOID.
- Suspicious contradictory final scores are logged as conflicts and not
  silently overwritten (review needed).
- Live and in-play scores never settle anything.
- Prediction accuracy is grouped by verified pre-kickoff publication time.
  Legacy or after-kickoff cards are separated as prediction_unverified.
- Bot outcomes and frozen accumulator ticket legs are stored separately.
- A ticket with any losing leg is LOSS; all won is WIN; a void with no loss
  remains VOID; otherwise unresolved tickets remain PENDING.
- Outcomes are auditable without inventing stakes or returns.
- Failed external API reads report stale fallback, not success.

SQLite data tables:
- settled_fixtures
- settled_selections
- settled_tickets
- settlement_runs

## Commands

    python -m pytest tests/test_settlement_service.py -q
    python scripts/settle_results.py --status
    python scripts/settle_results.py --days 3
    python scripts/settle_results.py --date 2026-10-23

Generated read-only daily report:

    cache/reports/settlement_YYYY-MM-DD.json

Flask dashboard route:

    GET /api/settlement-report?date=YYYY-MM-DD

## Suggested VPS timer

A three-hourly systemd schedule reconciles today and the preceding two days:

    sudo bash deploy/install_settlement_timer.sh /absolute/repo /absolute/python SERVICE_USER
    sudo journalctl -u betsightly-settlement.service -n 70 --no-pager

Do not execute on VPS until its deployed working tree and cache paths are
reconciled. The installer is committed to GitHub but NOT active anywhere.

### Important data limitations

Historical prediction cards published by the previous implementation have no
trusted publication timestamp, and some were generated after kickoff. They can
be compared against results but must not count toward audited pre-match model
accuracy. Likewise, synthetic/fair odds are never treated as bookmaker-confirmed
returns. A SportyBet-only event with no canonical LiveScore match ID remains
unsettled rather than matched by mere team names.

Bookmaker settlement may require adjusting void/cancelled treatment to the
actual operator rules. Review adjudicated fixtures manually before reporting
financial outcomes. This component deliberately reports wins, losses and voids
without claiming realized money won.

## Remaining

1. VPS revision diff and safe deployment (no guessing of live source).
2. Real external feed and 90-minute scoring checks for cup fixtures.
3. End-to-end reload and recovery tests (Batch 7).
4. Complete jackpot round ingestion and independently deployed rollover UI
   reconciliation from the newer live server code.
