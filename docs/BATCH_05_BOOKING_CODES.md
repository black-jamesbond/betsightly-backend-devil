# Batch 5 — SportyBet booking-code integrity and automated renewal

**Standalone BetSightly VPS/dashboard on port 5050 only.** This does not affect betsightly.com.

## What was fixed

Previously, tickets could reuse an old code indefinitely, report bookmaker
availability when prices could not be fetched, silently omit requested group
games, match a reversed home/away fixture, or claim "bookable" when the event
had already kicked off.

The stricter path now:

1. Resolves requested legs to exact SportyBet event IDs (not just team names).
2. Requires a fresh upcoming bookmaker board at issuance time.
3. Compares normalized home/away teams and known UTC kickoff.
4. Validates market, specifier, outcome, active status and current price.
5. Refuses duplicate events, unavailable/started games, bad prices and partial group tickets.
6. Computes an odds-aware signature; changes to any selected price or identifier
   trigger a new SportyBet code request.
7. Asks the bookmaker's internal readback endpoint to load the issued code and
   compares its exact event/market/outcome selections. A code is tagged
   readback_verified **only** if every selection matches exactly.
8. If the undocumented readback endpoint fails, a returned code is tagged
   issued_unverified and is NEVER represented as independently verified.
9. Stores the current code, quote fingerprint, issuance timestamp and earliest
   kickoff in cache/sportybet_codes.json. Legacy cache entries cannot be
   automatically treated as valid current quotes.

SportyBet has no officially supported public developer API; these internal
endpoints may change. **A booking code is a shareable betslip reservation, not a
placed wager.** This project never stakes funds.

## Automatic code monitoring

Only previously issued, still-upcoming tracked codes are checked:

    python scripts/refresh_booking_codes.py

The script fetches the current bookmaker board, compares each tracked ticket's
price signature, and requests a replacement code only on a change. If an event
starts, disappears, or an outcome closes, it marks that code unavailable.
Network errors are reported rather than silently represented as fresh prices.

Results are written atomically to:

    cache/booking_monitor_status.json

The dashboard provides GET /api/booking/status?ticket_id=<opaque ID> for each
tracked code. When displayed, it checks for renewed/invalidated codes every
60 seconds and updates the visible result.

**Tentative VPS schedule:** every 10 minutes, using a systemd timer:

    sudo bash deploy/install_booking_monitor_timer.sh /absolute/repo /absolute/python VPS_SERVICE_USER
    sudo systemctl list-timers 'betsightly-booking-monitor*'
    sudo journalctl -u betsightly-booking-monitor.service -n 70 --no-pager

Do not run the installer until confirming that the port-5050 server uses this
source checkout and its associated cache directory. The installer has not
been run on the VPS.

## Verified vs issued states

- readback_verified: SportyBet issued the code AND readback contained exactly
  the expected market/outcome/event IDs.
- issued_unverified: SportyBet issued a code, but independent readback could
  not be confirmed; display it with the warning.
- unavailable/expired: The tracked ticket is no longer safely bookable.
- error: The upstream provider could not be checked; don't promise freshness.

These are validation states at check time. SportyBet can still change odds or
availability before someone loads the code.

## Remaining work

- Reconcile GitHub with the newer live Flask dashboard before merging.
- Verify live SportyBet API shape, rate limits, and a real code+readback on VPS.
- Confirm installer timezone support and execution permissions on actual Linux.
- Finish **rollover and jackpot** reliability separately, including the unknown
  SportyBet jackpot automatic round feed and the newer live rollover code that
  is absent from GitHub main.
- Batch 6: settlement and immutable historical statistics.
- Batch 7: authentication and rate limiting on the currently exposed booking
  generation endpoints, production stress/restart and monitoring verification.

Unit tests use mocked API records, not live wagers or authentic share codes.
