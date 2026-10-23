"""Track issued bookmaker codes and refresh quotes without placing a wager.

Run as a VPS timer after verifying the real deployed checkout. A changed odds
signature triggers a fresh SportyBet share-code request; invalid or started
events are marked unavailable, never fabricated or reissued.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from services.bookmaker_verification import (
    BookingValidationError, ensure_not_expired, verify_and_reprice,
)


def refresh_tracked_codes(
    cache: dict, board: list[dict], *,
    issue_code, now_ms: int | None = None,
) -> dict:
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    outcomes = {}
    checked = renewed = unchanged = unavailable = failed = 0
    for identity, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        selections = entry.get("selections")
        if not isinstance(selections, list) or not selections:
            continue  # legacy cache lacking a verifiable snapshot
        checked += 1
        try:
            expected_legs = []
            for selection in selections:
                start_ms = selection.get("start_ms")
                if start_ms:
                    from datetime import datetime, timezone
                    dt = datetime.fromtimestamp(int(start_ms) / 1000, timezone.utc)
                    day, clock = dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
                else:
                    day, clock = "", ""
                expected_legs.append({
                    "home": selection.get("home", ""),
                    "away": selection.get("away", ""),
                    "match_date": day,
                    "kick_off_utc": clock,
                })
            quote = verify_and_reprice(
                selections, board, expected_legs=expected_legs,
                now_ms=now_ms,
            )
            if (entry.get("quote_signature") == quote["quote_signature"]
                    and ensure_not_expired(entry, now_ms)):
                unchanged += 1
                outcomes[identity] = {
                    "status": "unchanged",
                    "share_code": entry.get("share_code"),
                    "total_odds": quote["odds"],
                    "quote_checked_at_ms": now_ms,
                }
                continue
            result = issue_code(quote)
            renewed += 1
            outcomes[identity] = {
                "status": "renewed",
                "share_code": result["share_code"],
                "total_odds": quote["odds"],
                "quote_checked_at_ms": now_ms,
            }
        except BookingValidationError as exc:
            unavailable += 1
            outcomes[identity] = {
                "status": "unavailable", "reason": str(exc),
                "quote_checked_at_ms": now_ms,
            }
        except Exception as exc:
            failed += 1
            outcomes[identity] = {
                "status": "error", "reason": str(exc)[:300],
                "quote_checked_at_ms": now_ms,
            }
    return {
        "checked": checked, "renewed": renewed,
        "unchanged": unchanged, "unavailable": unavailable,
        "failed": failed, "tickets": outcomes,
    }


def run_once(*, save_status_path: Path, now_ms: int | None = None) -> dict:
    import sportybet_betting as sbb
    from services.weekly_fixture_collector import _atomic_json

    cached = sbb.codes_cache()
    active = {
        key: value for key, value in cached.items()
        if isinstance(value, dict) and value.get("selections")
        and ensure_not_expired(value, now_ms)
    }
    if not active:
        output = {"checked": 0, "renewed": 0, "unchanged": 0,
                  "unavailable": 0, "failed": 0, "tickets": {},
                  "reason": "no active tracked codes"}
        _write_status(save_status_path, output)
        return output

    # Required: fresh upstream board. A stale local cache is NOT proof of price.
    board = sbb.fetch_odds_table(save=True)
    def issue(quote):
        return sbb.booking_code_for(
            quote["selections"], use_cache=False,
            quote_signature=quote["quote_signature"],
            deadline_ms=quote["deadline_ms"],
        )
    output = refresh_tracked_codes(active, board, issue_code=issue, now_ms=now_ms)
    _write_status(save_status_path, output)
    return output


def _write_status(path: Path, status: dict) -> None:
    from services.weekly_fixture_collector import _atomic_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic JSON replacement; no API credentials or account data is stored.
    _atomic_json(path, status)
