"""Strict, price-aware validation for independent BetSightly SportyBet codes.

A response from SportyBet's share API establishes that a share code was issued,
not that every selection was independently read back from the bookmaker.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone


class BookingValidationError(ValueError):
    """A requested bookmaker slip is not currently verifiable/bookable."""


OPEN_STATUSES = {"", "not start", "not started", "notstart", "ns", "upcoming"}
CLOSED_BOOKING_STATUSES = {
    "closed", "suspended", "suspend", "stop", "disabled", "inactive", "unavailable",
}
MAX_KICKOFF_MISMATCH_SECONDS = 15 * 60


def normalized_pair(home: str, away: str) -> tuple[str, str]:
    from sportybet_filter import normalize_name
    return normalize_name(str(home or "")), normalize_name(str(away or ""))


def _is_bookable_event(event: dict, now_ms: int) -> bool:
    try:
        start = int(event.get("start_ms") or 0)
    except (TypeError, ValueError):
        return False
    if start <= now_ms or event.get("live"):
        return False
    status = str(event.get("status") or "").strip().lower()
    if status not in OPEN_STATUSES:
        return False
    booking = str(event.get("booking") or "").strip().lower()
    if booking in CLOSED_BOOKING_STATUSES:
        return False
    return True


def _selected_market(event: dict, selection: dict) -> tuple[float, dict]:
    market_id = str(selection.get("market_id") or "")
    specifier = str(selection.get("specifier") or "")
    outcome_id = str(selection.get("outcome_id") or "")
    if market_id != "18":
        raise BookingValidationError("Unsupported market ID; expected Over/Under 18")
    fields = {"total=1.5": "over_1.5", "total=2.5": "over_2.5"}
    field = fields.get(specifier)
    if not field:
        raise BookingValidationError("Unsupported total-goals specifier")
    price = (event.get("prices") or {}).get(field)
    if not price or str(price.get("market_id")) != market_id or str(price.get("specifier")) != specifier:
        raise BookingValidationError("Selected market is no longer offered")
    if str(price.get("outcome_id")) == outcome_id:
        raw_odds = price.get("odds")
    elif str(price.get("under_outcome_id") or "") == outcome_id:
        raw_odds = price.get("under_odds")
    else:
        raise BookingValidationError("Selected outcome is no longer offered")
    try:
        odds = float(raw_odds)
    except (TypeError, ValueError):
        raise BookingValidationError("Bookmaker outcome has no valid price")
    if not math.isfinite(odds) or odds <= 1.0:
        raise BookingValidationError("Bookmaker outcome has no valid price")
    return odds, price


def verify_and_reprice(
    selections: list[dict], live_board: list[dict], *,
    expected_legs: list[dict] | None = None, now_ms: int | None = None,
) -> dict:
    """Require exact event IDs, live market/outcome IDs and unstarted games.

    Never match on names alone at booking time. Never rely on ledger/live-board
    snapshots as evidence that a *new* code remains bookable after kickoff.
    """
    if not isinstance(selections, list) or not selections:
        raise BookingValidationError("Ticket has no selections")
    if expected_legs is not None and len(expected_legs) != len(selections):
        raise BookingValidationError("Displayed ticket and requested selections differ")
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if not isinstance(live_board, list):
        raise BookingValidationError("Live bookmaker board is unavailable")
    by_id: dict[str, dict] = {}
    for event in live_board:
        event_id = str(event.get("event_id") or "")
        if not event_id:
            continue
        if event_id in by_id:
            if by_id[event_id].get("prices") != event.get("prices"):
                raise BookingValidationError("Conflicting bookmaker records for event " + event_id)
            continue
        by_id[event_id] = event

    seen_events = set()
    quoted = []
    for index, selection in enumerate(selections):
        event_id = str(selection.get("event_id") or "")
        if not event_id or event_id in seen_events:
            raise BookingValidationError("Missing or repeated SportyBet event ID")
        seen_events.add(event_id)
        event = by_id.get(event_id)
        if not event:
            raise BookingValidationError(
                "Event " + event_id + " is no longer on SportyBet's upcoming board"
            )
        if not _is_bookable_event(event, now_ms):
            raise BookingValidationError("SportyBet event is started, closed, or unavailable: " + event_id)
        if expected_legs is not None:
            leg = expected_legs[index]
            home, away = normalized_pair(leg.get("home"), leg.get("away"))
            board_home, board_away = normalized_pair(event.get("home"), event.get("away"))
            if not home or not away or (home, away) != (board_home, board_away):
                raise BookingValidationError("Home/away fixture mismatch for " + event_id)
            date_str = str(leg.get("match_date") or "")
            kickoff = str(leg.get("kick_off_utc") or "")
            if kickoff:
                try:
                    expected_minute = int(kickoff[:2]) * 60 + int(kickoff[3:5])
                    actual = datetime.fromtimestamp(int(event["start_ms"]) / 1000, timezone.utc)
                    delta = abs(expected_minute - (actual.hour * 60 + actual.minute))
                    if min(delta, 1440 - delta) > MAX_KICKOFF_MISMATCH_SECONDS // 60:
                        raise BookingValidationError("Kickoff mismatch for " + event_id)
                    if date_str:
                        expected_ms = int(datetime.strptime(
                            date_str + " " + kickoff[:5], "%Y-%m-%d %H:%M"
                        ).replace(tzinfo=timezone.utc).timestamp() * 1000)
                        if abs(expected_ms - int(event["start_ms"])) > MAX_KICKOFF_MISMATCH_SECONDS * 1000:
                            raise BookingValidationError("Kickoff date mismatch for " + event_id)
                except BookingValidationError:
                    raise
                except (ValueError, IndexError):
                    raise BookingValidationError("Invalid fixture kickoff timestamp")
        odds, _market = _selected_market(event, selection)
        quoted.append({
            "event_id": event_id,
            "market_id": str(selection["market_id"]),
            "specifier": str(selection.get("specifier") or ""),
            "outcome_id": str(selection["outcome_id"]),
            "odds": odds,
            "home": event.get("home", ""),
            "away": event.get("away", ""),
            "start_ms": int(event["start_ms"]),
        })
    total = math.prod(q["odds"] for q in quoted)
    payload = [
        [q["event_id"], q["market_id"], q["specifier"], q["outcome_id"], round(q["odds"], 4), q["start_ms"]]
        for q in quoted
    ]
    signature = hashlib.sha256(
        json.dumps(sorted(payload), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"selections": quoted, "odds": round(total, 3),
            "quote_signature": signature, "checked_at_ms": now_ms,
            "deadline_ms": min(q["start_ms"] for q in quoted)}


def ensure_not_expired(saved: dict, now_ms: int | None = None) -> bool:
    """A cached code cannot be represented as current after first kickoff."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    return isinstance(saved, dict) and int(saved.get("deadline_ms") or 0) > now_ms
