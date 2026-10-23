"""Strict comparison of SportyBet share-code readback with the requested legs.

The endpoint is internal/undocumented. If it fails or changes shape, this
module refuses to claim a booking was independently verified.
"""
from __future__ import annotations

from services.bookmaker_verification import BookingValidationError


def requested_id_set(selections: list[dict]) -> set[tuple[str, str, str, str]]:
    rows = set()
    for s in selections:
        identity = (
            str(s.get("event_id") or ""),
            str(s.get("market_id") or ""),
            str(s.get("specifier") or ""),
            str(s.get("outcome_id") or ""),
        )
        if not all((identity[0], identity[1], identity[3])) or identity in rows:
            raise BookingValidationError("Invalid or duplicate requested booking leg")
        rows.add(identity)
    return rows


def readback_id_set(readback: dict) -> set[tuple[str, str, str, str]]:
    if not isinstance(readback, dict) or str(readback.get("bizCode")) != "10000":
        raise BookingValidationError("Bookmaker share-code readback unsuccessful")
    data = readback.get("data") or {}
    if data.get("unavailableOutcomes"):
        raise BookingValidationError("Bookmaker reports unavailable outcomes")
    legs = data.get("outcomes")
    if not isinstance(legs, list) or not legs:
        raise BookingValidationError("Bookmaker readback contains no legs")
    results = set()
    count = 0
    for leg in legs:
        event_id = str(leg.get("eventId") or "")
        markets = leg.get("markets") or []
        if not event_id or not markets:
            raise BookingValidationError("Readback leg lacks an event or market")
        for market in markets:
            market_id = str(market.get("id") or "")
            specifier = str(market.get("specifier") or "")
            outcomes = market.get("outcomes") or []
            if not market_id or not outcomes:
                raise BookingValidationError("Readback market lacks an outcome")
            for outcome in outcomes:
                outcome_id = str(outcome.get("id") or "")
                if not outcome_id:
                    raise BookingValidationError("Readback outcome lacks identity")
                count += 1
                results.add((event_id, market_id, specifier, outcome_id))
    if count != len(results):
        raise BookingValidationError("Bookmaker readback contains duplicate selections")
    return results


def verify_code_readback(
    code: str, original: list[dict], readback: dict,
) -> dict:
    if str((readback.get("data") or {}).get("shareCode") or "").upper() != code.upper():
        raise BookingValidationError("Bookmaker readback returned a different share code")
    requested = requested_id_set(original)
    received = readback_id_set(readback)
    if requested != received:
        raise BookingValidationError(
            f"Booking code has {len(received)} different legs; "
            f"expected exactly {len(requested)}"
        )
    return {"verification": "readback_verified",
            "verified_legs": len(received)}
