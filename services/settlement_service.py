"""Durable 90-minute football result settlement for the standalone dashboard.

Never changes published prediction JSON or bookmaker tickets. Results and their
provenance are stored separately and can be rechecked without double-counting.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from services.weekly_fixture_collector import utc_now

FINISHED = {"finished", "ft", "full time", "full-time"}
EXTRA_TIME = {"aet", "after et", "after extra time", "after pen.", "pen.", "after penalties", "ft pen."}
VOID = {"cancelled", "canceled", "postponed", "abandoned", "suspended", "awarded", "canc.", "postp."}
PENDING = {"not started", "ns", "in play", "ht", "half time", "", "tbd"}
RULES = {"Home Win", "Draw", "Away Win", "Over 1.5", "Under 1.5",
         "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5",
         "BTTS Yes", "BTTS No"}

DDL = [
    """CREATE TABLE IF NOT EXISTS settled_fixtures (
        source TEXT NOT NULL, source_id TEXT NOT NULL, match_date TEXT NOT NULL,
        result_status TEXT NOT NULL, home_score INTEGER, away_score INTEGER,
        raw_status TEXT NOT NULL, source_payload TEXT NOT NULL,
        updated_utc TEXT NOT NULL, PRIMARY KEY(source,source_id))""",
    """CREATE TABLE IF NOT EXISTS settled_selections (
        ticket_kind TEXT NOT NULL, ticket_id TEXT NOT NULL, leg_id TEXT NOT NULL,
        match_date TEXT NOT NULL, source_id TEXT NOT NULL, market TEXT NOT NULL,
        outcome TEXT NOT NULL, home_score INTEGER, away_score INTEGER,
        updated_utc TEXT NOT NULL,
        PRIMARY KEY(ticket_kind,ticket_id,leg_id))""",
    """CREATE TABLE IF NOT EXISTS settled_tickets (
        snapshot_mode TEXT NOT NULL, ticket_id TEXT NOT NULL,
        match_date TEXT NOT NULL, outcome TEXT NOT NULL,
        win_legs INTEGER NOT NULL, loss_legs INTEGER NOT NULL,
        void_legs INTEGER NOT NULL, pending_legs INTEGER NOT NULL,
        updated_utc TEXT NOT NULL,
        PRIMARY KEY(snapshot_mode,ticket_id))""",
    """CREATE TABLE IF NOT EXISTS settlement_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_date TEXT NOT NULL, started_utc TEXT NOT NULL,
        completed_utc TEXT NOT NULL, status TEXT NOT NULL,
        events INTEGER NOT NULL, settled INTEGER NOT NULL,
        pending INTEGER NOT NULL, void INTEGER NOT NULL,
        error TEXT NOT NULL DEFAULT '')""",
]

def init(db: sqlite3.Connection) -> None:
    for ddl in DDL:
        db.execute(ddl)
    db.commit()


def score(value) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = int(str(value).strip())
    except (ValueError, TypeError):
        return None
    return result if result >= 0 and str(value).strip() == str(result) else None


def normalize_final(fixture: dict) -> dict:
    """Never settle from live or penalty/extra-time score as if it were FT.

    Extra-time/penalty matches require explicit 90-minute FT score fields.
    An awarded result is VOID, not a sporting outcome.
    """
    raw = str(fixture.get("match_status") or "").strip()
    phase = str(fixture.get("match_raw_status") or "").strip().lower()
    status = (phase if phase in VOID | EXTRA_TIME else raw.lower())
    if status in VOID:
        return {"status": "void", "home_score": None, "away_score": None, "raw": raw}
    if status not in FINISHED | EXTRA_TIME:
        return {"status": "pending", "home_score": None, "away_score": None, "raw": raw}
    hft = score(fixture.get("match_hometeam_ft_score"))
    aft = score(fixture.get("match_awayteam_ft_score"))
    if status in EXTRA_TIME and (hft is None or aft is None):
        return {"status": "pending", "home_score": None, "away_score": None, "raw": raw}
    if hft is None or aft is None:
        if status not in FINISHED:
            return {"status": "pending", "home_score": None, "away_score": None, "raw": raw}
        hft = score(fixture.get("match_hometeam_score"))
        aft = score(fixture.get("match_awayteam_score"))
    if hft is None or aft is None:
        return {"status": "pending", "home_score": None, "away_score": None, "raw": raw}
    return {"status": "finished", "home_score": hft, "away_score": aft, "raw": raw}


def settle_market(selection: str, final: dict) -> str:
    state = final["status"]
    if state == "void":
        return "VOID"
    if state != "finished":
        return "PENDING"
    h, a = final["home_score"], final["away_score"]
    n = h + a
    values = {
        "Home Win": h > a, "Draw": h == a, "Away Win": a > h,
        "Over 1.5": n >= 2, "Under 1.5": n <= 1,
        "Over 2.5": n >= 3, "Under 2.5": n <= 2,
        "Over 3.5": n >= 4, "Under 3.5": n <= 3,
        "BTTS Yes": h > 0 and a > 0, "BTTS No": h == 0 or a == 0,
    }
    if selection not in values:
        return "UNSUPPORTED"
    return "WIN" if values[selection] else "LOSS"


def settle_fixture(db: sqlite3.Connection, fixture: dict) -> dict:
    fixture_id = str(fixture.get("match_id") or "")
    match_date = str(fixture.get("match_date") or "")
    if not fixture_id or not match_date:
        raise ValueError("fixture identity/date missing")
    final = normalize_final(fixture)
    existing = db.execute(
        "SELECT * FROM settled_fixtures WHERE source=? AND source_id=?",
        ("livescore", fixture_id)
    ).fetchone()
    # A transient stale feed may not downgrade a confirmed score or a VOID.
    if existing and existing["result_status"] in {"finished", "void"}:
        if final["status"] == "pending":
            return dict(existing)
        if (existing["result_status"] != final["status"] or
                (final["status"] == "finished" and
                 (existing["home_score"], existing["away_score"]) !=
                 (final["home_score"], final["away_score"]))):
            raise ValueError("conflicting final result for fixture " + fixture_id)
    with db:
        db.execute(
            """INSERT INTO settled_fixtures
            (source,source_id,match_date,result_status,home_score,away_score,
             raw_status,source_payload,updated_utc)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(source,source_id) DO UPDATE SET
              match_date=excluded.match_date,result_status=excluded.result_status,
              home_score=excluded.home_score,away_score=excluded.away_score,
              raw_status=excluded.raw_status,source_payload=excluded.source_payload,
              updated_utc=excluded.updated_utc""",
            ("livescore", fixture_id, match_date, final["status"],
             final["home_score"], final["away_score"], final["raw"],
             json.dumps(fixture, ensure_ascii=False), utc_now()),
        )
    return final


def record_selection(
    db: sqlite3.Connection, *, ticket_kind: str, ticket_id: str,
    leg_id: str, match_date: str, match_id: str, market: str,
) -> str:
    if market not in RULES:
        return "UNSUPPORTED"
    result = db.execute(
        "SELECT * FROM settled_fixtures WHERE source='livescore' AND source_id=?",
        (str(match_id),)
    ).fetchone()
    if result is None:
        outcome = "PENDING"
        h = a = None
    else:
        outcome = settle_market(market, dict(result) | {"status": result["result_status"]})
        h, a = result["home_score"], result["away_score"]
    with db:
        db.execute(
            """INSERT INTO settled_selections
            (ticket_kind,ticket_id,leg_id,match_date,source_id,market,outcome,
             home_score,away_score,updated_utc)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ticket_kind,ticket_id,leg_id) DO UPDATE SET
              outcome=excluded.outcome,home_score=excluded.home_score,
              away_score=excluded.away_score,updated_utc=excluded.updated_utc""",
            (ticket_kind, ticket_id, leg_id, match_date, str(match_id), market,
             outcome, h, a, utc_now()),
        )
    return outcome


def settle_snapshot_tickets(db: sqlite3.Connection, match_date: str,
                            snapshot: dict | None) -> int:
    """Settle frozen accumulator leg IDs; never invent missing bookmaker odds.

    A slip with a void and otherwise winning legs is VOID (stake return depends
    on verified bookmaker terms), not counted as a clean winning accumulator.
    """
    import hashlib
    total = 0
    for mode, tickets in (snapshot or {}).get("modes", {}).items():
        if not isinstance(tickets, list):
            continue
        for ticket in tickets:
            legs = ticket.get("legs") or []
            if not legs:
                continue
            identity = [(str(l.get("match_id") or ""), str(l.get("selection") or ""))
                        for l in legs]
            if any(not mid or market not in RULES for mid, market in identity):
                continue
            key = hashlib.sha256(json.dumps(
                [str(mode), identity], separators=(",", ":")
            ).encode("utf-8")).hexdigest()[:32]
            counts = {"WIN": 0, "LOSS": 0, "VOID": 0, "PENDING": 0}
            for index, (match_id, market) in enumerate(identity):
                verdict = record_selection(
                    db, ticket_kind="acca", ticket_id=key,
                    leg_id=str(index), match_date=match_date,
                    match_id=match_id, market=market,
                )
                counts[verdict] += 1
            if counts["LOSS"]:
                outcome = "LOSS"
            elif counts["PENDING"]:
                outcome = "PENDING"
            elif counts["VOID"]:
                outcome = "VOID"
            else:
                outcome = "WIN"
            with db:
                db.execute(
                    """INSERT INTO settled_tickets
                    (snapshot_mode,ticket_id,match_date,outcome,win_legs,
                    loss_legs,void_legs,pending_legs,updated_utc)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(snapshot_mode,ticket_id) DO UPDATE SET
                    outcome=excluded.outcome,win_legs=excluded.win_legs,
                    loss_legs=excluded.loss_legs,void_legs=excluded.void_legs,
                    pending_legs=excluded.pending_legs,updated_utc=excluded.updated_utc""",
                    (mode, key, match_date, outcome, counts["WIN"],
                     counts["LOSS"], counts["VOID"], counts["PENDING"], utc_now()),
                )
            total += 1
    return total


def settlement_report(db: sqlite3.Connection, match_date: str) -> dict:
    fixtures = db.execute(
        "SELECT result_status,COUNT(*) AS c FROM settled_fixtures "
        "WHERE match_date=? GROUP BY result_status", (match_date,)
    ).fetchall()
    counts = {r["result_status"]: r["c"] for r in fixtures}
    rows = db.execute(
        "SELECT ticket_kind,outcome,COUNT(*) AS c FROM settled_selections "
        "WHERE match_date=? GROUP BY ticket_kind,outcome", (match_date,)
    ).fetchall()
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["ticket_kind"], {})[row["outcome"]] = row["c"]
    for kind, values in by_kind.items():
        decided = values.get("WIN", 0) + values.get("LOSS", 0)
        values["settled"] = decided
        values["win_rate_pct"] = round(values.get("WIN", 0) / decided * 100, 1) if decided else None
    tickets = db.execute(
        "SELECT snapshot_mode,outcome,COUNT(*) AS c FROM settled_tickets "
        "WHERE match_date=? GROUP BY snapshot_mode,outcome", (match_date,)
    ).fetchall()
    ticket_summary = {}
    for row in tickets:
        ticket_summary.setdefault(row["snapshot_mode"], {})[row["outcome"]] = row["c"]
    return {"date": match_date, "fixtures": counts, "selections": by_kind,
            "tickets": ticket_summary, "updated_at_utc": utc_now()}


def sync_date(db: sqlite3.Connection, day: str, fixture_rows: list[dict],
              predictions: list[dict], bot_data: dict | None = None,
              build_snapshot: dict | None = None) -> dict:
    """Reconcile public predictions/bots without changing their source files."""
    init(db)
    started = utc_now()
    settled = pending = void = errors = 0
    for fixture in fixture_rows:
        try:
            final = settle_fixture(db, fixture)
            if final["status"] == "finished":
                settled += 1
            elif final["status"] == "void":
                void += 1
            else:
                pending += 1
        except ValueError:
            errors += 1
    for p in predictions:
        if not isinstance(p, dict) or not p.get("match_id"):
            continue
        selection = p.get("prediction")
        if selection in RULES:
            record_selection(
                db, ticket_kind="prediction",
                ticket_id=day + ":" + str(p["match_id"]),
                leg_id=str(p["match_id"]),
                match_date=day, match_id=str(p["match_id"]),
                market=selection,
            )
    for bot in (bot_data or {}).get("bots", []):
        for i, bet in enumerate(bot.get("history", [])):
            if not bet.get("match_id") or not bet.get("selection"):
                continue
            record_selection(
                db, ticket_kind="bot", ticket_id=day + ":" + str(bot.get("bot") or "unknown"),
                leg_id=str(i), match_date=day, match_id=str(bet["match_id"]),
                market=str(bet["selection"]),
            )
    settle_snapshot_tickets(db, day, build_snapshot)
    status = "partial" if errors else "completed"
    with db:
        db.execute(
            """INSERT INTO settlement_runs
            (match_date,started_utc,completed_utc,status,events,settled,pending,void,error)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (day, started, utc_now(), status, len(fixture_rows), settled,
             pending, void, f"{errors} conflicting fixtures" if errors else ""),
        )
    return settlement_report(db, day) | {"status": status, "conflicts": errors}
