"""Durable, idempotent fixture import for the independent VPS dashboard.

Stores canonical LiveScore fixtures in SQLite and optionally publishes compatible
per-date JSON snapshots for the existing Flask dashboard. No predictions are
modified. The source feed returns UTC fixture dates and kickoffs.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


LAGOS = ZoneInfo("Africa/Lagos")
TERMINAL = {"Finished"}
CREATE_FIXTURES = """
CREATE TABLE IF NOT EXISTS collected_fixtures (
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    match_date TEXT NOT NULL,
    match_time TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    first_seen_utc TEXT NOT NULL,
    last_seen_utc TEXT NOT NULL,
    PRIMARY KEY (source, source_id)
)
"""
CREATE_ACTIVE = """
CREATE TABLE IF NOT EXISTS active_fixture_ids (
    match_date TEXT NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    PRIMARY KEY (match_date, source, source_id)
)
"""
CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS fixture_import_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    started_utc TEXT NOT NULL,
    finished_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    fetched INTEGER NOT NULL,
    accepted INTEGER NOT NULL,
    rejected INTEGER NOT NULL,
    error TEXT NOT NULL DEFAULT ''
)
"""


def lagos_today() -> date:
    return datetime.now(LAGOS).date()


def next_sunday(day: date) -> date:
    """On Sunday start today; otherwise start at the upcoming Sunday."""
    return day + timedelta(days=(6 - day.weekday()) % 7)


def current_sunday(day: date) -> date:
    """The current Sunday-Saturday week (also correct after a missed timer run)."""
    return day - timedelta(days=(day.weekday() + 1) % 7)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=15000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(CREATE_FIXTURES)
    db.execute(CREATE_ACTIVE)
    db.execute(CREATE_RUNS)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_collected_fixtures_day "
        "ON collected_fixtures(match_date)"
    )
    db.commit()
    return db


def normalize(raw: dict, requested_day: str) -> dict | None:
    """Reject malformed/off-day/ID-less records rather than inventing identity."""
    if not isinstance(raw, dict):
        return None
    source_id = str(raw.get("match_id") or "").strip()
    home = str(raw.get("match_hometeam_name") or "").strip()
    away = str(raw.get("match_awayteam_name") or "").strip()
    match_day = str(raw.get("match_date") or "")
    kick_off = str(raw.get("match_time") or "")
    if not source_id or not home or not away or home == away:
        return None
    try:
        date.fromisoformat(match_day)
        datetime.strptime(kick_off, "%H:%M")
    except ValueError:
        return None
    if match_day != requested_day:
        return None
    record = dict(raw)
    record["match_id"] = source_id
    record["match_hometeam_name"] = home
    record["match_awayteam_name"] = away
    record["match_date"] = match_day
    record["match_time"] = kick_off
    record["match_status"] = str(raw.get("match_status") or "Not Started")
    record["source"] = "livescore"
    return record


def _atomic_json(path: Path, payload: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False
        ) as file:
            temp_name = file.name
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _load_saved_day(db: sqlite3.Connection, day: str, seen_ids: set[str]) -> list:
    # Historical records stay in the DB, but a refreshed card must not advertise
    # a match that the latest provider snapshot no longer contains.
    rows = db.execute(
        "SELECT source_id,payload_json FROM collected_fixtures WHERE match_date=? "
        "ORDER BY match_time, source_id", (day,)
    ).fetchall()
    return [json.loads(row["payload_json"]) for row in rows
            if row["source_id"] in seen_ids]


def default_fetcher(day: str) -> list:
    """Bypass raw-source cache so the scheduled job detects reschedules/results."""
    from fixtures_livescore import fetch_day
    return fetch_day(day, use_cache=False, timeout=25)


def collect_day(
    db: sqlite3.Connection,
    day: date,
    fetcher=None,
    cache_dir: Path | None = None,
    publish_cache: bool = False,
    attempts: int = 2,
) -> dict:
    """Commit a day independently. A failed/empty fetch never erases old data."""
    if fetcher is None:
        fetcher = default_fetcher
    day_str = day.isoformat()
    started = utc_now()
    raw = None
    error = None
    for retry in range(max(1, attempts)):
        try:
            raw = fetcher(day_str)
            if not isinstance(raw, list):
                raise ValueError("fixture provider did not return a list")
            if not raw:
                raise ValueError("fixture provider returned an empty day")
            break
        except Exception as exc:
            error = str(exc)
            if retry + 1 < max(1, attempts):
                time.sleep(1 + retry)
    try:
        if raw is None:
            raise RuntimeError(error or "fixture feed unavailable")
        unique = {}
        for fixture in raw:
            item = normalize(fixture, day_str)
            if item is not None:
                unique[item["match_id"]] = item
        if not unique:
            raise ValueError("no valid fixtures for UTC day " + day_str)
        stamped = utc_now()
        # An old feed response must never revert a final result to Not Started.
        with db:
            # The active board contains ONLY IDs from the latest successful
            # collection. Historical collected_fixtures rows are never deleted.
            db.execute("DELETE FROM active_fixture_ids WHERE match_date=? "
                       "AND source='livescore'", (day_str,))
            for event_id, item in unique.items():
                existing = db.execute(
                    "SELECT status, payload_json FROM collected_fixtures "
                    "WHERE source='livescore' AND source_id=?", (event_id,)
                ).fetchone()
                if existing and existing["status"] in TERMINAL and item["match_status"] not in TERMINAL:
                    item = json.loads(existing["payload_json"])
                db.execute(
                    "INSERT INTO active_fixture_ids(match_date,source,source_id) "
                    "VALUES (?,?,?)", (day_str, "livescore", event_id))
                db.execute(
                    "INSERT INTO collected_fixtures "
                    "(source,source_id,match_date,match_time,status,payload_json,"
                    "first_seen_utc,last_seen_utc) VALUES (?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(source,source_id) DO UPDATE SET "
                    "match_date=excluded.match_date, match_time=excluded.match_time,"
                    "status=excluded.status, payload_json=excluded.payload_json,"
                    "last_seen_utc=excluded.last_seen_utc",
                    (
                        "livescore", event_id, item["match_date"], item["match_time"],
                        item["match_status"], json.dumps(item, ensure_ascii=False),
                        stamped, stamped,
                    ),
                )
        if publish_cache:
            if cache_dir is None:
                raise ValueError("cache_dir is required when publish_cache=True")
            _atomic_json(Path(cache_dir) / ("events_" + day_str + ".json"),
                         _load_saved_day(db, day_str, set(unique)))
        result = {
            "date": day_str, "status": "success",
            "fetched": len(raw), "accepted": len(unique),
            "rejected": len(raw) - len(unique), "error": "",
        }
    except Exception as exc:
        result = {
            "date": day_str, "status": "error",
            "fetched": len(raw) if isinstance(raw, list) else 0,
            "accepted": 0, "rejected": 0, "error": str(exc),
        }
    with db:
        db.execute(
            "INSERT INTO fixture_import_runs "
            "(day,started_utc,finished_utc,status,fetched,accepted,rejected,error) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                day_str, started, utc_now(), result["status"],
                result["fetched"], result["accepted"],
                result["rejected"], result["error"],
            ),
        )
    return result


def collect_week(
    db: sqlite3.Connection, sunday: date, fetcher=None,
    cache_dir: Path | None = None, publish_cache: bool = False,
) -> list[dict]:
    return [
        collect_day(db, sunday + timedelta(days=offset),
                    fetcher=fetcher, cache_dir=cache_dir,
                    publish_cache=publish_cache)
        for offset in range(7)
    ]


def status_summary(db: sqlite3.Connection, start: date, days: int = 7) -> dict:
    dates = [(start + timedelta(days=offset)).isoformat()
             for offset in range(days)]
    counts = dict(db.execute(
        "SELECT match_date,COUNT(*) AS n FROM active_fixture_ids "
        "WHERE match_date>=? AND match_date<=? GROUP BY match_date",
        (dates[0], dates[-1]),
    ).fetchall())
    latest = db.execute(
        "SELECT day,status,fetched,accepted,rejected,finished_utc,error "
        "FROM fixture_import_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return {
        "start": dates[0], "end": dates[-1],
        "counts": {day: counts.get(day, 0) for day in dates},
        "latest_run": dict(latest) if latest else None,
    }
