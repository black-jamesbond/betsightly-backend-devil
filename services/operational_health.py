"""Read-only health checks for the standalone BetSightly automation pipeline."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def _iso_age_hours(value: str | None, now: datetime) -> float | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            return None
        return (now - timestamp.astimezone(timezone.utc)).total_seconds() / 3600
    except (TypeError, ValueError):
        return None


def _last(db: sqlite3.Connection, sql: str) -> dict | None:
    try:
        row = db.execute(sql).fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None


def inspect(root: Path, now: datetime | None = None) -> dict:
    """Does not create files, connect to the web, or open secrets."""
    now = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    local_day = now.astimezone(ZoneInfo("Africa/Lagos")).date()
    root = Path(root)
    db_file = root / "data/fixtures/fixture_store.sqlite3"
    checks = {}
    issues = []
    db = None
    if db_file.exists():
        try:
            db = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=3)
            db.row_factory = sqlite3.Row
            checks["database"] = "available"
        except sqlite3.Error:
            issues.append("fixture/settlement database unreadable")
            checks["database"] = "error"
    else:
        issues.append("fixture/settlement database missing (timers not installed)")
        checks["database"] = "missing"

    for name, sql, stamp, good_status, age_limit in (
        ("fixtures",
         "SELECT status,finished_utc AS at,error FROM fixture_import_runs "
         "ORDER BY id DESC LIMIT 1", "at", "success", 30),
        ("predictions",
         "SELECT status,finished_utc AS at,error FROM prediction_publish_runs "
         "ORDER BY id DESC LIMIT 1", "at", "published", 30),
        ("settlement",
         "SELECT status,completed_utc AS at,error FROM settlement_runs "
         "ORDER BY id DESC LIMIT 1", "at", "completed", 7),
    ):
        row = _last(db, sql) if db else None
        age = _iso_age_hours(row.get(stamp), now) if row else None
        if row is None or age is None:
            state = "missing"
        elif age > age_limit:
            state = "stale"
        elif name == "predictions" and row.get("status") in ("published", "unchanged"):
            state = "healthy"
        elif row.get("status") == good_status:
            state = "healthy"
        else:
            state = "error"
        checks[name] = {
            "state": state,
            "last_run": row.get(stamp) if row else None,
            "age_hours": round(age, 2) if age is not None else None,
        }
        if state != "healthy":
            issues.append(name + " job " + state)

    if db:
        db.close()

    monitor = root / "cache/booking_monitor_status.json"
    if monitor.exists():
        try:
            data = json.loads(monitor.read_text(encoding="utf-8"))
            age = (now.timestamp() - monitor.stat().st_mtime) / 3600
            if data.get("status") == "error":
                state = "error"
            elif age > 1:
                state = "stale"
            elif data.get("failed", 0) > 0:
                state = "error"
            else:
                state = "healthy"
            checks["booking_monitor"] = {
                "state": state, "age_hours": round(age, 2),
                "active_checked": data.get("checked", 0),
                "unavailable": data.get("unavailable", 0),
            }
        except (OSError, ValueError, TypeError):
            checks["booking_monitor"] = {"state": "error"}
    else:
        checks["booking_monitor"] = {"state": "missing"}
    if checks["booking_monitor"]["state"] != "healthy":
        issues.append("booking monitor " + checks["booking_monitor"]["state"])

    if not db_file.exists():
        overall = "unconfigured"
    elif issues:
        overall = "degraded"
    else:
        overall = "healthy"
    return {
        "status": overall,
        "observed_at_utc": now.isoformat(),
        "lagos_day": local_day.isoformat(),
        "checks": checks,
        "issues": issues,
    }
