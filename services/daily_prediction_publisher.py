"""Guarded publication of model predictions for the standalone VPS dashboard.

No bookmaker verification claim is made. Entries already published are never
re-predicted or overwritten; new predictions are released only pre-kickoff.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from services.weekly_fixture_collector import _atomic_json, utc_now


CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS prediction_publish_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date TEXT NOT NULL,
    started_utc TEXT NOT NULL,
    finished_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    fixture_count INTEGER NOT NULL DEFAULT 0,
    model_count INTEGER NOT NULL DEFAULT 0,
    published_count INTEGER NOT NULL DEFAULT 0,
    appended_count INTEGER NOT NULL DEFAULT 0,
    eligible_count INTEGER NOT NULL DEFAULT 0,
    skipped_started INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT ''
)
"""


def ensure_runs(db: sqlite3.Connection) -> None:
    db.execute(CREATE_RUNS)
    db.commit()


def record_run(
    db: sqlite3.Connection,
    day: date, started: str, status: str,
    fixture_count: int = 0, model_count: int = 0,
    published_count: int = 0, appended_count: int = 0,
    eligible_count: int = 0, skipped_started: int = 0, error: str = "",
) -> None:
    ensure_runs(db)
    with db:
        db.execute(
            "INSERT INTO prediction_publish_runs "
            "(match_date,started_utc,finished_utc,status,fixture_count,"
            "model_count,published_count,appended_count,eligible_count,"
            "skipped_started,error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (day.isoformat(), started, utc_now(), status, fixture_count,
             model_count, published_count, appended_count, eligible_count,
             skipped_started, error[:1000]),
        )


def load_fixtures(db: sqlite3.Connection, day: date) -> list[dict]:
    """Only stored, valid fixtures for this UTC fixture date."""
    rows = db.execute(
        "SELECT payload_json FROM collected_fixtures WHERE match_date=? "
        "ORDER BY match_time, source_id", (day.isoformat(),)
    ).fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def kickoff_utc(fixture: dict) -> datetime:
    return datetime.strptime(
        fixture["match_date"] + " " + fixture["match_time"], "%Y-%m-%d %H:%M"
    ).replace(tzinfo=timezone.utc)


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("existing prediction cache is not a list; refusing overwrite")
    if any(not isinstance(p, dict) for p in loaded):
        raise ValueError("existing prediction cache contains invalid items")
    return loaded


def _key(p: dict) -> str:
    return str(p.get("match_id") or "").strip()


def _validate_prediction(p: dict, fixture: dict) -> None:
    if not isinstance(p, dict):
        raise ValueError("prediction record must be an object")
    if _key(p) != str(fixture["match_id"]):
        raise ValueError("model prediction has wrong fixture ID")
    if (p.get("home_team") != fixture["match_hometeam_name"]
            or p.get("away_team") != fixture["match_awayteam_name"]):
        raise ValueError("model prediction team identity mismatch for " + _key(p))
    confidence = float(p.get("confidence", -1))
    if not (0 <= confidence <= 100):
        raise ValueError("invalid confidence for " + _key(p))
    if p.get("prediction") not in {"Home Win", "Away Win", "Draw"}:
        raise ValueError("unrecognized 1X2 prediction for " + _key(p))
    if p.get("data_quality") not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError("missing data quality for " + _key(p))


def prepare_publication(
    fixtures: list[dict], generated: list[dict], existing: list[dict],
    now_utc: datetime,
) -> tuple[list[dict], dict]:
    """Validate complete upcoming coverage; preserve published picks forever."""
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone aware")
    if not fixtures:
        raise ValueError("fixture store is empty")
    if not isinstance(generated, list) or not generated:
        raise ValueError("model produced zero predictions")
    if not isinstance(existing, list):
        raise ValueError("existing cache must be a list")

    fixture_map = {}
    for f in fixtures:
        key = _key(f)
        if not key or key in fixture_map:
            raise ValueError("missing or duplicate fixture identity")
        fixture_map[key] = f
    model_map = {}
    for p in generated:
        if not isinstance(p, dict):
            raise ValueError("model output contains a non-object")
        key = _key(p)
        if not key or key in model_map or key not in fixture_map:
            raise ValueError("invalid/duplicate/unknown model fixture identity")
        _validate_prediction(p, fixture_map[key])
        model_map[key] = p
    old_ids = [_key(p) for p in existing]
    if any(not key for key in old_ids) or len(set(old_ids)) != len(old_ids):
        raise ValueError("old cache has missing or duplicate fixture IDs")

    result = [dict(p) for p in existing]
    seen = set(old_ids)
    skipped_started = 0
    eligible_count = sum(
        p.get("data_quality") in ("HIGH", "MEDIUM")
        and float(p.get("confidence", 0)) >= 60
        for p in existing
    )
    missing_upcoming = []
    added = 0
    for key, fixture in fixture_map.items():
        if key in seen:
            # Published predictions remain immutable, even after new model runs.
            continue
        if fixture.get("match_status") not in ("", "Not Started", "NS", "TBD", None):
            skipped_started += 1
            continue
        if kickoff_utc(fixture) <= now_utc:
            skipped_started += 1
            continue
        if key not in model_map:
            missing_upcoming.append(key)
            continue
        p = dict(model_map[key])
        p["published_at_utc"] = now_utc.astimezone(timezone.utc).isoformat()
        p["recommendation_eligible"] = (
            p.get("data_quality") in ("HIGH", "MEDIUM")
            and float(p.get("confidence", 0)) >= 60
        )
        eligible_count += int(p["recommendation_eligible"])
        result.append(p)
        seen.add(key)
        added += 1
    if missing_upcoming:
        raise ValueError(
            "model omitted upcoming fixtures: " + ",".join(missing_upcoming[:8])
        )
    if not result:
        raise ValueError("nothing valid to publish; all fixtures already started")
    return result, {
        "published_count": len(result),
        "appended_count": added,
        "eligible_count": eligible_count,
        "skipped_started": skipped_started,
        "model_count": len(generated),
        "fixture_count": len(fixtures),
    }


def publish(
    day: date, cache_dir: Path, fixtures: list[dict],
    generated: list[dict], now_utc: datetime,
) -> dict:
    """Validate before any write; keep an archive of the previous snapshot."""
    cache_dir = Path(cache_dir)
    destination = cache_dir / ("predictions_" + day.isoformat() + ".json")
    existing = load_existing(destination)
    prepared, stats = prepare_publication(fixtures, generated, existing, now_utc)
    if stats["appended_count"] == 0:
        stats["status"] = "unchanged"
        return stats
    if destination.exists():
        archive_dir = cache_dir / "archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        # A unique backup filename is necessary even for repeated runs in 1 sec.
        import uuid
        archive = archive_dir / (
            "predictions_" + day.isoformat() + "_" + uuid.uuid4().hex + ".json"
        )
        shutil.copy2(destination, archive)
    _atomic_json(destination, prepared)
    stats["status"] = "published"
    stats["destination"] = str(destination)
    return stats
