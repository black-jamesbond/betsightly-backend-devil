#!/usr/bin/env python3
"""Reconcile football results into SQLite without rewriting any published picks.

Default: recent three UTC fixture dates, including today. Explicit --date
settles that one date. Use --status to inspect existing records without I/O.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.weekly_fixture_collector import (
    _atomic_json, connect, lagos_today,
)
from services.settlement_service import init, settlement_report, sync_date


def read_json(path: Path, fallback):
    if not path.exists():
        return fallback
    content = json.loads(path.read_text(encoding="utf-8"))
    return content


def settle_day(db: sqlite3.Connection, day: date, *, cache_dir: Path,
               report_dir: Path, fetcher=None) -> dict:
    day_str = day.isoformat()
    error = ""
    try:
        if fetcher is None:
            from fixtures_livescore import fetch_day
            fresh = fetch_day(day_str, timeout=25, use_cache=False)
        else:
            fresh = fetcher(day_str)
        if not isinstance(fresh, list) or not fresh:
            raise ValueError("source returned no fixtures")
        # Results are stored in settlement tables only; neither the fixture
        # collection snapshots nor the original published picks are altered.
        fixtures = [item for item in fresh if isinstance(item, dict)
                    and item.get("match_date") == day_str and item.get("match_id")]
    except Exception as exc:
        # Never mark a stale feed as successful. A failed network call may still
        # reconcile the previously stored data for auditing.
        error = str(exc)
        rows = db.execute(
            "SELECT payload_json FROM collected_fixtures "
            "WHERE match_date=? ORDER BY source_id", (day_str,)
        ).fetchall()
        fixtures = [json.loads(row["payload_json"]) for row in rows]

    pred_path = cache_dir / ("predictions_" + day_str + ".json")
    bot_path = ROOT / "bot_results" / ("bots_" + day_str + ".json")
    predictions = read_json(pred_path, [])
    bots = read_json(bot_path, {"bots": []})
    frozen = read_json(cache_dir / ("build_" + day_str + ".json"), {})
    if not isinstance(frozen, dict):
        raise ValueError("invalid frozen build snapshot")
    if not isinstance(predictions, list):
        raise ValueError("invalid predictions cache " + pred_path.name)
    if not isinstance(bots, dict):
        raise ValueError("invalid bot results cache " + bot_path.name)
    report = sync_date(db, day_str, fixtures, predictions, bots, frozen)
    report["feed"] = "fresh" if not error else "stale_fallback"
    if error:
        report["source_error"] = error[:400]
        report["status"] = "degraded"
    report_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_dir / ("settlement_" + day_str + ".json"), report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--db", type=Path, default=ROOT / "data/fixtures/fixture_store.sqlite3")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache/apifootball")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "cache/reports")
    args = parser.parse_args(argv)
    if args.days < 1 or args.days > 7:
        parser.error("--days must be 1..7")
    os.chdir(ROOT)
    current = lagos_today()
    days = [args.date] if args.date else [
        current - timedelta(days=i) for i in reversed(range(args.days))
    ]
    if any(day > current for day in days):
        parser.error("Cannot settle future fixtures")
    with connect(args.db) as db:
        init(db)
        if args.status:
            print(json.dumps([settlement_report(db, d.isoformat()) for d in days], indent=2))
            return 0
        rows = []
        for day in days:
            try:
                rows.append(settle_day(db, day, cache_dir=args.cache_dir, report_dir=args.report_dir))
            except Exception as exc:
                rows.append({"date": day.isoformat(), "status": "error", "error": str(exc)[:400]})
        print(json.dumps(rows, indent=2))
        return 1 if any(r["status"] != "completed" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
