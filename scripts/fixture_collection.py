#!/usr/bin/env python3
"""Fixture collection CLI, designed for VPS systemd timers.

Sunday: python scripts/fixture_collection.py weekly --publish-cache
Daily:  python scripts/fixture_collection.py daily --publish-cache
Check:  python scripts/fixture_collection.py status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.weekly_fixture_collector import (
    collect_day, collect_week, connect, lagos_today, next_sunday,
    status_summary,
)


def date_arg(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must be YYYY-MM-DD") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("weekly", "daily", "status"))
    parser.add_argument("--date", type=date_arg, default=None,
                        help="Override daily date (YYYY-MM-DD)")
    parser.add_argument("--start", type=date_arg, default=None,
                        help="Override start of weekly/status window")
    parser.add_argument("--db", type=Path,
                        default=ROOT / "data" / "fixtures" / "fixture_store.sqlite3")
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "cache" / "apifootball")
    parser.add_argument("--publish-cache", action="store_true",
                        help="Atomically update the dashboard's events_DATE.json files")
    args = parser.parse_args(argv)
    # LiveScore raw feed cache uses project-relative paths.
    os.chdir(ROOT)

    with connect(args.db) as db:
        if args.mode == "daily":
            rows = [collect_day(db, args.date or lagos_today(),
                                cache_dir=args.cache_dir,
                                publish_cache=args.publish_cache)]
        elif args.mode == "weekly":
            start = args.start or next_sunday(lagos_today())
            rows = collect_week(db, start, cache_dir=args.cache_dir,
                                publish_cache=args.publish_cache)
        else:
            start = args.start or lagos_today()
            print(json.dumps(status_summary(db, start), indent=2))
            return 0

        print(json.dumps({
            "mode": args.mode,
            "days": rows,
            "succeeded": sum(row["status"] == "success" for row in rows),
            "failed": sum(row["status"] != "success" for row in rows),
            "published_cache": args.publish_cache,
            "database": str(args.db),
        }, indent=2))
        return 1 if any(row["status"] != "success" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
