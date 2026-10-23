#!/usr/bin/env python3
"""Predict stored fixtures daily and publish only validated pre-match selections.

Examples:
    python scripts/publish_daily_predictions.py run
    python scripts/publish_daily_predictions.py run --offset 1 --import-if-missing
    python scripts/publish_daily_predictions.py status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.weekly_fixture_collector import (
    _atomic_json, collect_day, connect, lagos_today, utc_now,
)
from services.daily_prediction_publisher import (
    ensure_runs, load_fixtures, publish, record_run,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status"))
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument("--offset", type=int, default=0,
                        help="Days from Lagos-local today; use 1 for tomorrow")
    parser.add_argument("--db", type=Path,
                        default=ROOT / "data/fixtures/fixture_store.sqlite3")
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "cache/apifootball")
    parser.add_argument("--import-if-missing", action="store_true",
                        help="Attempt a fresh feed import only when day is absent")
    parser.add_argument("--timeout", type=int, default=1500)
    args = parser.parse_args(argv)
    if args.offset not in (0, 1):
        parser.error("--offset must be 0 (today) or 1 (tomorrow)")
    if args.timeout < 10 or args.timeout > 3600:
        parser.error("--timeout must be between 10 and 3600 seconds")
    if args.date and args.offset:
        parser.error("Specify --date or --offset, not both")

    os.chdir(ROOT)
    selected = args.date or lagos_today() + timedelta(days=args.offset)
    if selected < lagos_today() and args.mode == "run":
        parser.error("Refusing to publish retroactive predictions")
    started = utc_now()
    with connect(args.db) as db:
        ensure_runs(db)
        if args.mode == "status":
            latest = db.execute(
                "SELECT * FROM prediction_publish_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            summary = {
                "latest": dict(latest) if latest else None,
                "date": selected.isoformat(),
                "saved": len(load_fixtures(db, selected)),
                "publication_file": str(
                    args.cache_dir / ("predictions_" + selected.isoformat() + ".json")
                ),
            }
            print(json.dumps(summary, indent=2))
            return 0

        try:
            fixtures = load_fixtures(db, selected)
            if not fixtures and args.import_if_missing:
                import_result = collect_day(
                    db, selected, cache_dir=args.cache_dir,
                    publish_cache=False,
                )
                if import_result["status"] == "success":
                    fixtures = load_fixtures(db, selected)
            if not fixtures:
                raise RuntimeError(
                    "no stored fixtures for " + selected.isoformat()
                    + "; complete weekly/daily import first"
                )

            args.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="prediction-stage-", dir=args.cache_dir
            ) as tempdir:
                temp = Path(tempdir)
                fixture_file = temp / "fixtures.json"
                staged = temp / "model_output.json"
                _atomic_json(fixture_file, fixtures)
                command = [
                    sys.executable, str(ROOT / "predict_today.py"),
                    "--date", selected.isoformat(),
                    "--fixture-file", str(fixture_file),
                    "--save", str(staged),
                    "--no-cache", "--require-history",
                ]
                done = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True,
                    timeout=args.timeout, check=False,
                )
                if done.returncode:
                    raise RuntimeError(
                        "model exited " + str(done.returncode) + ": "
                        + (done.stderr.strip() or done.stdout.strip())[-700:]
                    )
                if not staged.exists():
                    raise RuntimeError("model exited without producing staged output")
                generated = json.loads(staged.read_text(encoding="utf-8"))
                output = publish(
                    selected, args.cache_dir, fixtures,
                    generated, datetime.now(timezone.utc),
                )
            record_run(
                db, selected, started, output["status"],
                fixture_count=output["fixture_count"],
                model_count=output["model_count"],
                published_count=output["published_count"],
                appended_count=output["appended_count"],
                eligible_count=output["eligible_count"],
                skipped_started=output["skipped_started"],
            )
            print(json.dumps({"date": selected.isoformat(), **output}, indent=2))
            return 0
        except Exception as exc:
            record_run(db, selected, started, "error", error=str(exc))
            print(json.dumps({
                "date": selected.isoformat(), "status": "error",
                "error": str(exc),
                "published_cache_preserved": True,
            }, indent=2), file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
