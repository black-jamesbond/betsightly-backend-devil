"""Offline tests for operator access and monitoring."""
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services.dashboard_security import (
    WindowLimiter, needs_operator_auth, valid_operator_token,
)
from services.operational_health import inspect
from services.settlement_service import init
from services.weekly_fixture_collector import connect


class OperatorAccessTests(unittest.TestCase):
    def test_protected_actions_require_auth(self):
        for path, method in (
            ("/api/generate", "GET"),
            ("/api/group-code", "GET"),
            ("/api/build/code", "GET"),
            ("/api/bots/run", "POST"),
        ):
            self.assertTrue(needs_operator_auth(path, method))
        self.assertTrue(needs_operator_auth("/api/build", "GET", {"rebuild": "1"}))

    def test_public_data_remains_public(self):
        for path in ("/", "/api/predictions", "/api/settlement-report",
                     "/api/operations/health", "/api/booking/status"):
            self.assertFalse(needs_operator_auth(path, "GET"))
        self.assertFalse(needs_operator_auth("/api/build", "GET"))

    def test_missing_or_weak_key_is_refused(self):
        key = "e" * 48
        self.assertFalse(valid_operator_token(key, None))
        self.assertFalse(valid_operator_token(key, "short"))
        self.assertFalse(valid_operator_token("incorrect" * 8, key))
        self.assertTrue(valid_operator_token(key, key))

    def test_costly_action_limiter(self):
        limiter = WindowLimiter(max_events=2, window_sec=10)
        self.assertTrue(limiter.allowed("one", now=100))
        self.assertTrue(limiter.allowed("one", now=101))
        self.assertFalse(limiter.allowed("one", now=102))
        self.assertTrue(limiter.allowed("one", now=111))


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 10, 23, 9, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_database_never_reported_healthy(self):
        result = inspect(self.root, self.now)
        self.assertEqual(result["status"], "unconfigured")
        self.assertEqual(result["checks"]["database"], "missing")

    def test_no_runs_is_degraded(self):
        db_file = self.root / "data/fixtures/fixture_store.sqlite3"
        db = connect(db_file)
        init(db)
        db.close()
        result = inspect(self.root, self.now)
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["checks"]["fixtures"]["state"], "missing")

    def test_recent_successful_jobs_can_report_healthy(self):
        db_file = self.root / "data/fixtures/fixture_store.sqlite3"
        db = connect(db_file)
        init(db)
        db.execute(
            "CREATE TABLE IF NOT EXISTS prediction_publish_runs "
            "(status TEXT,finished_utc TEXT,error TEXT)"
        )
        stamp = (self.now - timedelta(minutes=15)).isoformat()
        db.execute(
            "INSERT INTO fixture_import_runs "
            "(day,started_utc,finished_utc,status,fetched,accepted,rejected,error) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("2026-10-23", stamp, stamp, "success", 5, 5, 0, ""),
        )
        db.execute("INSERT INTO prediction_publish_runs VALUES (?,?,?)",
                   ("published", stamp, ""))
        db.execute(
            "INSERT INTO settlement_runs "
            "(match_date,started_utc,completed_utc,status,events,settled,pending,void,error) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            ("2026-10-23", stamp, stamp, "completed", 5, 2, 3, 0, ""),
        )
        db.commit()
        db.close()
        folder = self.root / "cache"
        folder.mkdir()
        monitor = folder / "booking_monitor_status.json"
        monitor.write_text('{"checked":2,"failed":0}')
        # Filesystem timestamp is genuinely recent; inspect at actual time.
        state = inspect(self.root, datetime.now(timezone.utc))
        self.assertEqual(state["checks"]["booking_monitor"]["state"], "healthy")
        self.assertIn(state["checks"]["fixtures"]["state"], ("stale", "healthy"))


if __name__ == "__main__":
    unittest.main()
