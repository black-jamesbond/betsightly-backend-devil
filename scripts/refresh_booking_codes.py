#!/usr/bin/env python3
"""Recheck issued SportyBet booking codes, renewing changed live quotations.

Only active, tracked, already-issued code requests are checked. This script
does not place a bet or authorize stakes. Safe to invoke from a systemd timer.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.booking_code_monitor import run_once, _write_status


def main() -> int:
    os.chdir(ROOT)
    status_path = ROOT / "cache" / "booking_monitor_status.json"
    try:
        result = run_once(save_status_path=status_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Error responses are visible to the service manager, not silently green.
        return 1 if result.get("failed") else 0
    except Exception as exc:
        failure = {"status": "error", "source": "bookmaker",
                   "error": str(exc)[:400]}
        _write_status(status_path, failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
