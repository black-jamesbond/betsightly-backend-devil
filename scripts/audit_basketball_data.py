#!/usr/bin/env python3
"""Audit NBA historical game source readiness; never generate data.

Usage:
 python scripts/audit_basketball_data.py
 python scripts/audit_basketball_data.py --enforce
 python scripts/audit_basketball_data.py --path /path/to/real.csv

Without --enforce exits 0 and reports NOT_READY in JSON for an existing
historical repository snapshot; --enforce exits 1 if data is not publishable.
Requires no API keys, network calls or extra pip packages.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "basketball" / "nba_games_raw.csv"
REQUIRED = {
    "GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP", "WL",
    "PTS", "SEASON", "TEAM_NAME",
}


def evaluate_csv(
    path: Path, *,
    today: date | None = None,
    min_games: int = 1200,
    min_seasons: int = 3,
    max_lag_days: int = 90,
) -> dict:
    today = today or datetime.now(timezone.utc).date()
    report = {
        "file": str(path), "checked_as_of_utc": today.isoformat(),
        "status": "NOT_READY", "rows": 0, "games": 0,
        "seasons": [], "first_game": None, "last_game": None,
        "invalid_games": 0, "invalid_rows": 0, "findings": [],
    }
    if not path.is_file():
        report["findings"].append("MISSING_SOURCE_FILE")
        return report
    games = defaultdict(list)
    seasons = set()
    dates = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing_columns = REQUIRED - set(reader.fieldnames or ())
            if missing_columns:
                report["findings"].append(
                    "MISSING_COLUMNS:" + ",".join(sorted(missing_columns))
                )
                return report
            for row in reader:
                report["rows"] += 1
                try:
                    day = date.fromisoformat(row["GAME_DATE"][:10])
                    if not row["GAME_ID"] or not row["TEAM_ID"] or not row["TEAM_NAME"]:
                        raise ValueError("required identity missing")
                    if row["WL"] not in ("W", "L"):
                        raise ValueError("invalid win/loss")
                    if int(row["PTS"]) < 0:
                        raise ValueError("invalid points")
                    if "vs." not in row["MATCHUP"] and "@" not in row["MATCHUP"]:
                        raise ValueError("missing home/away marker")
                    dates.append(day)
                    seasons.add(row["SEASON"])
                    games[row["GAME_ID"]].append(row)
                except (TypeError, KeyError, ValueError):
                    report["invalid_rows"] += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        report["findings"].append("UNREADABLE_FILE:" + type(exc).__name__)
        return report

    for game_id, rows in games.items():
        if (len(rows) != 2 or rows[0]["TEAM_ID"] == rows[1]["TEAM_ID"]
                or Counter(["home" if "vs." in x["MATCHUP"] else "away"
                            for x in rows]) != {"home": 1, "away": 1}
                or rows[0]["GAME_DATE"][:10] != rows[1]["GAME_DATE"][:10]
                or (rows[0]["WL"] == rows[1]["WL"])):
            report["invalid_games"] += 1
    report["games"] = len(games)
    report["seasons"] = sorted(seasons)
    report["first_game"] = min(dates).isoformat() if dates else None
    report["last_game"] = max(dates).isoformat() if dates else None
    report["age_days"] = (today - max(dates)).days if dates else None

    if report["invalid_rows"]:
        report["findings"].append("INVALID_ROWS")
    if report["invalid_games"]:
        report["findings"].append("INVALID_TWO_TEAM_GAMES")
    if report["games"] < min_games:
        report["findings"].append("INSUFFICIENT_GAMES")
    if len(seasons) < min_seasons:
        report["findings"].append("INSUFFICIENT_SEASONS")
    if report["age_days"] is None or report["age_days"] > max_lag_days:
        report["findings"].append("STALE_SOURCE")
    if report["age_days"] is not None and report["age_days"] < 0:
        report["findings"].append("FUTURE_GAMES_IN_HISTORY")

    report["status"] = "READY_FOR_FEATURE_REVIEW" if not report["findings"] else "NOT_READY"
    report["disclaimer"] = (
        "Schema/freshness readiness is not proof of pre-tipoff feature "
        "provenance, source licensing, calibrated accuracy, or bookmaker odds."
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    result = evaluate_csv(args.path)
    print(json.dumps(result, indent=2))
    return 1 if args.enforce and result["status"] != "READY_FOR_FEATURE_REVIEW" else 0


if __name__ == "__main__":
    sys.exit(main())
