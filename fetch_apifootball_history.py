"""
Fetch historical match results from apifootball.com (last N years, all leagues).

Walks the date range in small chunks via get_events, strips each match down to
a compact result row, and appends to data/apifootball/history.csv.

Resumable: progress checkpoint in data/apifootball/fetch_checkpoint.json.
Re-running skips completed chunks.

Usage:
    python fetch_apifootball_history.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                        [--chunk-days 3] [--pause 4.0]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

API_KEY = "7233da77b5d53606a174f2759f0d947beb18aa17c00bf350d852f35b80aa7455"
BASE = "https://apiv3.apifootball.com/"

OUT_DIR = Path("data/apifootball")
OUT_CSV = OUT_DIR / "history.csv"
CHECKPOINT = OUT_DIR / "fetch_checkpoint.json"

CSV_FIELDS = [
    "match_id", "date", "time", "country", "league_id", "league_name",
    "home_team", "away_team", "home_score", "away_score",
    "ht_home_score", "ht_away_score", "season",
]


def fetch_chunk(start: date, end: date) -> list:
    url = (
        f"{BASE}?action=get_events&from={start.isoformat()}&to={end.isoformat()}"
        f"&APIkey={API_KEY}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    if isinstance(data, dict):
        # API error payload, e.g. {"error": 404, "message": "No event found"}
        msg = str(data.get("message", data))
        if "No event found" in msg or data.get("error") == 404:
            return []
        raise RuntimeError(f"API error: {data}")
    return data


def strip_match(m: dict) -> dict | None:
    """Keep only finished matches with real scores."""
    if m.get("match_status") != "Finished":
        return None
    hs, as_ = m.get("match_hometeam_score", ""), m.get("match_awayteam_score", "")
    if hs == "" or as_ == "":
        return None
    try:
        int(hs), int(as_)
    except ValueError:
        return None
    return {
        "match_id": m.get("match_id", ""),
        "date": m.get("match_date", ""),
        "time": m.get("match_time", ""),
        "country": m.get("country_name", ""),
        "league_id": m.get("league_id", ""),
        "league_name": m.get("league_name", ""),
        "home_team": m.get("match_hometeam_name", ""),
        "away_team": m.get("match_awayteam_name", ""),
        "home_score": hs,
        "away_score": as_,
        "ht_home_score": m.get("match_hometeam_halftime_score", ""),
        "ht_away_score": m.get("match_awayteam_halftime_score", ""),
        "season": m.get("league_year", ""),
    }


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done_chunks": [], "rows_written": 0}


def save_checkpoint(cp: dict) -> None:
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(cp, f)


def main():
    parser = argparse.ArgumentParser()
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=4 * 365)
    parser.add_argument("--start", default=default_start.isoformat())
    parser.add_argument("--end", default=default_end.isoformat())
    parser.add_argument("--chunk-days", type=int, default=3)
    parser.add_argument("--pause", type=float, default=4.0,
                        help="Seconds between calls (1000/hr limit -> 3.6s min)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build chunk list
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=args.chunk_days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)

    cp = load_checkpoint()
    done = set(cp["done_chunks"])
    todo = [(a, b) for a, b in chunks if a.isoformat() not in done]

    print(f"Range {start} -> {end}: {len(chunks)} chunks total, "
          f"{len(done)} already done, {len(todo)} to fetch", flush=True)

    # Open CSV in append mode (write header if new)
    new_file = not OUT_CSV.exists()
    f_out = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()

    total_rows = cp["rows_written"]
    consecutive_errors = 0
    i = 0

    while i < len(todo):
        a, b = todo[i]
        try:
            matches = fetch_chunk(a, b)
            rows = [r for r in (strip_match(m) for m in matches) if r]
            for r in rows:
                writer.writerow(r)
            f_out.flush()
            total_rows += len(rows)
            consecutive_errors = 0

            cp["done_chunks"].append(a.isoformat())
            cp["rows_written"] = total_rows
            save_checkpoint(cp)

            print(f"[{i+1}/{len(todo)}] {a} -> {b}: {len(rows)} finished matches "
                  f"(total {total_rows:,})", flush=True)
            i += 1
        except Exception as e:
            consecutive_errors += 1
            print(f"[{i+1}/{len(todo)}] {a} -> {b}: ERROR {e}", flush=True)
            if consecutive_errors >= 30:
                print("30 consecutive errors -- giving up (checkpoint saved, "
                      "re-run to resume).", flush=True)
                break
            if consecutive_errors >= 5:
                # Probably a network outage -- wait it out and retry same chunk
                print(f"  ({consecutive_errors} consecutive errors -- "
                      f"waiting 5 min for network to recover)", flush=True)
                time.sleep(300)
            else:
                time.sleep(30)  # brief back-off, retry same chunk

        time.sleep(args.pause)

    f_out.close()
    print(f"\nDone. {total_rows:,} rows in {OUT_CSV}", flush=True)
    remaining = len(todo) - len([c for c in cp['done_chunks'] if c]) + len(done)
    if len(cp["done_chunks"]) < len(chunks):
        print(f"({len(chunks) - len(cp['done_chunks'])} chunks remaining -- "
              f"re-run to resume)", flush=True)


if __name__ == "__main__":
    main()
