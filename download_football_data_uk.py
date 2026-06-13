"""
Download football-data.co.uk datasets.

Two formats exist on the site:
1. Main leagues  : per-season files  https://www.football-data.co.uk/mmz4281/{season}/{div}.csv
2. Extra leagues : one file per country (all seasons) https://www.football-data.co.uk/new/{code}.csv

Downloads everything into data/football-data-uk/raw/ then a separate script
merges them into the training format.

Usage:
    python download_football_data_uk.py [--seasons N]
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

BASE = "https://www.football-data.co.uk"
RAW_DIR = Path("data/football-data-uk/raw")

# Main league division codes -> human name
MAIN_DIVS = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League 1",
    "E3": "England League 2",
    "EC": "England Conference",
    "SC0": "Scotland Premiership",
    "SC1": "Scotland Championship",
    "SC2": "Scotland League 1",
    "SC3": "Scotland League 2",
    "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda Division",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Jupiler League",
    "P1": "Portugal Liga I",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}

# Extra league country files (each contains ALL seasons)
EXTRA_CODES = {
    "ARG": "Argentina Primera Division",
    "AUT": "Austria Bundesliga",
    "BRA": "Brazil Serie A",
    "CHN": "China Super League",
    "DNK": "Denmark Superliga",
    "FIN": "Finland Veikkausliiga",
    "IRL": "Ireland Premier Division",
    "JPN": "Japan J-League",
    "MEX": "Mexico Liga MX",
    "NOR": "Norway Eliteserien",
    "POL": "Poland Ekstraklasa",
    "ROU": "Romania Liga 1",
    "RUS": "Russia Premier League",
    "SWE": "Sweden Allsvenskan",
    "SWZ": "Switzerland Super League",
    "USA": "USA MLS",
}


def season_codes(n: int) -> list:
    """Last n season codes, newest first. 2526 = 2025/26."""
    codes = []
    start_year = 25  # 2025/26 is the current season (June 2026)
    for i in range(n):
        y1 = start_year - i
        y2 = y1 + 1
        codes.append(f"{y1:02d}{y2:02d}")
    return codes


def download(url: str, dest: Path, pause: float = 0.4) -> bool:
    if dest.exists() and dest.stat().st_size > 200:
        print(f"    skip (cached): {dest.name}")
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 200:  # empty/placeholder file
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        time.sleep(pause)
        return True
    except Exception as e:
        print(f"    FAILED {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, default=6, help="How many recent seasons for main leagues")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0

    seasons = season_codes(args.seasons)
    print(f"Main leagues: {len(MAIN_DIVS)} divisions x {len(seasons)} seasons ({seasons})")
    for season in seasons:
        print(f"  Season {season}:")
        for div in MAIN_DIVS:
            url = f"{BASE}/mmz4281/{season}/{div}.csv"
            dest = RAW_DIR / f"main_{season}_{div}.csv"
            if download(url, dest):
                ok += 1
            else:
                fail += 1

    print(f"\nExtra leagues: {len(EXTRA_CODES)} countries (all seasons per file)")
    for code in EXTRA_CODES:
        url = f"{BASE}/new/{code}.csv"
        dest = RAW_DIR / f"extra_{code}.csv"
        print(f"  {code} ({EXTRA_CODES[code]})")
        if download(url, dest):
            ok += 1
        else:
            fail += 1

    print(f"\nDone. Downloaded/cached: {ok}, failed: {fail}")
    total_size = sum(f.stat().st_size for f in RAW_DIR.glob("*.csv"))
    print(f"Raw data size: {total_size/1024/1024:.1f} MB in {len(list(RAW_DIR.glob('*.csv')))} files")


if __name__ == "__main__":
    main()
