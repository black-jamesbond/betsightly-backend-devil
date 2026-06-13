"""
Merge football-data.co.uk raw CSVs into data/api-football/matches.csv
(the canonical training + prediction-history file).

Handles:
- Two raw formats (main per-season files, extra all-season country files)
- Team-name alignment for leagues already present in the API-Football data
  (fuzzy-matched per league so team histories aren't split across spellings)
- Deduplication on (date, home_team, away_team)

Usage:
    python merge_football_data_uk.py [--dry-run]
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/football-data-uk/raw")
MATCHES_CSV = Path("data/api-football/matches.csv")

CSV_COLUMNS = [
    "home_team", "away_team", "date", "home_score", "away_score",
    "ht_home_score", "ht_away_score", "league_id", "league_name",
    "country", "league_tier", "season", "home_team_id", "away_team_id",
]

# Division code -> (league_name, country, tier).
# Names for overlapping leagues match the API-Football naming so rows merge
# into the same league; the rest use football-data naming.
MAIN_DIVS = {
    "E0":  ("Premier League", "England", 1),
    "E1":  ("Championship", "England", 2),
    "E2":  ("League One", "England", 3),
    "E3":  ("League Two", "England", 4),
    "EC":  ("National League", "England", 5),
    "SC0": ("Premiership", "Scotland", 1),
    "SC1": ("Championship", "Scotland", 2),
    "SC2": ("League One", "Scotland", 3),
    "SC3": ("League Two", "Scotland", 4),
    "D1":  ("Bundesliga", "Germany", 1),
    "D2":  ("2. Bundesliga", "Germany", 2),
    "I1":  ("Serie A", "Italy", 1),
    "I2":  ("Serie B", "Italy", 2),
    "SP1": ("La Liga", "Spain", 1),
    "SP2": ("Segunda Division", "Spain", 2),
    "F1":  ("Ligue 1", "France", 1),
    "F2":  ("Ligue 2", "France", 2),
    "N1":  ("Eredivisie", "Netherlands", 1),
    "B1":  ("Pro League", "Belgium", 1),
    "P1":  ("Primeira Liga", "Portugal", 1),
    "T1":  ("Super Lig", "Turkey", 1),
    "G1":  ("Super League", "Greece", 1),
}

EXTRA_FILES = {
    "ARG": ("Primera Division", "Argentina", 1),
    "AUT": ("Bundesliga", "Austria", 1),
    "BRA": ("Serie A", "Brazil", 1),
    "CHN": ("Super League", "China", 1),
    "DNK": ("Superliga", "Denmark", 1),
    "FIN": ("Veikkausliiga", "Finland", 1),
    "IRL": ("Premier Division", "Ireland", 1),
    "JPN": ("J1 League", "Japan", 1),
    "MEX": ("Liga MX", "Mexico", 1),
    "NOR": ("Eliteserien", "Norway", 1),
    "POL": ("Ekstraklasa", "Poland", 1),
    "ROU": ("Liga 1", "Romania", 1),
    "RUS": ("Premier League", "Russia", 1),
    "SWE": ("Allsvenskan", "Sweden", 1),
    "SWZ": ("Super League", "Switzerland", 1),
    "USA": ("MLS", "USA", 1),
}


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("all", b"", 0, 0, f"cannot decode {path}")


def parse_main_file(path: Path, season_code: str, div: str) -> pd.DataFrame:
    league, country, tier = MAIN_DIVS[div]
    df = _read_csv(path)
    needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    out = pd.DataFrame({
        "home_team": df["HomeTeam"].astype(str).str.strip(),
        "away_team": df["AwayTeam"].astype(str).str.strip(),
        "date": pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce"),
        "home_score": pd.to_numeric(df["FTHG"], errors="coerce"),
        "away_score": pd.to_numeric(df["FTAG"], errors="coerce"),
        "ht_home_score": pd.to_numeric(df.get("HTHG"), errors="coerce"),
        "ht_away_score": pd.to_numeric(df.get("HTAG"), errors="coerce"),
    })
    out["league_id"] = ""
    out["league_name"] = league
    out["country"] = country
    out["league_tier"] = tier
    # 2526 -> 2025
    out["season"] = 2000 + int(season_code[:2])
    out["home_team_id"] = ""
    out["away_team_id"] = ""
    return out


def parse_extra_file(path: Path, code: str) -> pd.DataFrame:
    league, country, tier = EXTRA_FILES[code]
    df = _read_csv(path)
    needed = {"Date", "Home", "Away", "HG", "AG"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    out = pd.DataFrame({
        "home_team": df["Home"].astype(str).str.strip(),
        "away_team": df["Away"].astype(str).str.strip(),
        "date": pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce"),
        "home_score": pd.to_numeric(df["HG"], errors="coerce"),
        "away_score": pd.to_numeric(df["AG"], errors="coerce"),
    })
    out["ht_home_score"] = ""
    out["ht_away_score"] = ""
    out["league_id"] = ""
    out["league_name"] = league
    out["country"] = country
    out["league_tier"] = tier
    # Season column like "2012/2013" or plain year
    season_raw = df.get("Season")
    if season_raw is not None:
        out["season"] = season_raw.astype(str).str[:4]
    else:
        out["season"] = out["date"].dt.year
    out["home_team_id"] = ""
    out["away_team_id"] = ""
    return out


def build_name_map(existing: pd.DataFrame, new: pd.DataFrame) -> dict:
    """
    For each (league_name, country) present in BOTH datasets, fuzzy-match the
    football-data team names to the API-Football names so each physical team
    has ONE name in the merged file.
    """
    name_map = {}
    existing_keys = set(zip(existing["league_name"], existing["country"]))
    new_keys = set(zip(new["league_name"], new["country"]))

    for key in sorted(existing_keys & new_keys):
        lg, ct = key
        ex_teams = set(existing.loc[
            (existing["league_name"] == lg) & (existing["country"] == ct), "home_team"
        ]) | set(existing.loc[
            (existing["league_name"] == lg) & (existing["country"] == ct), "away_team"
        ])
        nw_teams = set(new.loc[
            (new["league_name"] == lg) & (new["country"] == ct), "home_team"
        ]) | set(new.loc[
            (new["league_name"] == lg) & (new["country"] == ct), "away_team"
        ])

        mapped = 0
        for t in nw_teams:
            if t in ex_teams:
                continue
            match = difflib.get_close_matches(t, list(ex_teams), n=1, cutoff=0.62)
            if match:
                name_map[(lg, ct, t)] = match[0]
                mapped += 1
        if mapped:
            print(f"  {lg} ({ct}): mapped {mapped} team-name variants")
    return name_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 1. Parse all raw files
    frames = []
    for f in sorted(RAW_DIR.glob("main_*.csv")):
        _, season_code, div = f.stem.split("_")
        frames.append(parse_main_file(f, season_code, div))
    for f in sorted(RAW_DIR.glob("extra_*.csv")):
        code = f.stem.split("_")[1]
        frames.append(parse_extra_file(f, code))

    new = pd.concat([fr for fr in frames if len(fr)], ignore_index=True)
    new = new.dropna(subset=["date", "home_score", "away_score"])
    new = new[(new["home_team"] != "nan") & (new["away_team"] != "nan")]
    new["home_score"] = new["home_score"].astype(int)
    new["away_score"] = new["away_score"].astype(int)
    print(f"Parsed {len(new):,} matches from football-data.co.uk "
          f"({new['date'].min().date()} to {new['date'].max().date()})")

    # 2. Load existing canonical file
    existing = pd.read_csv(MATCHES_CSV, low_memory=False)
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
    print(f"Existing matches.csv: {len(existing):,} rows")

    # 3. Align team names for overlapping leagues
    print("\nAligning team names for overlapping leagues:")
    name_map = build_name_map(existing, new)

    def remap(row_team, lg, ct):
        return name_map.get((lg, ct, row_team), row_team)

    if name_map:
        new["home_team"] = [
            remap(t, lg, ct) for t, lg, ct in
            zip(new["home_team"], new["league_name"], new["country"])
        ]
        new["away_team"] = [
            remap(t, lg, ct) for t, lg, ct in
            zip(new["away_team"], new["league_name"], new["country"])
        ]

    # 4. Dedup: drop new rows that already exist (same date + teams)
    existing["_key"] = (
        existing["date"].dt.strftime("%Y-%m-%d") + "|" +
        existing["home_team"].astype(str) + "|" + existing["away_team"].astype(str)
    )
    new["_key"] = (
        new["date"].dt.strftime("%Y-%m-%d") + "|" +
        new["home_team"].astype(str) + "|" + new["away_team"].astype(str)
    )
    before = len(new)
    new = new[~new["_key"].isin(set(existing["_key"]))]
    new = new.drop_duplicates(subset="_key")
    print(f"\nDedup: {before:,} -> {len(new):,} new rows "
          f"({before - len(new):,} duplicates removed)")

    # 5. Merge + write
    new["date"] = new["date"].dt.strftime("%Y-%m-%d")
    existing["date"] = existing["date"].dt.strftime("%Y-%m-%d")
    merged = pd.concat(
        [existing[CSV_COLUMNS], new[CSV_COLUMNS]], ignore_index=True
    ).sort_values("date").reset_index(drop=True)

    print(f"Merged total: {len(merged):,} matches, "
          f"{merged.groupby(['league_name','country']).ngroups} leagues")

    if args.dry_run:
        print("\nDRY RUN - not writing.")
        return

    # Backup then write
    backup = MATCHES_CSV.with_suffix(".csv.bak")
    MATCHES_CSV.replace(backup)
    merged.to_csv(MATCHES_CSV, index=False, encoding="utf-8")
    print(f"\nBackup saved: {backup}")
    print(f"Written: {MATCHES_CSV} ({MATCHES_CSV.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
