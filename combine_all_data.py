"""Combine all training data sources into one mega-CSV."""
import csv, glob, os, sys

OUT = "all_training_data.csv"
FIELDS = [
    "date", "season", "league", "country", "division_tier",
    "home_team", "away_team",
    "home_goals", "away_goals", "total_goals", "result",
    "ht_home_goals", "ht_away_goals", "ht_result",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellow", "away_yellow", "home_red", "away_red",
    "home_elo", "away_elo", "home_form3", "home_form5", "away_form3", "away_form5",
    "odd_home", "odd_draw", "odd_away", "odd_over25", "odd_under25",
    "source"
]

def safe_int(v):
    try:
        return str(int(float(v)))
    except:
        return ""

def safe_float(v):
    try:
        return str(round(float(v), 4))
    except:
        return ""

def result_code(v):
    v = str(v).strip().upper()
    if v in ("H", "HOME", "1"):
        return "H"
    if v in ("A", "AWAY", "2"):
        return "A"
    if v in ("D", "DRAW", "X"):
        return "D"
    return v

def load_real_football():
    rows = []
    with open("real_football_data.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            hg = safe_int(r.get("hgoal", ""))
            ag = safe_int(r.get("vgoal", ""))
            tg = safe_int(r.get("totgoal", ""))
            res = r.get("result", "")
            if res == "A":
                res = "A"
            elif res == "H":
                res = "H"
            elif res == "D":
                res = "D"
            rows.append({
                "date": r.get("Date", ""),
                "season": r.get("Season", ""),
                "league": "",
                "country": "",
                "division_tier": r.get("tier", ""),
                "home_team": r.get("home", ""),
                "away_team": r.get("visitor", ""),
                "home_goals": hg,
                "away_goals": ag,
                "total_goals": tg,
                "result": res,
                "source": "real_football_data"
            })
    return rows

def load_historical_raw():
    rows = []
    with open("data/historical/raw/matches.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            res = r.get("result", "").strip().lower()
            if res == "home_team":
                res = "H"
            elif res == "away_team":
                res = "A"
            elif res == "draw":
                res = "D"
            else:
                res = result_code(res)
            hg = safe_int(r.get("home_team_score", ""))
            ag = safe_int(r.get("away_team_score", ""))
            tg = ""
            if hg and ag:
                tg = str(int(hg) + int(ag))
            rows.append({
                "date": "",
                "season": r.get("season", ""),
                "league": r.get("division", ""),
                "country": "",
                "division_tier": r.get("tier", ""),
                "home_team": r.get("home_team_name", ""),
                "away_team": r.get("away_team_name", ""),
                "home_goals": hg,
                "away_goals": ag,
                "total_goals": tg,
                "result": res,
                "source": "historical_raw"
            })
    return rows

def load_github_football():
    rows = []
    with open("data/github-football/Matches.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            hg = safe_int(r.get("FTHome", ""))
            ag = safe_int(r.get("FTAway", ""))
            tg = ""
            if hg and ag:
                tg = str(int(hg) + int(ag))
            rows.append({
                "date": r.get("MatchDate", ""),
                "season": "",
                "league": r.get("Division", ""),
                "country": "",
                "division_tier": "",
                "home_team": r.get("HomeTeam", ""),
                "away_team": r.get("AwayTeam", ""),
                "home_goals": hg,
                "away_goals": ag,
                "total_goals": tg,
                "result": result_code(r.get("FTResult", "")),
                "ht_home_goals": safe_int(r.get("HTHome", "")),
                "ht_away_goals": safe_int(r.get("HTAway", "")),
                "ht_result": result_code(r.get("HTResult", "")),
                "home_shots": safe_int(r.get("HomeShots", "")),
                "away_shots": safe_int(r.get("AwayShots", "")),
                "home_shots_on_target": safe_int(r.get("HomeTarget", "")),
                "away_shots_on_target": safe_int(r.get("AwayTarget", "")),
                "home_corners": safe_int(r.get("HomeCorners", "")),
                "away_corners": safe_int(r.get("AwayCorners", "")),
                "home_fouls": safe_int(r.get("HomeFouls", "")),
                "away_fouls": safe_int(r.get("AwayFouls", "")),
                "home_yellow": safe_int(r.get("HomeYellow", "")),
                "away_yellow": safe_int(r.get("AwayYellow", "")),
                "home_red": safe_int(r.get("HomeRed", "")),
                "away_red": safe_int(r.get("AwayRed", "")),
                "home_elo": safe_float(r.get("HomeElo", "")),
                "away_elo": safe_float(r.get("AwayElo", "")),
                "home_form3": safe_float(r.get("Form3Home", "")),
                "home_form5": safe_float(r.get("Form5Home", "")),
                "away_form3": safe_float(r.get("Form3Away", "")),
                "away_form5": safe_float(r.get("Form5Away", "")),
                "odd_home": safe_float(r.get("OddHome", "")),
                "odd_draw": safe_float(r.get("OddDraw", "")),
                "odd_away": safe_float(r.get("OddAway", "")),
                "odd_over25": safe_float(r.get("Over25", "")),
                "odd_under25": safe_float(r.get("Under25", "")),
                "source": "github_football"
            })
    return rows

LEAGUE_MAP = {
    "E0": ("England", "Premier League"), "E1": ("England", "Championship"),
    "E2": ("England", "League One"), "E3": ("England", "League Two"),
    "EC": ("England", "Conference"), "SC0": ("Scotland", "Premiership"),
    "SC1": ("Scotland", "Championship"), "SC2": ("Scotland", "League One"),
    "SC3": ("Scotland", "League Two"), "D1": ("Germany", "Bundesliga"),
    "D2": ("Germany", "2. Bundesliga"), "I1": ("Italy", "Serie A"),
    "I2": ("Italy", "Serie B"), "SP1": ("Spain", "La Liga"),
    "SP2": ("Spain", "Segunda"), "F1": ("France", "Ligue 1"),
    "F2": ("France", "Ligue 2"), "N1": ("Netherlands", "Eredivisie"),
    "B1": ("Belgium", "Jupiler League"), "P1": ("Portugal", "Primeira Liga"),
    "T1": ("Turkey", "Super Lig"), "G1": ("Greece", "Super League"),
}

def load_football_data_uk():
    rows = []
    for fp in sorted(glob.glob("data/football-data-uk/raw/*.csv")):
        fname = os.path.basename(fp)
        # Parse season/league from filename like main_2526_E0.csv or extra_ARG.csv
        parts = fname.replace(".csv", "").split("_")
        season_code = ""
        league_code = ""
        if parts[0] == "main" and len(parts) >= 3:
            season_code = parts[1]
            league_code = parts[2]
        elif parts[0] == "extra":
            league_code = parts[1] if len(parts) > 1 else ""

        country, league_name = LEAGUE_MAP.get(league_code, ("", league_code))

        try:
            with open(fp, encoding="utf-8", errors="replace") as f:
                # Clean BOM
                content = f.read()
                if content.startswith("﻿"):
                    content = content[1:]
                import io
                reader = csv.DictReader(io.StringIO(content))
                for r in reader:
                    # Different files have different column names
                    home = r.get("Home", r.get("HomeTeam", r.get("HT", "")))
                    away = r.get("Away", r.get("AwayTeam", r.get("AT", "")))
                    hg = safe_int(r.get("HG", r.get("FTHG", "")))
                    ag = safe_int(r.get("AG", r.get("FTAG", "")))
                    res = result_code(r.get("Res", r.get("FTR", "")))
                    date = r.get("Date", "")
                    tg = ""
                    if hg and ag:
                        tg = str(int(hg) + int(ag))

                    ct = r.get("﻿Country", r.get("Country", country))
                    lg = r.get("League", league_name)

                    row = {
                        "date": date,
                        "season": season_code,
                        "league": lg,
                        "country": ct if ct else country,
                        "division_tier": "",
                        "home_team": home,
                        "away_team": away,
                        "home_goals": hg,
                        "away_goals": ag,
                        "total_goals": tg,
                        "result": res,
                        "ht_home_goals": safe_int(r.get("HTHG", "")),
                        "ht_away_goals": safe_int(r.get("HTAG", "")),
                        "ht_result": result_code(r.get("HTR", "")),
                        "home_shots": safe_int(r.get("HS", "")),
                        "away_shots": safe_int(r.get("AS", "")),
                        "home_shots_on_target": safe_int(r.get("HST", "")),
                        "away_shots_on_target": safe_int(r.get("AST", "")),
                        "home_corners": safe_int(r.get("HC", "")),
                        "away_corners": safe_int(r.get("AC", "")),
                        "home_fouls": safe_int(r.get("HF", "")),
                        "away_fouls": safe_int(r.get("AF", "")),
                        "home_yellow": safe_int(r.get("HY", "")),
                        "away_yellow": safe_int(r.get("AY", "")),
                        "home_red": safe_int(r.get("HR", "")),
                        "away_red": safe_int(r.get("AR", "")),
                        "odd_home": safe_float(r.get("PSCH", r.get("B365H", r.get("AvgCH", "")))),
                        "odd_draw": safe_float(r.get("PSCD", r.get("B365D", r.get("AvgCD", "")))),
                        "odd_away": safe_float(r.get("PSCA", r.get("B365A", r.get("AvgCA", "")))),
                        "odd_over25": safe_float(r.get("P>2.5", r.get("BbAv>2.5", ""))),
                        "odd_under25": safe_float(r.get("P<2.5", r.get("BbAv<2.5", ""))),
                        "source": "football_data_uk"
                    }
                    rows.append(row)
        except Exception as e:
            print(f"  Warning: {fp}: {e}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    print("Loading real_football_data.csv...")
    d1 = load_real_football()
    print(f"  {len(d1):,} rows")

    print("Loading data/historical/raw/matches.csv...")
    d2 = load_historical_raw()
    print(f"  {len(d2):,} rows")

    print("Loading data/github-football/Matches.csv...")
    d3 = load_github_football()
    print(f"  {len(d3):,} rows")

    print("Loading data/football-data-uk/raw/...")
    d4 = load_football_data_uk()
    print(f"  {len(d4):,} rows")

    all_rows = d1 + d2 + d3 + d4
    print(f"\nTotal combined: {len(all_rows):,} rows")

    print(f"Writing {OUT}...")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            clean = {k: row.get(k, "") for k in FIELDS}
            writer.writerow(clean)

    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    print(f"Done! {OUT} = {size_mb:.1f} MB, {len(all_rows):,} rows")
