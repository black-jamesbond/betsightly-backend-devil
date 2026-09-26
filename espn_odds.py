"""
Real bookmaker odds via ESPN's public API (which carries DraftKings prices).

DraftKings blocks direct scraping (403), but ESPN's free soccer scoreboard
embeds live DraftKings moneyline + Over/Under 2.5 lines. We pull those for the
major leagues ESPN covers, convert American odds to decimal, and key them by
normalized team names so they can be attached to our predictions.

Coverage note: ESPN/DK cover major leagues only (MLS, big Europe, some South
America) — NOT the friendlies / obscure lower leagues that dominate a typical
slate. So real odds attach to a MINORITY of matches, and the total line is 2.5
(DraftKings doesn't offer an Over 1.5 line here).

    from espn_odds import fetch_odds
    odds = fetch_odds("2026-08-01")   # {(home_norm, away_norm): {...decimal odds}}
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime

from predict_today import normalize_name

# Major soccer leagues ESPN exposes with DraftKings odds.
LEAGUES = [
    "eng.1", "eng.2", "eng.3", "eng.4", "esp.1", "esp.2", "ita.1", "ita.2",
    "ger.1", "ger.2", "fra.1", "fra.2", "ned.1", "por.1", "sco.1", "bel.1",
    "tur.1", "gre.1", "usa.1", "usa.nwsl", "mex.1", "bra.1", "arg.1", "chi.1",
    "col.1", "uru.1", "nor.1", "swe.1", "den.1", "fin.1", "irl.1", "rus.1",
    "aut.1", "sui.1", "pol.1", "rou.1", "jpn.1", "kor.1", "aus.1", "chn.1",
    "uefa.champions", "uefa.europa", "uefa.europa.conf", "conmebol.libertadores",
    "concacaf.league",
]

_ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard?dates={d}"


def _am2dec(american):
    """American odds string/number -> decimal odds."""
    if american in (None, "", "EVEN", "even"):
        return None
    try:
        a = float(str(american).replace("+", ""))
    except ValueError:
        return None
    if a == 0:
        return None
    return round(1 + a / 100, 3) if a > 0 else round(1 + 100 / (-a), 3)


def _close(node):
    """Prefer the closing line, fall back to open."""
    if not node:
        return None
    c = node.get("close") or node.get("open") or {}
    return c.get("odds")


def fetch_odds(date_str, verbose=False):
    """Return {(home_norm, away_norm): odds_dict} of real DraftKings decimal odds
    for the given date across the major leagues ESPN covers."""
    ymd = date_str.replace("-", "")
    out = {}
    leagues_hit = 0
    for lg in LEAGUES:
        try:
            req = urllib.request.Request(_ESPN.format(lg=lg, d=ymd),
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
        except Exception:
            continue
        events = data.get("events", [])
        got = 0
        for e in events:
            comp = (e.get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            home = away = None
            for c in competitors:
                nm = (c.get("team") or {}).get("displayName", "")
                if c.get("homeAway") == "home":
                    home = nm
                elif c.get("homeAway") == "away":
                    away = nm
            if not home or not away:
                continue
            odds_list = comp.get("odds") or []
            o = next((x for x in odds_list if isinstance(x, dict)), None)
            if not o:
                continue
            ml = o.get("moneyline") or {}
            tot = o.get("total") or {}
            rec = {
                "home": _am2dec(_close(ml.get("home"))),
                "draw": _am2dec((o.get("drawOdds") or {}).get("moneyLine")),
                "away": _am2dec(_close(ml.get("away"))),
                "over25": _am2dec(_close(tot.get("over"))),
                "under25": _am2dec(_close(tot.get("under"))),
                "ou_line": o.get("overUnder"),
                "provider": (o.get("provider") or {}).get("name", "DraftKings"),
                "espn_home": home, "espn_away": away, "league": lg,
            }
            if not any(rec[k] for k in ("home", "draw", "away", "over25")):
                continue
            out[(normalize_name(home), normalize_name(away))] = rec
            got += 1
        if got:
            leagues_hit += 1
            if verbose:
                print(f"  {lg}: {got} matches with odds")
    if verbose:
        print(f"ESPN/DraftKings: {len(out)} matches with real odds "
              f"across {leagues_hit} leagues for {date_str}")
    return out


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    odds = fetch_odds(d, verbose=True)
    print(f"\nSample real odds for {d}:")
    for (h, a), r in list(odds.items())[:8]:
        print(f"  {r['espn_home']} v {r['espn_away']:22s} "
              f"H {r['home']} / D {r['draw']} / A {r['away']}  |  "
              f"O{r['ou_line']} {r['over25']} / U {r['under25']}  [{r['league']}]")
