"""
Predict the games SportyBet is actually offering.

The card used to be built from a free fixture feed, which meant much of it was
games you could not back. This pulls SportyBet's own list instead and runs the
same calibrated engine over it, writing the same predictions cache the
dashboard already reads — so every game on the card is one you can bet on, and
the odds behind it are SportyBet's own.

Live scores and final results still come from the LiveScore fixture feed: a game
that matches one of its fixtures keeps that fixture's id, which is what the
dashboard's live status and settlement are keyed on. Games it cannot match (the
long tail of small leagues) carry SportyBet's own event id instead and simply
have no live score, which the dashboard already renders as "no live data".

Usage:
    python predict_sportybet.py [--date YYYY-MM-DD] [--limit N] [--print]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache" / "apifootball"

import predict_today as pt            # the calibrated engine, reused as-is
import sportybet_filter as sb


def _local_date(ms) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _utc_hhmm(ms) -> str:
    try:
        return datetime.utcfromtimestamp(int(ms) / 1000).strftime("%H:%M")
    except (TypeError, ValueError, OSError):
        return ""


def sportybet_games(date_str: str, max_age_min: int = 30,
                    exclude_started: bool = True) -> list:
    """SportyBet's games for a date, from their list (refreshed if stale)."""
    age = sb._cache_age_min()
    if age is None or age > max_age_min:
        events = sb.fetch_sportybet_events()
    else:
        events = json.loads(sb.CACHE.read_text(encoding="utf-8"))
    games = [e for e in events
             if e.get("start_ms") and _local_date(e["start_ms"]) == date_str]
    if exclude_started:
        # A game that has kicked off is no longer backable, and its prices have
        # left the board; it stays out of the card but its result still shows in
        # the "Played today" section of the tickets already built.
        games = [e for e in games
                 if str(e.get("status", "")).lower() in ("", "not start")]
    games.sort(key=lambda e: e["start_ms"])
    return games


def livescore_index(date_str: str) -> dict:
    """Today's LiveScore fixtures, keyed by team pair, for ids and results."""
    try:
        fixtures = pt.fetch_fixtures(date_str)
    except Exception as exc:
        print(f"  LiveScore fixtures unavailable ({exc}); using SportyBet ids")
        return {}
    index = {}
    for m in fixtures:
        home = m.get("match_hometeam_name", "")
        away = m.get("match_awayteam_name", "")
        if home and away:
            index.setdefault((pt.normalize_name(home), pt.normalize_name(away)), m)
    return index


def _livescore_match(index: dict, home: str, away: str):
    """The LiveScore fixture behind a SportyBet game, or None."""
    hn, an = pt.normalize_name(home), pt.normalize_name(away)
    if (hn, an) in index:
        return index[(hn, an)]
    if (an, hn) in index:
        return index[(an, hn)]
    pairs = set(index)
    teams = {t for pair in pairs for t in pair}
    if sb.match_kind(home, away, pairs, teams) is None:
        return None
    for (h2, a2), fixture in index.items():
        if sb._close(hn, h2) and sb._close(an, a2):
            return fixture
        if sb._close(hn, a2) and sb._close(an, h2):
            return fixture
    return None


def predict_game(game: dict, hist, index: dict) -> dict:
    """One calibrated read, in the shape the dashboard expects."""
    home, away = game["home"], game["away"]
    hs = pt.find_team_stats(home, hist)
    as_ = pt.find_team_stats(away, hist)
    pred = pt.predict_match(home, away, hs, as_, pt.get_h2h(home, away, hist))
    engine = "built-in"
    if pt._HYBRID is not None:
        pred = pt._HYBRID.apply(pred, pt.normalize_name(home), pt.normalize_name(away))
        engine = "hybrid-dc"
    elif pt._CALIBRATOR is not None:
        pred = pt._CALIBRATOR.apply(pred)
        engine = "calibrated"

    fixture = _livescore_match(index, home, away)
    row = dict(pred)
    row.update({
        "home_team": home,
        "away_team": away,
        "league": game.get("league", ""),
        "country": game.get("country", ""),
        "kick_off": _utc_hhmm(game.get("start_ms")),
        # The dashboard's live status and settlement are keyed on this id, so a
        # fixture we can follow keeps the LiveScore id; otherwise SportyBet's.
        "match_id": str((fixture or {}).get("match_id") or game.get("event_id") or ""),
        "sb_event_id": game.get("event_id", ""),
        "live_tracked": bool(fixture),
        # SportyBet's own word for the game's state ("Not start", "H1", "Ended"),
        # plus the score they show. Used where no live feed reaches the fixture.
        "sb_status": game.get("status", ""),
        "sb_score": game.get("score", ""),
        "engine": engine,
        "sportybet": True,          # by construction: these came from their list
        "source": "sportybet",
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict SportyBet's games")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of games (for a quick test run)")
    parser.add_argument("--include-started", action="store_true",
                        help="also predict games that have already kicked off")
    parser.add_argument("--print", action="store_true", dest="show")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  SPORTYBET CARD - {args.date}")
    print(f"{'=' * 70}")

    games = sportybet_games(args.date, exclude_started=not args.include_started)
    print(f"  {len(games)} games SportyBet offers on {args.date} (not yet started)")
    if not games:
        print("  Nothing to predict — their list covers a rolling 72 hours.")
        return 1
    if args.limit:
        games = games[:args.limit]
        print(f"  limited to the first {len(games)}")

    hist = pt.load_historical()
    if hist is None:
        print("  No historical data — predictions would be priors only.")
        return 1
    index = livescore_index(args.date)
    print(f"  {len(index)} LiveScore fixtures available for live scores")

    rows = []
    for i, game in enumerate(games, 1):
        try:
            rows.append(predict_game(game, hist, index))
        except Exception as exc:
            print(f"  skipped {game['home']} v {game['away']}: {exc}")
        if i % 100 == 0:
            print(f"  ... {i}/{len(games)} predicted")

    out = CACHE / f"predictions_{args.date}.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    tmp.replace(out)

    tracked = sum(1 for r in rows if r["live_tracked"])
    quality = {}
    for r in rows:
        quality[r.get("data_quality")] = quality.get(r.get("data_quality"), 0) + 1
    print(f"\n  {len(rows)} predictions written to {out.name}")
    print(f"  with live scores: {tracked} | quality: {quality}")
    if args.show:
        for r in sorted(rows, key=lambda x: x["kick_off"])[:25]:
            print("   %-5s %-34s %-9s %5s%%  %-8s %s"
                  % (r["kick_off"], f"{r['home_team']} v {r['away_team']}"[:34],
                     r["prediction"], r["confidence"], r.get("data_quality"),
                     "live" if r["live_tracked"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
