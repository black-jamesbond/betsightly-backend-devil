"""
Accumulator builder — assembles the safest route to a target odds figure.

Uses the calibrated hybrid predictions for a date, converts each market's
probability into FAIR decimal odds (1/p), then greedily stacks the highest
probability legs until the combined odds reach the target.

"Safest" = fewest, most-confident legs that still clear the target, preferring
matches with real historical data behind them (HIGH/MEDIUM data quality).

Usage:
    python build_accas.py [--date YYYY-MM-DD] [--targets 2,5,10,100]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache" / "apifootball"

# Kick-offs are stored in UTC (what the feeds send); print them in local time.
try:
    from fixtures_livescore import to_local_hhmm as _local
except Exception:
    def _local(value, *_args, **_kwargs):
        return value

# market -> (probability field, selection label, minimum probability to consider)
# Accas use ONLY the Over goals markets — no 1X2 (home/away) or BTTS.
MARKETS = [
    ("over_1.5", "Over 1.5", 0.55),
    ("over_2.5", "Over 2.5", 0.50),
]

# The Over/Under 1.5 line on its own, for the dashboard's "1.5 only" filter.
MARKETS_OVER_15 = [MARKETS[0]]


def resolve_markets(name=None):
    """Market list for a filter name.

    "over_1.5" (or "1.5") restricts tickets to the Over 1.5 line; anything else
    keeps the default mix of Over 1.5 and Over 2.5.
    """
    key = (name or "").strip().lower()
    return MARKETS_OVER_15 if key in ("over_1.5", "o1.5", "1.5") else MARKETS

# Leg-odds preferences: ignore anything shorter than MIN_LEG_ODDS (too stingy to
# be worth a slot) and aim each leg near SWEET_ODDS.
MIN_LEG_ODDS = 1.25
MAX_LEG_ODDS = 1.75
SWEET_ODDS = 1.30


def load(date_str):
    f = CACHE / f"predictions_{date_str}.json"
    if not f.exists():
        raise SystemExit(f"No predictions for {date_str} — run predict_today.py first.")
    with open(f, "r", encoding="utf-8") as fh:
        return json.load(fh)


def kickable_ids(date_str):
    """match_ids that have NOT kicked off yet — you can only back these.
    The prediction cache carries no status, so read it from the fixtures."""
    ev = CACHE / f"events_{date_str}.json"
    if not ev.exists():
        return None  # unknown; caller keeps everything
    with open(ev, "r", encoding="utf-8") as fh:
        fixtures = json.load(fh)
    ok = set()
    for m in fixtures:
        status = (m.get("match_status") or "").strip()
        started = (
            status not in ("", "Not Started")          # live minute / Finished / HT
            or m.get("match_live") == "1"
            or m.get("match_hometeam_score", "") != ""
        )
        if not started:
            ok.add(str(m.get("match_id", "")))
    return ok


def build_pool(preds, open_ids, min_quality=("HIGH", "MEDIUM"),
               min_odds=MIN_LEG_ODDS, max_odds=MAX_LEG_ODDS, sweet=SWEET_ODDS,
               odds_lookup=None, markets=None, started_ids=None,
               id_lookup=None):
    """One candidate leg per match PER MARKET.

    A match can contribute both its Over 1.5 and its Over 2.5 leg, because the
    rules allow the same fixture on two tickets as long as the *option* differs.
    Each leg carries a `selection_key` (match + market) which is what ticket
    building now treats as unique.

    Pass `odds_lookup(home, away) -> (event, kind)` to price legs at SportyBet's
    real Over/Under lines instead of fair odds (1/probability). In that mode a
    leg only qualifies if SportyBet is actually offering the price, and every
    leg carries the identifiers its booking code needs.

    `markets` restricts which lines may be used — pass resolve_markets("over_1.5")
    to build tickets from the Over/Under 1.5 line only.

    `started_ids` are matches that have already kicked off. Those are allowed
    through without a SportyBet price (priced at fair odds instead) so a ticket
    built this morning keeps appearing — and settles — after its games play.

    `id_lookup(home, away) -> (event, kind)` supplies booking-code ids for a leg
    the board can no longer price — an in-play or finished game. It is only
    consulted when there is no live price, and it is what keeps a ticket that
    has already kicked off bookable.
    """
    markets = markets or MARKETS
    pool = []
    for p in preds:
        # Only matches that haven't kicked off yet
        if open_ids is not None and str(p.get("match_id", "")) not in open_ids:
            continue
        if p.get("data_quality") not in min_quality:
            continue

        event = None
        match_id = str(p.get("match_id", ""))
        still_to_play = started_ids is None or match_id not in started_ids
        if odds_lookup is not None:
            event, _kind = odds_lookup(p["home_team"], p["away_team"])
            if not event and still_to_play:
                continue        # not on SportyBet and not started: nothing to price

        cands = []
        for field, label, floor in markets:
            prob = (p.get(field) or 0) / 100.0
            if prob < floor:
                continue
            price = (event or {}).get("prices", {}).get(field) if event else None
            if odds_lookup is not None and not price and still_to_play:
                continue        # the line exists on SportyBet but not this total
            odds = (price["odds"] if price else round(1.0 / prob, 2))
            if odds < min_odds or odds > max_odds:
                continue
            cands.append((odds, label, prob, price))
        if not cands:
            continue

        for odds, label, prob, price in cands:
            match_id = str(p.get("match_id", ""))
            leg = {
                "match_id": match_id,
                "match": f"{p['home_team']} vs {p['away_team']}",
                "home": p["home_team"],
                "away": p["away_team"],
                "league": p.get("league", ""),
                "kick_off": p.get("kick_off", ""),
                # Raw UTC kick-off, kept next to the local one so the leg can be
                # matched back to its own fixture later (see leg_ids_for).
                "kick_off_utc": p.get("kick_off", ""),
                "selection": label,
                "field": field,
                "prob": prob,
                "odds": odds,
                "quality": p.get("data_quality"),
                "fair_odds": round(1.0 / prob, 2),
                # What must be unique between tickets: the option, not the match.
                "selection_key": f"{match_id}|{label}",
            }
            ids = leg_ids(event if price else None, field)
            if ids is None and id_lookup is not None:
                # The board has dropped this fixture, but the live board or the
                # ledger may still know its identifiers.
                other, _kind = id_lookup(p["home_team"], p["away_team"])
                ids = leg_ids_for(other, leg)
            if ids:
                leg["sb"] = ids
            pool.append(leg)
    # order by closeness to the sweet spot so accas are built from ~1.30 legs out
    pool.sort(key=lambda x: abs(x["odds"] - sweet))
    return pool


def leg_ids(event, field, side="over"):
    """SportyBet's identifiers for one Over/Under line of an event, or None.

    Kept here rather than in the SportyBet client so every caller — the builder,
    the dashboard's settlement step and the booking-code route — resolves ids the
    same way, including for legs read back out of a frozen snapshot.

    `side` is "over" or "under": the dashboard's Over/Under 1.5 panels use the
    *under* line whenever the model rates the game below 50% for goals.
    """
    if not event:
        return None
    price = (event.get("prices") or {}).get(field)
    if not price:
        return None
    if side == "under":
        outcome, odds = price.get("under_outcome_id"), price.get("under_odds")
    else:
        outcome, odds = price.get("outcome_id"), price.get("odds")
    if not outcome:
        return None
    return {
        "event_id": event.get("event_id"),
        "market_id": price["market_id"],
        "specifier": price["specifier"],
        "outcome_id": outcome,
        "odds": odds,
    }


def _hhmm_minutes(value):
    """Minutes since midnight for an "HH:MM" string, or None."""
    try:
        hour, minute = str(value).split(":")[:2]
        return (int(hour) % 24) * 60 + int(minute)
    except (TypeError, ValueError):
        return None


def event_matches_leg(event, leg, started=None, now_ms=None):
    """Is `event` really this leg's fixture, or just the same two teams?

    Names are not enough on their own: the same pairing can turn up twice in one
    day (a league game and a cup tie, a rescheduled replay), and binding a leg to
    the wrong meeting would hand back a booking code for the wrong match. So

    * a leg that carries its raw UTC kick-off must agree with the event's start
      time to within an hour, and
    * a leg whose game has already kicked off (`started=True` — fixtures frozen
      before the raw time was stored) only accepts an event that has also
      kicked off, never a later meeting of the same clubs.
    """
    if not event or not event.get("event_id") or not event.get("prices"):
        return False
    start = event.get("start_ms")
    if not start:
        return True                     # nothing to check the fixture against

    leg_minutes = _hhmm_minutes(leg.get("kick_off_utc"))
    if leg_minutes is not None:
        event_minutes = int(start // 60000) % 1440    # UTC minutes since midnight
        gap = abs(leg_minutes - event_minutes)
        return min(gap, 1440 - gap) <= 60   # tolerates a 1h zone difference

    if started:
        now = now_ms if now_ms is not None else time.time() * 1000
        return int(start) <= now + 600_000
    return True


def leg_ids_for(event, leg, started=None, side="over"):
    """leg_ids() for a leg, refusing an event belonging to a different fixture."""
    if not event_matches_leg(event, leg, started=started):
        return None
    return leg_ids(event, leg_field(leg), side=side)


def leg_teams(leg):
    """(home, away) for a leg, from its own fields or its 'A vs B' label."""
    home, away = leg.get("home"), leg.get("away")
    if home and away:
        return home, away
    parts = str(leg.get("match", "")).split(" vs ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else ("", "")


def leg_field(leg):
    """Which Over/Under line a leg sits on, e.g. "over_1.5"."""
    field = leg.get("field")
    if field:
        return field
    for key, label, _floor in MARKETS:
        if label == leg.get("selection"):
            return key
    return ""


def build_acca(pool, target, used_selections):
    """Greedily add the best unused options until combined odds >= target.

    `used_selections` holds selection keys already spent on other tickets, so an
    option never appears twice. The same match may turn up on another ticket with
    a different option (Over 1.5 there, Over 2.5 here); it just cannot appear
    twice inside one ticket.
    """
    legs, odds, prob = [], 1.0, 1.0
    in_ticket = set()
    for c in pool:
        if odds >= target:
            break
        if c["selection_key"] in used_selections:
            continue
        if c["match"] in in_ticket:
            continue
        legs.append(c)
        in_ticket.add(c["match"])
        odds *= c["odds"]
        prob *= c["prob"]
    if odds < target:
        return None
    return {"legs": legs, "odds": round(odds, 2), "prob": prob}


def build_ticket_set(pool, target, n=10, seed=0, used_selections=None):
    """Up to `n` tickets for a target, each option used at most once.

    Uniqueness is enforced on the *selection* (match + market), not the match: a
    fixture can back two different tickets with two different options, but no
    option — say "Man City home" or "Arsenal Over 1.5" — is repeated across the
    set. Deterministic (fixed seed) so the same tickets persist across refreshes
    and keep their tracked results.

    Pass a shared `used_selections` set to hold that rule across several calls —
    the dashboard does, so no option repeats anywhere on the page (every tier,
    every section), not merely within one tier.
    """
    import random
    tickets, seen = [], set()
    used_sel = used_selections if used_selections is not None else set()

    # Phase 1: greedily take the best options still unused
    while len(tickets) < n:
        a = build_acca(pool, target, used_sel)
        if not a:
            break
        key = frozenset(l["selection_key"] for l in a["legs"])
        if key in seen:
            break
        seen.add(key)
        tickets.append(a)
        for l in a["legs"]:
            used_sel.add(l["selection_key"])

    # Phase 2: top up with shuffled combinations, still never reusing an option
    rng = random.Random(seed)
    attempts = 0
    while len(tickets) < n and attempts < n * 40:
        attempts += 1
        shuffled = pool[:]
        rng.shuffle(shuffled)
        a = build_acca(shuffled, target, used_sel)
        if not a:
            break                        # nothing left to build from
        key = frozenset(l["selection_key"] for l in a["legs"])
        if key in seen:
            continue
        seen.add(key)
        tickets.append(a)
        for l in a["legs"]:
            used_sel.add(l["selection_key"])
    return tickets


def main(date_str, targets):
    preds = load(date_str)
    open_ids = kickable_ids(date_str)
    pool = build_pool(preds, open_ids)
    if not pool:
        raise SystemExit("No qualifying selections (need HIGH/MEDIUM data quality "
                         "on a match that hasn't kicked off).")

    print(f"\nACCUMULATOR SELECTIONS — {date_str}")
    print(f"Candidate pool: {len(pool)} upcoming matches with usable data\n")

    for t in targets:
        acca = build_acca(pool, t, set())
        print("=" * 74)
        if not acca:
            print(f"  ~{t} ODDS — not reachable from today's pool")
            continue
        print(f"  ~{t} ODDS TARGET  ->  {acca['odds']} fair odds   "
              f"| {len(acca['legs'])} legs | {acca['prob']*100:.1f}% chance to land")
        print("=" * 74)
        for i, l in enumerate(acca["legs"], 1):
            # Stored kick-offs are UTC; print them in local time.
            print(f"  {i:>2}. {_local(l['kick_off']):>5}  {l['match'][:40]:<40} "
                  f"{l['selection']:<10} @{l['odds']:<5} ({l['prob']*100:.0f}%) [{l['quality']}]")
        stake = 10
        print(f"      ${stake} returns ${stake * acca['odds']:.2f} "
              f"if all {len(acca['legs'])} land")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--targets", default="2,5,10,100")
    a = ap.parse_args()
    main(a.date, [float(x) for x in a.targets.split(",")])
