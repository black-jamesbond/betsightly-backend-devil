"""
Fit the production Dixon-Coles model on all labeled history and save it.

Run periodically (e.g. from the nightly automation) so ratings stay fresh.
Output: models/dixon_coles/model.json

Usage:
    python fit_dixon_coles.py
"""
from __future__ import annotations

import time
from pathlib import Path

import predict_today as pt
from dixon_coles import DixonColes

OUT = Path("models/dixon_coles/model.json")


def main():
    print("Loading historical data...")
    hist = pt.load_historical().sort_values("date").reset_index(drop=True)
    labeled = hist[hist["league"].notna()]
    print(f"Fitting Dixon-Coles on {len(labeled):,} labeled matches "
          f"({labeled['date'].min().date()} -> {labeled['date'].max().date()})")

    t0 = time.time()
    # as_of=None -> uses the latest available data; time-decay + lookback keep it fresh
    dc = DixonColes.fit(hist)
    print(f"Fitted {len(dc.leagues)} leagues in {time.time() - t0:.0f}s")

    dc.save(OUT)
    total_teams = sum(len(m.teams) for m in dc.leagues.values())
    print(f"Saved -> {OUT}  ({total_teams:,} team ratings across {len(dc.leagues)} leagues)")


if __name__ == "__main__":
    main()
