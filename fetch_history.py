#!/usr/bin/env python3
"""
Backfill training data from the last two FIFA World Cups (2018, 2022).

  python fetch_history.py

Kalshi didn't exist for those tournaments, so there are no real market
prices. Instead each historical match becomes three pseudo-markets
(team1 / team2 / tie) with:
  - label   = the real result (from the public international-results dataset)
  - elo_*   = OUR OWN Elo, computed by replaying every international match
              since 1990 up to the day of the game (no lookahead)
  - price   = Elo-implied probability (a proxy — flagged is_history=1 so the
              model can discount it vs a real market price)

Output: data/dataset_history.json — merged into TRAINING ONLY by the tabular
research loop. The test set stays 100% real 2026 Kalshi markets.
"""

import csv
import io
import json
from pathlib import Path

import requests

OUT = Path("data") / "dataset_history.json"
RESULTS_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")

K_BASE, K_WC, HOME_ADV, START_ELO = 30, 60, 100, 1500


def replay_elo_and_collect() -> list[dict]:
    r = requests.get(RESULTS_URL, timeout=30)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))

    elo: dict[str, float] = {}
    examples = []

    def rating(team):
        return elo.get(team, START_ELO)

    for row in rows:
        date = row["date"]
        if date < "1990-01-01":
            continue
        h, a = row["home_team"], row["away_team"]
        try:
            hs, as_ = int(row["home_score"]), int(row["away_score"])
        except ValueError:
            continue
        neutral = row["neutral"] == "TRUE"
        is_wc = row["tournament"] == "FIFA World Cup"
        wc_year = date[:4]

        rh, ra = rating(h), rating(a)
        adv = 0 if neutral else HOME_ADV
        expected_h = 1 / (1 + 10 ** ((ra - rh - adv) / 400))

        # snapshot BEFORE updating — features must not see the result
        if is_wc and wc_year in ("2018", "2022"):
            diff = rh - ra          # neutral venue at a WC
            we = 1 / (1 + 10 ** (-diff / 400))
            p_draw = max(0.10, 0.29 - 0.20 * abs(we - 0.5))
            p1 = round((1 - p_draw) * we, 4)
            p2 = round((1 - p_draw) * (1 - we), 4)
            outcome = ("team1" if hs > as_ else
                       "team2" if as_ > hs else "tie")
            event = f"WCH{wc_year}-{h[:3].upper()}{a[:3].upper()}-{date}"
            for side, price, won in (
                (h, p1, outcome == "team1"),
                (a, p2, outcome == "team2"),
                ("Tie", round(p_draw, 4), outcome == "tie"),
            ):
                edge = (round(diff / 400, 4) if side == h else
                        round(-diff / 400, 4) if side == a else 0.0)
                examples.append({
                    "ticker": f"{event}-{side[:3].upper()}",
                    "event": event,
                    "category": "match_winner",
                    "title": f"{h} vs {a} Winner? ({wc_year} WC)",
                    "outcome": side if side != "Tie" else "Tie",
                    "close_time": date,
                    "yes_price_4h": min(max(price, 0.02), 0.98),
                    "yes_price_24h": min(max(price, 0.02), 0.98),
                    "momentum": 0.0,
                    "open_interest": 0.0,
                    "total_volume": 0.0,
                    "result": 1 if won else 0,
                    "is_history": 1,
                    "has_elo": 1,
                    "elo_edge": edge,
                    "elo_absdiff": round(abs(diff) / 400, 4),
                })

        # Elo update (after snapshot)
        score_h = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        k = K_WC if is_wc else K_BASE
        goal_mult = 1.0 + 0.5 * min(abs(hs - as_), 3)
        delta = k * goal_mult * (score_h - expected_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    return examples


if __name__ == "__main__":
    examples = replay_elo_and_collect()
    OUT.write_text(json.dumps(examples, indent=1))
    by_year = {}
    for e in examples:
        by_year[e["event"][3:7]] = by_year.get(e["event"][3:7], 0) + 1
    yes = sum(e["result"] for e in examples)
    print(f"{len(examples)} pseudo-markets from WC 2018+2022 → {OUT}")
    print(f"  by year: {by_year} | yes-rate {yes / len(examples):.2%}")
