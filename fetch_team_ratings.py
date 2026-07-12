#!/usr/bin/env python3
"""
Fetch World Football Elo ratings (eloratings.net) for all WC-2026 teams
appearing in our Kalshi tickers → data/team_ratings.json

  python fetch_team_ratings.py

Elo is the team-strength feature: it summarizes decades of results,
including everything that happened in this tournament (updated weekly).
"""

import json
from pathlib import Path

import requests

OUT = Path("data") / "team_ratings.json"

# Kalshi ticker code (IOC-style) → eloratings.net 2-letter code
KALSHI_TO_ELO = {
    "ARG": "AR", "AUS": "AU", "AUT": "AT", "BEL": "BE", "BIH": "BA",
    "BRA": "BR", "CAN": "CA", "CIV": "CI", "COD": "CD", "COL": "CO",
    "CPV": "CV", "CRO": "HR", "CUW": "CW", "CZE": "CZ", "DZA": "DZ",
    "ECU": "EC", "EGY": "EG", "ENG": "EN", "ESP": "ES", "FRA": "FR",
    "GER": "DE", "GHA": "GH", "HTI": "HT", "IRI": "IR", "IRQ": "IQ",
    "JOR": "JO", "JPN": "JP", "KOR": "KR", "KSA": "SA", "MAR": "MA",
    "MEX": "MX", "NED": "NL", "NOR": "NO", "NZL": "NZ", "PAN": "PA",
    "PAR": "PY", "POR": "PT", "QAT": "QA", "RSA": "ZA", "SCO": "SC",
    "SEN": "SN", "SUI": "CH", "SWE": "SE", "TUN": "TN", "TUR": "TR",
    "URU": "UY", "USA": "US", "UZB": "UZ",
}


def main():
    r = requests.get("https://www.eloratings.net/World.tsv", timeout=15)
    r.raise_for_status()
    elo_by_code = {}
    for line in r.text.splitlines():
        parts = line.split("\t")
        if len(parts) > 3 and parts[2] and parts[3].isdigit():
            elo_by_code[parts[2]] = int(parts[3])

    ratings, missing = {}, []
    for kalshi, elo_code in KALSHI_TO_ELO.items():
        if elo_code in elo_by_code:
            ratings[kalshi] = elo_by_code[elo_code]
        else:
            missing.append(kalshi)

    OUT.write_text(json.dumps(ratings, indent=1, sort_keys=True))
    print(f"{len(ratings)} teams rated → {OUT}")
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:8]
    for code, elo in top:
        print(f"  {code}: {elo}")
    if missing:
        print(f"MISSING: {missing}")


if __name__ == "__main__":
    main()
