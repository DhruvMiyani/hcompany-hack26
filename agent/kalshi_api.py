"""
Kalshi World Cup market data via REST API.

Discovers KXWC* event tickers from parlay market legs, then fetches
each event group directly. Returns only pure FIFA markets — no mixed parlays.
"""

import requests
from typing import Optional

from .decision import Market

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

# Market categories in priority order (most liquid / easiest to model first)
PRIORITY_CATEGORIES = [
    "KXWCGAME",       # Full-time result (most liquid)
    "KXWCBTTS",       # Both teams to score
    "KXWCTOTAL",      # Total goals over/under
    "KXWC1H",         # First-half result
    "KXWCADVANCE",    # Advance to next round
    "KXWCSPREAD",     # Handicap / spread
    "KXWC1HTOTAL",    # First-half total goals
    "KXWCFIRSTGOAL",  # First goalscorer
]

CATEGORY_LABELS = {
    "KXWCGAME":      "match_winner",
    "KXWCBTTS":      "both_teams_score",
    "KXWCTOTAL":     "total_goals",
    "KXWC1H":        "first_half_winner",
    "KXWCADVANCE":   "advance",
    "KXWCSPREAD":    "spread",
    "KXWC1HTOTAL":   "first_half_totals",
    "KXWCFIRSTGOAL": "first_goalscorer",
    "KXWCGOAL":      "anytime_goalscorer",
    "KXWCCORNERS":   "corners",
    "KXWCTCORNERS":  "team_corners",
    "KXWC1HBTTS":    "first_half_btts",
    "KXWC1HSPREAD":  "first_half_spread",
}

MATCH_MAP = {
    "26JUL11ARGSUI": "ARG vs SUI",
    "26JUL14FRAESP": "FRA vs ESP",
}


def get_open_wc_markets(max_per_category: int = 6) -> list[Market]:
    """
    Discover all open KXWC World Cup markets.

    Strategy:
    1. Scan parlay legs to find KXWC event tickers.
    2. Query each event ticker for its individual markets.
    3. Return markets with real pricing, prioritised by category liquidity.
    """
    event_tickers = _discover_kxwc_events()
    if not event_tickers:
        return []

    # Group by category prefix, prioritise
    by_category: dict[str, list[str]] = {}
    for et in event_tickers:
        prefix = _category_prefix(et)
        by_category.setdefault(prefix, []).append(et)

    ordered_events: list[str] = []
    seen = set()
    for cat in PRIORITY_CATEGORIES:
        for et in by_category.get(cat, []):
            if et not in seen:
                ordered_events.append(et)
                seen.add(et)
    for et in event_tickers:
        if et not in seen:
            ordered_events.append(et)
            seen.add(et)

    markets: list[Market] = []
    cat_counts: dict[str, int] = {}

    for et in ordered_events:
        prefix = _category_prefix(et)
        if cat_counts.get(prefix, 0) >= max_per_category:
            continue

        fetched = _fetch_event_markets(et)
        cat_counts[prefix] = cat_counts.get(prefix, 0) + len(fetched)
        markets.extend(fetched)

    return markets


def get_historical_wc_context() -> dict:
    """
    Fetch any settled KXWC markets for historical context.
    Also returns current price snapshots as baseline.

    Returns a dict with:
        settled: list of resolved markets with results
        snapshot: current open market prices
    """
    settled = []
    try:
        r = requests.get(
            f"{KALSHI_API}/markets",
            params={"limit": 200, "status": "settled"},
            timeout=15,
        )
        if r.status_code == 200:
            for m in r.json().get("markets", []):
                if m.get("event_ticker", "").startswith("KXWC") or \
                   m.get("ticker", "").startswith("KXWC"):
                    settled.append({
                        "ticker": m.get("ticker"),
                        "title": m.get("title"),
                        "result": m.get("result"),
                        "expiration_value": m.get("expiration_value"),
                    })
    except Exception:
        pass

    open_markets = get_open_wc_markets()
    snapshot = [
        {
            "ticker": m.ticker,
            "name": m.name,
            "yes_price": m.yes_price,
            "no_price": m.no_price,
            "volume": m.volume,
        }
        for m in open_markets
    ]

    return {"settled": settled, "snapshot": snapshot}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _discover_kxwc_events() -> list[str]:
    """Scan open parlay markets to find all KXWC event tickers."""
    try:
        r = requests.get(
            f"{KALSHI_API}/markets",
            params={"limit": 500, "status": "open"},
            timeout=15,
        )
        r.raise_for_status()
        markets = r.json().get("markets", [])
    except Exception:
        return []

    found: set[str] = set()
    for m in markets:
        legs = m.get("mve_selected_legs", []) or []
        for leg in legs:
            et = leg.get("event_ticker", "")
            if et.startswith("KXWC"):
                found.add(et)
            mt = leg.get("market_ticker", "")
            if mt.startswith("KXWC"):
                # Derive event ticker by stripping last segment
                parts = mt.rsplit("-", 1)
                if len(parts) == 2:
                    found.add(parts[0])

    return sorted(found)


def _fetch_event_markets(event_ticker: str) -> list[Market]:
    """Fetch all markets under one event ticker."""
    try:
        r = requests.get(
            f"{KALSHI_API}/markets",
            params={"event_ticker": event_ticker, "status": "open", "limit": 50},
            timeout=10,
        )
        r.raise_for_status()
        raw = r.json().get("markets", [])
    except Exception:
        return []

    result = []
    for m in raw:
        yes_price = _price(m.get("yes_ask_dollars"))
        if yes_price is None or yes_price == 0.0:
            yes_price = _price(m.get("yes_bid_dollars"))
        if yes_price is None or yes_price == 0.0:
            continue

        no_price = _price(m.get("no_ask_dollars"))
        if no_price is None or no_price >= 0.99:
            no_price = round(1.0 - yes_price, 3)

        last_price = _price(m.get("last_price_dollars"))
        prev_price = _price(m.get("previous_price_dollars"))
        momentum = (round(last_price - prev_price, 4)
                    if last_price and prev_price else None)
        volume = float(m.get("volume_fp") or 0) or None
        close_time = (m.get("close_time") or "")[:16].replace("T", " ")

        ticker = m.get("ticker", "")
        prefix = _category_prefix(event_ticker)
        category = CATEGORY_LABELS.get(prefix, prefix.lower())

        match = "Unknown"
        for code, label in MATCH_MAP.items():
            if code in event_ticker:
                match = label
                break

        title = m.get("title") or ticker
        display_name = f"[{match}] {title}" if match != "Unknown" else title

        result.append(Market(
            name=display_name,
            ticker=ticker,
            yes_price=yes_price,
            no_price=no_price,
            volume=volume,
            closes=close_time or None,
            category=category,
            match=match,
            last_price=last_price,
            outcome=m.get("yes_sub_title") or None,
            momentum=momentum,
        ))

    return result


def _category_prefix(event_ticker: str) -> str:
    """Extract KXWC category from event ticker e.g. KXWCGAME → KXWCGAME."""
    for cat in CATEGORY_LABELS:
        if event_ticker.startswith(cat):
            return cat
    # Fallback: first hyphen-delimited segment starting with KXWC
    parts = event_ticker.split("-")
    return parts[0] if parts else event_ticker


def _price(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return round(f / 100.0, 3) if f > 1 else round(f, 3)
    except (TypeError, ValueError):
        return None
