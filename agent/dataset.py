"""
Real training/test data from settled Kalshi World Cup markets.

Every example is a market the way the model WOULD have seen it before the
match (price snapshots from candlesticks at T-24h and T-4h before close) plus
the settled outcome (result: yes/no). No hindsight leakage: post-match prices
are never used as features.

Train/test split is BY EVENT (match), not by market — markets from the same
match are correlated (Winner-FRA, Winner-TIE, BTTS share the same game), so a
random per-market split would leak outcomes across the boundary.
"""

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .kalshi_api import KALSHI_API, CATEGORY_LABELS

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_PATH = DATA_DIR / "dataset_train.json"
TEST_PATH = DATA_DIR / "dataset_test.json"

# Core categories: liquid, binary, and modelable from price alone
DATASET_SERIES = [
    "KXWCGAME", "KXWCBTTS", "KXWCTOTAL", "KXWCADVANCE", "KXWC1H", "KXWCSPREAD",
]

TEST_FRACTION = 0.25


def _get(path: str, **params) -> dict:
    r = requests.get(f"{KALSHI_API}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_settled_markets(series: str, max_markets: int = 400) -> list[dict]:
    out, cursor = [], None
    while len(out) < max_markets:
        params = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        page = _get("/markets", **params)
        markets = page.get("markets", [])
        out.extend(m for m in markets if m.get("result") in ("yes", "no"))
        cursor = page.get("cursor")
        if not cursor or not markets:
            break
    return out[:max_markets]


def _snapshot_price(series: str, ticker: str, close_iso: str) -> dict | None:
    """Hourly candles for the 30h before close → prices at T-24h and T-4h."""
    try:
        close = datetime.fromisoformat(close_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    end = close - timedelta(hours=4)
    start = close - timedelta(hours=30)
    try:
        d = _get(
            f"/series/{series}/markets/{ticker}/candlesticks",
            start_ts=int(start.timestamp()), end_ts=int(end.timestamp()),
            period_interval=60,
        )
    except requests.RequestException:
        return None
    candles = d.get("candlesticks", [])
    if not candles:
        return None

    def price(c):
        p = (c.get("price") or {}).get("close_dollars")
        return float(p) if p else None

    first, last = candles[0], candles[-1]
    p24, p4 = price(first), price(last)
    if p4 is None or not (0.01 <= p4 <= 0.99):
        return None
    oi = float(last.get("open_interest_fp") or 0)
    return {
        "yes_price_24h": p24,
        "yes_price_4h": p4,
        "momentum": round(p4 - p24, 4) if p24 else 0.0,
        "open_interest": oi,
    }


def build_examples(series: str, max_markets: int = 400,
                   sleep_s: float = 0.15, log=print) -> list[dict]:
    markets = fetch_settled_markets(series, max_markets)
    log(f"  {series}: {len(markets)} settled markets")
    examples = []
    for m in markets:
        snap = _snapshot_price(series, m["ticker"], m.get("close_time", ""))
        time.sleep(sleep_s)  # stay well under Kalshi rate limits
        if snap is None:
            continue
        event = m.get("event_ticker") or m["ticker"].rsplit("-", 1)[0]
        examples.append({
            "ticker": m["ticker"],
            "event": event,
            "category": CATEGORY_LABELS.get(series, series.lower()),
            "title": m.get("title", ""),
            "outcome": m.get("yes_sub_title", ""),
            "close_time": m.get("close_time", ""),
            **snap,
            "total_volume": float(m.get("volume_fp") or 0),
            "result": 1 if m["result"] == "yes" else 0,
        })
    log(f"  {series}: {len(examples)} examples with pre-match prices")
    return examples


def split_by_event(examples: list[dict],
                   test_fraction: float = TEST_FRACTION) -> tuple[list, list]:
    """Deterministic split on a hash of the event ticker — same event never
    appears on both sides."""
    train, test = [], []
    for ex in examples:
        h = int(hashlib.sha1(ex["event"].encode()).hexdigest(), 16) % 100
        (test if h < test_fraction * 100 else train).append(ex)
    return train, test


RATINGS_PATH = DATA_DIR / "team_ratings.json"
_EVENT_TEAMS = None  # lazy: {"KALSHI_CODE": elo}
_TEAMS_RE = None


def _team_ratings() -> dict:
    global _EVENT_TEAMS, _TEAMS_RE
    if _EVENT_TEAMS is None:
        import re
        _EVENT_TEAMS = (json.loads(RATINGS_PATH.read_text())
                        if RATINGS_PATH.exists() else {})
        _TEAMS_RE = re.compile(r"-26[A-Z]{3}\d{2}([A-Z]{3})([A-Z]{3})$")
    return _EVENT_TEAMS


def _elo_features(ex: dict) -> dict:
    """Team-strength features oriented by the market's outcome.

    Winner-FRA in FRA-vs-ESP gets France's Elo edge; Winner-ESP the inverse;
    Tie/BTTS/total markets get closeness (|diff|) only. Scaled by 400 (one
    Elo 'class' ≈ 10x odds), so values live in roughly [-1, 1].
    """
    ratings = _team_ratings()
    m = _TEAMS_RE.search(ex.get("event", ""))
    if not m:
        return {"has_elo": 0, "elo_edge": 0.0, "elo_absdiff": 0.0}
    t1, t2 = m.group(1), m.group(2)
    e1, e2 = ratings.get(t1), ratings.get(t2)
    if e1 is None or e2 is None:
        return {"has_elo": 0, "elo_edge": 0.0, "elo_absdiff": 0.0}
    suffix = ex.get("ticker", "").rsplit("-", 1)[-1]
    edge = ((e1 - e2) if suffix == t1 else
            (e2 - e1) if suffix == t2 else 0)
    return {"has_elo": 1,
            "elo_edge": round(edge / 400.0, 4),
            "elo_absdiff": round(abs(e1 - e2) / 400.0, 4)}


def enrich(examples: list[dict]) -> list[dict]:
    """Add derived features from sibling markets (same event + category).

    A 3-way winner event has three markets whose prices should sum to ~1.
    Where they don't, something is mispriced — and being the favorite (or a
    distant longshot) inside your own event is signal the raw price alone
    doesn't carry.
    """
    groups: dict[tuple, list[dict]] = {}
    for ex in examples:
        groups.setdefault((ex["event"], ex["category"]), []).append(ex)

    for group in groups.values():
        prices = sorted((e["yes_price_4h"] for e in group), reverse=True)
        implied_sum = sum(prices)
        for e in group:
            p = e["yes_price_4h"]
            e["n_outcomes"] = len(group)
            e["implied_sum"] = round(implied_sum, 4)
            e["price_share"] = round(p / implied_sum, 4) if implied_sum else 0.0
            e["price_rank"] = prices.index(p) + 1
            e["is_favorite"] = 1 if e["price_rank"] == 1 else 0
            e["fav_gap"] = round(prices[0] - p, 4)
            if "has_elo" not in e:      # history rows carry precomputed Elo
                e.update(_elo_features(e))
    return examples


def load_history() -> list[dict]:
    """WC 2018/2022 pseudo-markets (fetch_history.py) — training only."""
    path = DATA_DIR / "dataset_history.json"
    return enrich(json.loads(path.read_text())) if path.exists() else []


def load_dataset() -> tuple[list[dict], list[dict]]:
    train = json.loads(TRAIN_PATH.read_text()) if TRAIN_PATH.exists() else []
    test = json.loads(TEST_PATH.read_text()) if TEST_PATH.exists() else []
    return enrich(train), enrich(test)


def build_and_save(max_per_series: int = 400, log=print) -> tuple[int, int]:
    all_examples = []
    for series in DATASET_SERIES:
        try:
            all_examples.extend(build_examples(series, max_per_series, log=log))
        except requests.RequestException as e:
            log(f"  {series}: fetch failed ({e}) — skipping series")
    train, test = split_by_event(all_examples)
    DATA_DIR.mkdir(exist_ok=True)
    TRAIN_PATH.write_text(json.dumps(train, indent=1))
    TEST_PATH.write_text(json.dumps(test, indent=1))
    log(f"\nDataset: {len(train)} train / {len(test)} test "
        f"({len({e['event'] for e in all_examples})} events)")
    return len(train), len(test)
