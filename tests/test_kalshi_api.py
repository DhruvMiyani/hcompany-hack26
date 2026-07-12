"""Kalshi REST parsing — mocked HTTP, no network."""

from unittest.mock import patch, MagicMock

from agent import kalshi_api
from agent.kalshi_api import _category_prefix, _price, _fetch_event_markets


def test_price_conversion():
    assert _price("0.42") == 0.42
    assert _price(0.3) == 0.3
    assert _price(None) is None
    assert _price("garbage") is None


def test_category_prefix_longest_known_wins():
    assert _category_prefix("KXWCGAME-26JUL14FRAESP") == "KXWCGAME"
    assert _category_prefix("KXWCADVANCE-26JUL14FRAESP") == "KXWCADVANCE"
    assert _category_prefix("KXWCTOTAL-26JUL14FRAESP") == "KXWCTOTAL"


def _api_market(**over):
    m = {
        "ticker": "KXWCGAME-26JUL14FRAESP-TIE",
        "title": "France vs Spain Winner?",
        "yes_sub_title": "Reg Time: Tie",
        "yes_ask_dollars": "0.30",
        "no_ask_dollars": "0.70",
        "last_price_dollars": "0.29",
        "volume_fp": "33855",
        "close_time": "2026-07-14T19:00:00Z",
    }
    m.update(over)
    return m


def _mock_get(markets):
    resp = MagicMock()
    resp.json.return_value = {"markets": markets}
    resp.raise_for_status.return_value = None
    return patch.object(kalshi_api.requests, "get", return_value=resp)


def test_fetch_event_markets_builds_full_market():
    with _mock_get([_api_market()]):
        out = _fetch_event_markets("KXWCGAME-26JUL14FRAESP")
    assert len(out) == 1
    m = out[0]
    assert m.ticker == "KXWCGAME-26JUL14FRAESP-TIE"
    assert m.name == "[FRA vs ESP] France vs Spain Winner?"  # MATCH_MAP prefix
    assert m.outcome == "Reg Time: Tie"                      # drives browser outcome row
    assert m.yes_price == 0.30
    assert m.no_price == 0.70
    assert m.category == "match_winner"
    assert m.match == "FRA vs ESP"
    assert m.volume == 33855.0


def test_fetch_event_markets_skips_unpriced_markets():
    dead = _api_market(yes_ask_dollars=None, yes_bid_dollars=None)
    with _mock_get([dead, _api_market()]):
        out = _fetch_event_markets("KXWCGAME-26JUL14FRAESP")
    assert len(out) == 1


def test_fetch_event_markets_derives_no_price_when_missing():
    with _mock_get([_api_market(no_ask_dollars=None)]):
        out = _fetch_event_markets("KXWCGAME-26JUL14FRAESP")
    assert out[0].no_price == 0.70  # 1 - yes


def test_fetch_event_markets_survives_network_error():
    with patch.object(kalshi_api.requests, "get", side_effect=OSError("boom")):
        assert _fetch_event_markets("KXWCGAME-26JUL14FRAESP") == []
