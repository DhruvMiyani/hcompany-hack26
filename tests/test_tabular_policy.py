"""Live tabular policy — edge picking, direction, stake sizing."""

import numpy as np
import pytest

from agent.decision import Market
from agent.tabular_policy import TabularPolicy, market_to_example


class _StubModel:
    def __init__(self, probs):
        self._probs = probs

    def predict_proba(self, examples):
        return np.array(self._probs)


def _policy(probs):
    p = object.__new__(TabularPolicy)
    p.name = "tabular:stub"
    p.model = _StubModel(probs)
    return p


def _mk(ticker, price, **over):
    d = dict(name=f"mkt {ticker}", ticker=ticker, yes_price=price,
             no_price=round(1 - price, 2), volume=50_000,
             category="match_winner", momentum=0.0)
    d.update(over)
    return Market(**d)


MARKETS = [
    _mk("KXWCGAME-26JUL14FRAESP-FRA", 0.42),
    _mk("KXWCGAME-26JUL14FRAESP-TIE", 0.30),
    _mk("KXWCGAME-26JUL14FRAESP-ESP", 0.30),
]


def test_picks_largest_edge_side_yes():
    # model: TIE is worth 0.45 vs price 0.30 → +15c edge, biggest
    d = _policy([0.44, 0.45, 0.28]).decide(MARKETS, max_amount=5.0)
    assert d.ticker == "KXWCGAME-26JUL14FRAESP-TIE"
    assert d.direction == "Yes"
    assert d.confidence == pytest.approx(0.45)


def test_bets_no_when_yes_overpriced():
    # model: FRA only 0.25 likely vs price 0.42 → NO edge +0.17
    d = _policy([0.25, 0.31, 0.29]).decide(MARKETS, max_amount=5.0)
    assert d.ticker == "KXWCGAME-26JUL14FRAESP-FRA"
    assert d.direction == "No"


def test_returns_none_when_no_edge_clears_margin():
    assert _policy([0.43, 0.31, 0.29]).decide(MARKETS) is None


def test_stake_is_kelly_sized_and_clamped():
    d = _policy([0.44, 0.60, 0.28]).decide(MARKETS, max_amount=5.0)
    kelly = (0.60 - 0.30) / 0.70
    assert d.amount == pytest.approx(round(min(5 * kelly, 5.0), 2))
    assert 1.0 <= d.amount <= 5.0


def test_market_to_example_derives_event_from_ticker():
    ex = market_to_example(MARKETS[0])
    assert ex["event"] == "KXWCGAME-26JUL14FRAESP"
    assert ex["yes_price_4h"] == 0.42


def test_real_champion_trains_and_decides():
    """Integration: the actual champion trains from the committed dataset."""
    pytest.importorskip("xgboost")
    policy = TabularPolicy()
    d = policy.decide(MARKETS, max_amount=5.0)
    assert d is None or (d.ticker in {m.ticker for m in MARKETS}
                         and 1.0 <= d.amount <= 5.0)