"""Simple model — feature enrichment, logistic regression, edge strategy."""

import numpy as np
import pytest

from agent.dataset import enrich
from agent.simple_model import SimpleModel, featurize, edge_strategy, CATEGORIES


def _ex(price, result, event="EV1", category="match_winner", **over):
    d = {"event": event, "category": category, "ticker": "T",
         "yes_price_4h": price, "momentum": 0.0,
         "open_interest": 50_000, "result": result}
    d.update(over)
    return d


def test_enrich_computes_sibling_structure():
    exs = enrich([_ex(0.42, 0), _ex(0.30, 1), _ex(0.27, 0)])
    fav = next(e for e in exs if e["yes_price_4h"] == 0.42)
    dog = next(e for e in exs if e["yes_price_4h"] == 0.27)
    assert fav["is_favorite"] == 1 and fav["price_rank"] == 1
    assert dog["is_favorite"] == 0 and dog["price_rank"] == 3
    assert fav["implied_sum"] == pytest.approx(0.99)
    assert dog["fav_gap"] == pytest.approx(0.15)


def test_enrich_groups_by_event_and_category():
    exs = enrich([_ex(0.6, 1, event="A"), _ex(0.6, 1, event="B")])
    assert all(e["n_outcomes"] == 1 for e in exs)   # different events → no siblings


def test_featurize_length_is_stable():
    ex = enrich([_ex(0.5, 1)])[0]
    assert len(featurize(ex)) == 11 + len(CATEGORIES)


def test_elo_features_oriented_by_ticker_suffix(monkeypatch, tmp_path):
    import json
    from agent import dataset
    ratings = tmp_path / "team_ratings.json"
    ratings.write_text(json.dumps({"FRA": 2163, "ESP": 2190}))
    monkeypatch.setattr(dataset, "RATINGS_PATH", ratings)
    monkeypatch.setattr(dataset, "_EVENT_TEAMS", None)  # bust the cache

    def mk(suffix):
        return _ex(0.4, 1, event="KXWCGAME-26JUL14FRAESP",
                   ticker=f"KXWCGAME-26JUL14FRAESP-{suffix}")
    fra, esp, tie = enrich([mk("FRA"), mk("ESP"), mk("TIE")])
    assert fra["elo_edge"] == pytest.approx((2163 - 2190) / 400)
    assert esp["elo_edge"] == pytest.approx((2190 - 2163) / 400)
    assert tie["elo_edge"] == 0.0                      # tie: no team side
    assert tie["elo_absdiff"] == pytest.approx(27 / 400)
    assert all(e["has_elo"] == 1 for e in (fra, esp, tie))
    monkeypatch.setattr(dataset, "_EVENT_TEAMS", None)  # don't leak to other tests


def test_elo_features_absent_without_ratings():
    ex = enrich([_ex(0.5, 1, event="NOPATTERN")])[0]
    assert ex["has_elo"] == 0 and ex["elo_edge"] == 0.0


def test_logistic_regression_learns_separable_data():
    # price perfectly predicts outcome → model must learn it
    train = enrich([_ex(0.8, 1, event=f"E{i}") for i in range(40)]
                   + [_ex(0.2, 0, event=f"F{i}") for i in range(40)])
    model = SimpleModel().fit(train, epochs=300)
    test = enrich([_ex(0.85, 1, event="X"), _ex(0.15, 0, event="Y")])
    assert model.predict(test) == [1, 0]
    probs = model.predict_proba(test)
    assert probs[0] > 0.7 and probs[1] < 0.3


def test_model_save_load_roundtrip(tmp_path):
    train = enrich([_ex(0.8, 1, event=f"E{i}") for i in range(20)]
                   + [_ex(0.2, 0, event=f"F{i}") for i in range(20)])
    model = SimpleModel().fit(train, epochs=100)
    path = tmp_path / "m.json"
    model.save(path)
    loaded = SimpleModel.load(path)
    test = enrich([_ex(0.9, 1, event="Z")])
    assert loaded.predict(test) == model.predict(test)


def test_edge_strategy_only_bets_on_disagreement():
    exs = enrich([_ex(0.50, 1, event="A"), _ex(0.50, 0, event="B")])
    # model agrees with price on B (skip), disagrees on A (bet YES)
    s = edge_strategy(exs, np.array([0.70, 0.52]), margin=0.05)
    assert s["n_bets"] == 1 and s["skipped"] == 1
    assert s["pnl"] == pytest.approx(1.0)   # won $1 at 50c


def test_edge_strategy_no_bets_returns_none_roi():
    exs = enrich([_ex(0.5, 1)])
    s = edge_strategy(exs, np.array([0.5]), margin=0.05)
    assert s["n_bets"] == 0 and s["roi"] is None
