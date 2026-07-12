"""Classification + trading metrics used by the offline evaluator."""

import pytest

from agent.metrics import (
    classification_metrics, confusion, implied_baseline,
    trading_metrics, evaluate_predictor,
)


def _ex(price, result):
    return {"yes_price_4h": price, "result": result}


def test_confusion_counts():
    c = confusion([1, 1, 0, 0], [1, 0, 1, 0])
    assert c == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}


def test_perfect_predictor():
    m = classification_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_always_yes_has_full_recall_low_precision():
    m = classification_metrics([1, 0, 0, 0], [1, 1, 1, 1])
    assert m["recall"] == 1.0
    assert m["precision"] == 0.25
    assert m["accuracy"] == 0.25
    # F1 is the harmonic mean
    assert m["f1"] == pytest.approx(2 * 0.25 * 1.0 / 1.25)


def test_empty_or_mismatched_inputs_are_safe():
    assert classification_metrics([], [])["accuracy"] is None
    assert classification_metrics([1], [1, 0])["n"] == 0


def test_implied_baseline_thresholds_at_50c():
    exs = [_ex(0.42, 0), _ex(0.58, 1), _ex(0.50, 0)]
    assert implied_baseline(exs) == [0, 1, 0]


def test_trading_metrics_yes_win_pays_odds():
    # bet Yes at 25c and win → profit 3x stake
    t = trading_metrics([_ex(0.25, 1)], [1])
    assert t["pnl"] == pytest.approx(3.0)
    assert t["roi"] == pytest.approx(3.0)


def test_trading_metrics_no_win_pays_inverse_odds():
    # bet No at 80c (No costs 20c) and win → profit 4x stake
    t = trading_metrics([_ex(0.80, 0)], [0])
    assert t["pnl"] == pytest.approx(4.0)


def test_trading_metrics_loss_costs_stake():
    t = trading_metrics([_ex(0.60, 0)], [1])
    assert t["pnl"] == -1.0
    assert t["hit_rate"] == 0.0


def test_betting_the_market_price_is_ev_neutral_when_prices_are_calibrated():
    # 10 markets at 70c, exactly 7 settle YES → implied strategy nets ~0
    exs = [_ex(0.70, 1 if i < 7 else 0) for i in range(10)]
    t = trading_metrics(exs, implied_baseline(exs))
    assert t["pnl"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_predictor_combines_both_views():
    exs = [_ex(0.3, 1), _ex(0.7, 1), _ex(0.4, 0)]
    out = evaluate_predictor(exs, [1, 1, 0])
    assert out["accuracy"] == 1.0
    assert out["trading"]["n_bets"] == 3
