"""Reward shaping for settled bets — the GRPO learning signal."""

import pytest

from agent.reward import compute_reward, compute_skip_reward


def _bet(**over):
    d = {"direction": "Yes", "amount": 2.0, "confidence": 0.7, "volume": 0}
    d.update(over)
    return d


def test_pending_and_void_give_zero_reward():
    assert compute_reward(_bet(), {"status": "pending"}) == 0.0
    assert compute_reward(_bet(), {"status": "void", "profit_loss": 3}) == 0.0


def test_confident_win_beats_timid_win():
    bold = compute_reward(_bet(confidence=0.8), {"status": "won", "profit_loss": 2.0})
    timid = compute_reward(_bet(confidence=0.3), {"status": "won", "profit_loss": 2.0})
    assert bold > timid


def test_overconfident_loss_hit_hardest():
    cocky = compute_reward(_bet(confidence=0.9), {"status": "lost", "profit_loss": -2.0})
    cautious = compute_reward(_bet(confidence=0.3), {"status": "lost", "profit_loss": -2.0})
    assert cocky < cautious
    # -0.40 calibration penalty applies on top of the P&L base
    assert cocky <= -1.0 - 0.40 + 0.15  # base(-1) + penalty, sizing at most +0.10


def test_base_reward_is_pnl_per_dollar():
    r = compute_reward(
        _bet(confidence=0.5, amount=4.0), {"status": "won", "profit_loss": 2.0})
    # base 0.5, no calibration bonus at conf 0.5, sizing off-Kelly → -0.05
    assert r == pytest.approx(0.5 + 0.0 - 0.05)


def test_kelly_aligned_sizing_gets_bonus():
    # conf 0.8 → kelly f*=0.6 → ideal stake $3 of max $5
    good = compute_reward(_bet(confidence=0.8, amount=3.0),
                          {"status": "won", "profit_loss": 1.0})
    bad = compute_reward(_bet(confidence=0.8, amount=1.0),
                         {"status": "won", "profit_loss": 1.0 / 3})
    assert good == pytest.approx(1 / 3.0 + 0.25 + 0.10, abs=1e-3)
    assert bad == pytest.approx(1 / 3.0 + 0.25 - 0.05, abs=1e-3)


def test_deep_market_liquidity_bonus():
    deep = compute_reward(_bet(volume=2_000_000), {"status": "won", "profit_loss": 1.0})
    thin = compute_reward(_bet(volume=500), {"status": "won", "profit_loss": 1.0})
    assert deep - thin == pytest.approx(0.05)


def test_confidence_is_clamped_not_crashed():
    r = compute_reward(_bet(confidence=250), {"status": "won", "profit_loss": 1.0})
    assert isinstance(r, float)


def test_skip_reward_direction():
    assert compute_skip_reward(10, recent_win_rate=0.8) < 0   # should have bet
    assert compute_skip_reward(10, recent_win_rate=0.2) > 0   # smart skip
    assert compute_skip_reward(10, recent_win_rate=0.5) == 0.0
