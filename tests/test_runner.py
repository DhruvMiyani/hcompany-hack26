"""Ensemble decision logic + browser-answer parsing in agent/runner.py."""

from unittest.mock import patch, MagicMock

from agent import runner
from agent.decision import BetDecision


def _grpo_pick(ticker="KXWCGAME-26JUL14FRAESP-TIE", direction="Yes", conf=0.7):
    return BetDecision(skip=False, ticker=ticker, direction=direction,
                       market="France vs Spain Winner?", amount=1.0, confidence=conf)


def _holo_pick(ticker="KXWCGAME-26JUL14FRAESP-FRA", direction="Yes", conf=0.35):
    return BetDecision(skip=False, ticker=ticker, direction=direction,
                       market="France vs Spain Winner?", amount=5.0, confidence=conf)


def _with_trained(trained: bool):
    stub = MagicMock()
    stub.is_trained = trained
    return patch.object(runner, "get_grpo", return_value=stub)


def test_trained_grpo_with_valid_ticker_wins(markets):
    with _with_trained(True):
        out = runner._pick_decision(_grpo_pick(), _holo_pick(), markets)
    assert out["source"] == "grpo"
    assert out["bet"].ticker == "KXWCGAME-26JUL14FRAESP-TIE"


def test_agreement_boosts_confidence(markets):
    grpo = _grpo_pick(conf=0.6)
    holo = _holo_pick(ticker=grpo.ticker, direction="Yes", conf=0.5)
    with _with_trained(True):
        out = runner._pick_decision(grpo, holo, markets)
    assert out["source"] == "grpo+holo_agree"
    assert out["bet"].confidence > 0.6


def test_hallucinated_ticker_falls_back_to_holo(markets):
    with _with_trained(True):
        out = runner._pick_decision(
            _grpo_pick(ticker="KXWCGAME-NOTREAL-XXX"), _holo_pick(), markets)
    assert out["source"] == "holo"


def test_untrained_grpo_falls_back_to_holo(markets):
    with _with_trained(False):
        out = runner._pick_decision(_grpo_pick(), _holo_pick(), markets)
    assert out["source"] == "holo"


def test_grpo_none_or_skip_falls_back_to_holo(markets):
    with _with_trained(True):
        assert runner._pick_decision(None, _holo_pick(), markets)["source"] == "holo"
        skip = BetDecision(skip=True, skip_reason="nothing good")
        assert runner._pick_decision(skip, _holo_pick(), markets)["source"] == "holo"


# ── browser answer parsing ────────────────────────────────────────────────────

def test_parse_field_extracts_typed_values():
    answer = "STATUS: Bet placed successfully\nFILLED_PRICE: 0.31\nORDER_ID: ab12"
    assert runner._parse_field(answer, "FILLED_PRICE", float) == 0.31
    assert runner._parse_field(answer, "ORDER_ID", str) == "ab12"
    assert runner._parse_field(answer, "MISSING", float) is None


def test_parse_check_results_reads_bet_lines():
    answer = """Here is the portfolio:
BET: France vs Spain Winner? | Yes | $5.00 wagered | STATUS: Open | P&L: Pending
BET: Argentina to advance | Yes | $3.00 wagered | STATUS: Won | P&L: $1.80
BET: Over 2.5 goals | No | $2.00 wagered | STATUS: Lost | P&L: -2.00
SUMMARY: 3 bets, 1 won, 1 lost, -0.20 net
"""
    bets = runner._parse_check_results(answer)
    assert [b["status"] for b in bets] == ["pending", "won", "lost"]
    assert bets[0]["profit_loss"] is None            # pending → no P&L yet
    assert bets[1]["profit_loss"] == 1.80
    assert bets[2]["profit_loss"] == -2.00
    assert bets[0]["amount"] == 5.00


def test_parse_check_results_ignores_malformed_lines():
    assert runner._parse_check_results("BET: too | short\nnot a bet line") == []
