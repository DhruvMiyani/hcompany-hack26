"""GRPO policy prompt: short-id round-trip, decision parsing, clamping, scoring."""

import pytest

from agent.policy_prompt import (
    build_prompt, parse_index, parse_decision, score_completion, _extract_json,
)


# ── build_prompt ──────────────────────────────────────────────────────────────

def test_build_prompt_assigns_sequential_short_ids(markets):
    prompt, id_map = build_prompt(markets, ["rule one"], max_amount=5.0)
    assert list(id_map) == ["M1", "M2", "M3", "M4"]
    assert id_map["M2"].ticker == "KXWCGAME-26JUL14FRAESP-TIE"
    for mid in id_map:
        assert f"[{mid}]" in prompt


def test_build_prompt_includes_outcome_momentum_and_close(markets):
    from agent.decision import Market
    m = Market(name="X", ticker="KXWCGAME-26JUL14FRAESP-TIE",
               yes_price=0.30, no_price=0.70, volume=33_855,
               category="match_winner", outcome="Reg Time: Tie",
               momentum=0.04, closes="2026-07-14 19:00")
    prompt, id_map = build_prompt([m], [])
    line = prompt.splitlines()[1]
    assert '"Reg Time: Tie"' in line
    assert "Δ24h=+0.04" in line
    assert "closes 07-14 19:00" in line
    # richer line must still round-trip through the training-reward parser
    assert parse_index(prompt)["M1"]["ticker"] == m.ticker


def test_build_prompt_omits_missing_extras(markets):
    prompt, _ = build_prompt(markets[3:4], [])   # fixture market: no outcome/momentum
    line = prompt.splitlines()[1]
    assert "Δ24h" not in line and '"' not in line


def test_build_prompt_mentions_max_amount_and_schema(markets):
    prompt, _ = build_prompt(markets, [], max_amount=3.5)
    assert "Max bet: $3.50" in prompt
    assert '"market_id"' in prompt


def test_parse_index_round_trips_built_prompt(markets):
    prompt, id_map = build_prompt(markets, [])
    index = parse_index(prompt)
    assert set(index) == set(id_map)
    assert index["M1"]["ticker"] == "KXWCGAME-26JUL14FRAESP-FRA"
    assert index["M1"]["yes"] == pytest.approx(0.42)
    assert index["M1"]["volume"] == pytest.approx(254_571)
    assert index["M3"]["category"] == "both_teams_score"


# ── _extract_json ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("wrapper", [
    '{"skip": false, "market_id": "M1"}',
    '```json\n{"skip": false, "market_id": "M1"}\n```',
    '```\n{"skip": false, "market_id": "M1"}\n```',
    'Sure! Here is my pick: {"skip": false, "market_id": "M1"} hope that helps',
])
def test_extract_json_handles_fences_and_prose(wrapper):
    assert _extract_json(wrapper) == {"skip": False, "market_id": "M1"}


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("no json here at all") is None


# ── parse_decision ────────────────────────────────────────────────────────────

def _decide(payload, markets, max_amount=5.0):
    _, id_map = build_prompt(markets, [])
    return parse_decision(payload, id_map, max_amount)


def test_parse_decision_resolves_short_id_to_real_ticker(markets):
    d = _decide('{"skip": false, "market_id": "M2", "direction": "Yes",'
                ' "amount": 1.0, "confidence": 0.71}', markets)
    assert d.ticker == "KXWCGAME-26JUL14FRAESP-TIE"
    assert d.market == "[FRA vs ESP] France vs Spain Winner?"
    assert d.direction == "Yes"
    assert d.confidence == pytest.approx(0.71)


def test_parse_decision_is_case_insensitive_on_id(markets):
    d = _decide('{"market_id": "m1", "direction": "No", "amount": 2}', markets)
    assert d.ticker == "KXWCGAME-26JUL14FRAESP-FRA"


def test_parse_decision_clamps_absurd_amount_to_max(markets):
    d = _decide('{"market_id": "M1", "direction": "Yes", "amount": 14600000,'
                ' "confidence": 0.5}', markets, max_amount=5.0)
    assert d.amount == 5.0


def test_parse_decision_clamps_amount_to_at_least_one_dollar(markets):
    d = _decide('{"market_id": "M1", "direction": "Yes", "amount": 0.01}', markets)
    assert d.amount == 1.0


def test_parse_decision_normalises_percent_confidence(markets):
    d = _decide('{"market_id": "M1", "direction": "Yes", "amount": 2,'
                ' "confidence": 71}', markets)
    assert d.confidence == pytest.approx(0.71)


def test_parse_decision_non_numeric_amount_falls_back(markets):
    d = _decide('{"market_id": "M1", "direction": "Yes", "amount": "lots"}', markets)
    assert d.amount == 1.0


def test_parse_decision_unknown_id_keeps_no_ticker(markets):
    d = _decide('{"market_id": "M99", "direction": "Yes", "amount": 2}', markets)
    assert d.ticker is None


def test_parse_decision_unparseable_returns_none(markets):
    assert _decide("KXWC KXWC KXWC…", markets) is None


# ── score_completion (GRPO training reward) ───────────────────────────────────

def test_score_valid_liquid_pick_beats_thin_market(markets):
    prompt, _ = build_prompt(markets, [])
    liquid = score_completion(
        prompt, '{"skip": false, "market_id": "M1", "direction": "Yes",'
                ' "amount": 1, "confidence": 0.42}')
    thin = score_completion(
        prompt, '{"skip": false, "market_id": "M4", "direction": "Yes",'
                ' "amount": 1, "confidence": 0.06}')
    assert liquid > thin


def test_score_invalid_market_id_penalised(markets):
    prompt, _ = build_prompt(markets, [])
    assert score_completion(prompt, '{"market_id": "M99", "direction": "Yes"}') == -0.6


def test_score_skip_slightly_negative(markets):
    prompt, _ = build_prompt(markets, [])
    assert score_completion(prompt, '{"skip": true}') == -0.1


def test_score_unparseable_gets_graded_partial_credit(markets):
    prompt, _ = build_prompt(markets, [])
    noise = score_completion(prompt, "KXWC KXWC KXWC")
    on_track = score_completion(prompt, '{"market_id": "M1", "direction": "Yes"')
    assert noise == -1.0
    assert -1.0 < on_track < 0  # truncated-but-on-track beats noise
