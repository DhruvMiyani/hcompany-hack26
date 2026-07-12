"""Dataset construction — event-level split and snapshot hygiene."""

from agent.dataset import split_by_event


def _ex(event, ticker="T"):
    return {"event": event, "ticker": ticker, "yes_price_4h": 0.5, "result": 1}


def test_split_keeps_events_intact():
    examples = [_ex(f"EV{i % 20}", ticker=f"T{i}") for i in range(200)]
    train, test = split_by_event(examples)
    train_events = {e["event"] for e in train}
    test_events = {e["event"] for e in test}
    assert not (train_events & test_events)   # no leakage across the boundary
    assert len(train) + len(test) == 200


def test_split_is_deterministic():
    examples = [_ex(f"EV{i}") for i in range(100)]
    a = split_by_event(examples)
    b = split_by_event(examples)
    assert a == b


def test_split_fraction_roughly_respected():
    examples = [_ex(f"EV{i}") for i in range(1000)]
    _, test = split_by_event(examples, test_fraction=0.25)
    assert 0.15 < len(test) / 1000 < 0.35
