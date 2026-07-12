"""Shared fixtures — all tests run offline (no network, no model loads)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.decision import Market  # noqa: E402


@pytest.fixture
def markets() -> list[Market]:
    """A realistic snapshot of France vs Spain markets (from a live run)."""
    return [
        Market(
            name="[FRA vs ESP] France vs Spain Winner?",
            ticker="KXWCGAME-26JUL14FRAESP-FRA",
            yes_price=0.42, no_price=0.58, volume=254_571,
            category="match_winner", match="FRA vs ESP",
            outcome="Reg Time: France",
        ),
        Market(
            name="[FRA vs ESP] France vs Spain Winner?",
            ticker="KXWCGAME-26JUL14FRAESP-TIE",
            yes_price=0.30, no_price=0.70, volume=33_855,
            category="match_winner", match="FRA vs ESP",
            outcome="Reg Time: Tie",
        ),
        Market(
            name="[FRA vs ESP] Both teams to score?",
            ticker="KXWCBTTS-26JUL14FRAESP-BTTS",
            yes_price=0.60, no_price=0.40, volume=93_597,
            category="both_teams_score", match="FRA vs ESP",
        ),
        Market(
            name="[FRA vs ESP] Over 5.5 goals?",
            ticker="KXWCTOTAL-26JUL14FRAESP-6",
            yes_price=0.06, no_price=0.94, volume=6_319,
            category="total_goals", match="FRA vs ESP",
        ),
    ]


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Point agent.memory at a throwaway SQLite DB."""
    from agent import memory
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "test_memory.db")
    memory.init_db()
    return memory
