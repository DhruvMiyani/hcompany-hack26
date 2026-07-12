"""server.py backend logic — bet-watch phase machine + markets snapshot."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
server = importlib.import_module("server")


@pytest.fixture
def bet_log(monkeypatch, tmp_path):
    log = tmp_path / "last_bet.log"
    monkeypatch.setattr(server, "BET_LOG", log)
    return log


# ── bet_watch phase machine ───────────────────────────────────────────────────

def test_no_log_means_idle(bet_log):
    assert server.bet_watch()["phase"] == "idle"


def test_model_phase_reports_deciding(bet_log):
    bet_log.write_text("[Phase 1] fetching...\n[Phase 2A] GRPO policy model...\n")
    assert server.bet_watch()["phase"] == "deciding"


def test_decision_line_is_parsed(bet_log):
    bet_log.write_text(
        "[GRPO] → Yes | KXWCGAME-26JUL14FRAESP-TIE | $1.00 | conf=71%\n")
    d = server.bet_watch()["decision"]
    assert d == {"direction": "Yes", "ticker": "KXWCGAME-26JUL14FRAESP-TIE",
                 "amount": 1.0, "confidence": 71}


def test_watch_url_switches_phase_to_executing(bet_log):
    sid = "d345a9ac-c00c-4c0e-87a5-eebd23a1b82a"
    bet_log.write_text(f"[Phase 3] placing...\n  [execute] watch → "
                       f"https://platform.eu.hcompany.ai/agent-view/{sid}\n")
    w = server.bet_watch()
    assert w["phase"] == "executing"
    assert w["watch_url"].endswith(sid)


@pytest.mark.parametrize("marker,result", [
    ("Bet placed successfully!", "placed"),
    ("Insufficient funds", "insufficient_funds"),
    ("Execution failed.", "failed"),
    ("Skipped — no suitable market found.", "skipped"),
    ("Traceback (most recent call last):\n  ModuleNotFoundError", "failed"),
])
def test_terminal_states(bet_log, marker, result):
    bet_log.write_text(f"[Phase 1] ...\n{marker}\n")
    w = server.bet_watch()
    assert w["phase"] == "done"
    assert w["result"] == result


# ── markets snapshot parsing ──────────────────────────────────────────────────

SAMPLE_LOG = """[Phase 1] Kalshi REST API — fetching pure WC markets...
  57 WC markets total, 15 sent to model.
    [match_winner] KXWCGAME-26JUL14FRAESP-FRA | Yes=0.42 | Vol=$254,571
    [match_winner] KXWCGAME-26JUL14FRAESP-TIE | Yes=0.3 | Vol=$33,855
    [total_goals] KXWCTOTAL-26JUL14FRAESP-6 | Yes=0.06 | Vol=$6,319
"""


def test_markets_snapshot_parses_counts_and_rows(bet_log):
    bet_log.write_text(SAMPLE_LOG)
    snap = server.markets_snapshot()
    assert snap["total"] == 57
    assert snap["sent"] == 15
    assert len(snap["markets"]) == 3
    assert snap["markets"][0] == {
        "type": "match_winner", "ticker": "KXWCGAME-26JUL14FRAESP-FRA",
        "yes": 0.42, "volume": 254571,
    }


def test_markets_snapshot_empty_without_log(bet_log):
    assert server.markets_snapshot() == {"total": 0, "sent": 0, "markets": []}


# ── stats over a real (temp) DB ───────────────────────────────────────────────

def test_stats_reflects_stored_bets(tmp_db):
    tmp_db.store_bet(session_id="s1", market="France vs Spain Winner?",
                     direction="Yes", amount=5.0, odds=0.3, raw_answer="")
    bet2 = tmp_db.store_bet(session_id="s2", market="Both teams to score?",
                            direction="No", amount=2.0, odds=0.4, raw_answer="")
    tmp_db.update_bet_outcome(bet2, "won", 3.0)

    s = server.stats()
    assert s["total_bets"] == 2
    assert s["wins"] == 1
    assert s["win_rate"] == 1.0          # 1 settled, 1 won
    assert s["wagered"] == 7.0
    assert s["pnl"] == 3.0
    statuses = {b["market"]: b["status"] for b in s["bets"]}
    assert statuses["France vs Spain Winner?"] == "pending"
    assert statuses["Both teams to score?"] == "won"
