"""H Company browser agent runner — API scrape + browser execute."""

import os
import re
import sys
from typing import Optional

from hai_agents import Client, wait_for_session

from .kalshi_api import get_open_wc_markets
from .tasks import execute_bet_task, check_results_task
from .decision import decide_bet, BetDecision

AGENT_VIEW_HOST = "https://platform.eu.hcompany.ai"

# Use flash for speed — execution is a well-defined single task, not research
EXECUTE_AGENT = "h/web-surfer-flash"
CHECK_AGENT   = "h/web-surfer-flash"


def _client() -> Client:
    return Client(api_key=os.environ["HAI_API_KEY"])


def _kalshi_creds() -> tuple[str, str, str]:
    url      = os.getenv("KALSHI_URL", "https://demo.kalshi.co")
    email    = os.environ["KALSHI_EMAIL"]
    password = os.environ["KALSHI_PASSWORD"]
    return url, email, password


def _browser_session(client: Client, task: str, agent: str, label: str) -> tuple:
    session = client.sessions.create_session(agent=agent, messages=task)
    live_url = f"{AGENT_VIEW_HOST}/agent-view/{session.id}"
    print(f"  [{label}] session={session.id}", file=sys.stderr, flush=True)
    print(f"  [{label}] watch → {live_url}", file=sys.stderr, flush=True)
    result = wait_for_session(client, session.id, timeout_seconds=600)
    print(f"  [{label}] done — status={result.status}", file=sys.stderr, flush=True)
    return result, session.id


def run_full_bet_cycle(
    strategy_rules: list[str],
    lessons: list[dict],
    max_amount: float,
) -> dict:
    """
    Phase 1 — Kalshi REST API → market list (instant, no browser)
    Phase 2 — Holo model → BetDecision
    Phase 3 — h/web-surfer-flash → places the bet
    """
    # ── Phase 1: Kalshi API ──────────────────────────────────────────────────
    print("\n[Phase 1] Fetching open FIFA markets via Kalshi API...", file=sys.stderr)
    markets = get_open_wc_markets()
    print(f"  Found {len(markets)} soccer market(s).", file=sys.stderr)
    for m in markets[:5]:
        print(f"    {m.name[:60]} | Yes={m.yes_price} No={m.no_price}", file=sys.stderr)

    # ── Phase 2: Holo model decides ──────────────────────────────────────────
    print("\n[Phase 2] Holo model deciding which bet to place...", file=sys.stderr)
    decision: BetDecision = decide_bet(
        markets=markets,
        strategy_rules=strategy_rules,
        lessons=lessons,
        max_amount=max_amount,
        min_confidence=0.3,
    )
    print(f"  skip={decision.skip}", file=sys.stderr)
    if not decision.skip:
        print(
            f"  → {decision.direction} | '{decision.market[:60]}'\n"
            f"    ${decision.amount:.2f} | confidence={decision.confidence:.0%}\n"
            f"    {decision.reasoning}",
            file=sys.stderr,
        )

    if decision.skip:
        return {
            "skipped": True,
            "skip_reason": decision.skip_reason,
            "decision": decision.model_dump(),
            "execution": None,
            "markets_found": len(markets),
        }

    # ── Phase 3: Browser executes ────────────────────────────────────────────
    print("\n[Phase 3] h/web-surfer-flash placing the bet...", file=sys.stderr)
    client = _client()
    url, email, password = _kalshi_creds()
    task = execute_bet_task(url, email, password, decision.market, decision.direction, decision.amount, decision.ticker or "")
    exec_result, exec_sid = _browser_session(client, task, EXECUTE_AGENT, "execute")
    exec_answer = exec_result.answer or ""

    return {
        "skipped": False,
        "decision": decision.model_dump(),
        "execution": {
            "answer": exec_answer,
            "status": exec_result.status,
            "filled_price": _parse_field(exec_answer, "FILLED_PRICE", float),
            "order_id": _parse_field(exec_answer, "ORDER_ID", str),
            "success": "placed successfully" in exec_answer.lower(),
            "session_id": exec_sid,
        },
        "markets_found": len(markets),
    }


def run_check_results() -> dict:
    client = _client()
    url, email, password = _kalshi_creds()
    task = check_results_task(url, email, password)
    result, sid = _browser_session(client, task, CHECK_AGENT, "check")
    answer = result.answer or ""
    return {
        "session_id": sid,
        "status": result.status,
        "answer": answer,
        "bets": _parse_check_results(answer),
    }


def _parse_field(text: str, label: str, cast):
    m = re.search(rf"{label}:\s*(\S+)", text, re.IGNORECASE)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (ValueError, TypeError):
        return m.group(1)


def _parse_check_results(text: str) -> list[dict]:
    bets = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("BET:"):
            continue
        parts = [p.strip() for p in line[4:].split("|")]
        if len(parts) < 4:
            continue
        amount = None
        m = re.search(r"[\d.]+", parts[2] if len(parts) > 2 else "")
        if m:
            amount = float(m.group())
        status_part = parts[3] if len(parts) > 3 else ""
        pl_part = parts[4] if len(parts) > 4 else ""
        status = "pending"
        if "won" in status_part.lower():
            status = "won"
        elif "lost" in status_part.lower():
            status = "lost"
        pl = None
        m = re.search(r"-?[\d.]+", pl_part)
        if m and "pending" not in pl_part.lower():
            pl = float(m.group())
        bets.append({
            "market": parts[0],
            "direction": parts[1],
            "amount": amount,
            "status": status,
            "profit_loss": pl,
        })
    return bets
