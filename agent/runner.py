"""H Company browser agent runner — two-model pipeline + browser execution."""

import os
import re
import sys
from typing import Optional

from hai_agents import Client, wait_for_session

from .kalshi_api import get_open_wc_markets
from .tasks import execute_bet_task, check_results_task
from .decision import decide_bet, BetDecision
from .grpo_model import get_model as get_grpo

AGENT_VIEW_HOST = "https://platform.eu.hcompany.ai"

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
    session  = client.sessions.create_session(agent=agent, messages=task)
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
    Phase 1   Kalshi REST API       → 102 pure WC markets (no browser)
    Phase 2A  GRPO policy model     → our fine-tuned Qwen2.5-1.5B
    Phase 2B  Holo validator        → sanity check + fallback
    Phase 3   H Company browser     → h/web-surfer-flash places the bet
    """

    # ── Phase 1: Kalshi REST API ─────────────────────────────────────────────
    print("\n[Phase 1] Kalshi REST API — fetching pure WC markets...", file=sys.stderr)
    all_markets = get_open_wc_markets()

    priority = [m for m in all_markets if m.category in (
        "match_winner", "advance", "both_teams_score", "total_goals"
    )]
    others = sorted(
        [m for m in all_markets if m not in priority and (m.volume or 0) > 50_000],
        key=lambda m: -(m.volume or 0),
    )[:6]
    markets = priority + others

    print(f"  {len(all_markets)} WC markets total, {len(markets)} sent to model.", file=sys.stderr)
    for m in markets[:6]:
        vol_str = f"Vol=${m.volume:,.0f}" if m.volume else "Vol=?"
        print(f"    [{m.category}] {m.ticker} | Yes={m.yes_price} | {vol_str}", file=sys.stderr)

    # ── Phase 2·0: tabular champion (XGBoost edge picker) ───────────────────
    # The research-loop champion scores every market in milliseconds. If any
    # side has edge > margin it decides directly and we skip the LLM phases.
    tab_decision = None
    print("\n[Phase 2] Tabular champion — scoring all markets for edge...",
          file=sys.stderr)
    try:
        from .tabular_policy import TabularPolicy
        tab = TabularPolicy()
        tab_decision = tab.decide(markets, max_amount)
        if tab_decision:
            print(
                f"  [XGB] → {tab_decision.direction} | {tab_decision.ticker} "
                f"| ${tab_decision.amount:.2f} "
                f"| conf={tab_decision.confidence:.0%}",
                file=sys.stderr,
            )
            print(f"  [XGB] {tab_decision.reasoning}", file=sys.stderr)
        else:
            print(f"  [{tab.name}] no market clears the edge margin — "
                  "falling back to GRPO/Holo", file=sys.stderr)
    except Exception as e:
        print(f"  [XGB] unavailable ({e}) — falling back to GRPO/Holo",
              file=sys.stderr)

    # ── Phase 2A: GRPO policy model ──────────────────────────────────────────
    grpo        = get_grpo()
    grpo_decision: Optional[BetDecision] = None

    if tab_decision is not None:
        # XGBoost found an edge — decision made in milliseconds, skip the
        # slow LLM phases entirely.
        decision = {"bet": tab_decision, "source": "xgb_edge"}
        return _execute_decision(decision, all_markets, markets)

    print("\n[Phase 2A] GRPO policy model (Qwen2.5-1.5B)...", file=sys.stderr)
    if grpo.is_available():
        trained_label = "fine-tuned" if grpo.is_trained else "base (cold start)"
        print(f"  [GRPO] Status: {trained_label}", file=sys.stderr)
        grpo_decision = grpo.predict(markets, strategy_rules, lessons, max_amount)
        if grpo_decision and not grpo_decision.skip:
            print(
                f"  [GRPO] → {grpo_decision.direction} | {grpo_decision.ticker} "
                f"| ${grpo_decision.amount:.2f} | conf={grpo_decision.confidence:.0%}",
                file=sys.stderr,
            )
        elif grpo_decision and grpo_decision.skip:
            print(f"  [GRPO] skip — {grpo_decision.skip_reason}", file=sys.stderr)
        else:
            print("  [GRPO] No parseable output", file=sys.stderr)
    else:
        print("  [GRPO] Not available (install transformers+trl+peft)", file=sys.stderr)

    # ── Phase 2B: Holo validator / fallback ──────────────────────────────────
    print("\n[Phase 2B] Holo model (holo3-1-35b-a3b) — validate / fallback...", file=sys.stderr)
    holo_decision: BetDecision = decide_bet(
        markets=markets,
        strategy_rules=strategy_rules,
        lessons=lessons,
        max_amount=max_amount,
        min_confidence=0.3,
    )
    if not holo_decision.skip:
        print(
            f"  [Holo] → {holo_decision.direction} | {holo_decision.ticker} "
            f"| ${holo_decision.amount:.2f} | conf={holo_decision.confidence:.0%}",
            file=sys.stderr,
        )

    # ── Ensemble: pick final decision ─────────────────────────────────────────
    decision = _pick_decision(grpo_decision, holo_decision, markets)
    return _execute_decision(decision, all_markets, markets)


def _execute_decision(decision: dict, all_markets: list, markets: list) -> dict:
    """Phase 3: hand the final decision to the browser agent."""
    print(f"\n[Decision] source={decision.get('source')} skip={decision['bet'].skip}", file=sys.stderr)

    final: BetDecision = decision["bet"]
    source: str        = decision["source"]

    if final.skip:
        return {
            "skipped": True,
            "skip_reason": final.skip_reason,
            "decision": final.model_dump(),
            "decision_source": source,
            "execution": None,
            "markets_found": len(all_markets),
        }

    print(
        f"  → {final.direction} on {final.ticker}\n"
        f"    {final.market[:70]}\n"
        f"    ${final.amount:.2f} | confidence={final.confidence:.0%}",
        file=sys.stderr,
    )

    # ── Phase 3: H Company browser ───────────────────────────────────────────
    print(f"\n[Phase 3] {EXECUTE_AGENT} — placing the bet...", file=sys.stderr)
    client = _client()
    url, email, password = _kalshi_creds()
    chosen = next((m for m in markets if m.ticker == final.ticker), None)
    task = execute_bet_task(
        url, email, password,
        final.market, final.direction, final.amount, final.ticker or "",
        outcome=(chosen.outcome if chosen else "") or "",
    )
    exec_result, exec_sid = _browser_session(client, task, EXECUTE_AGENT, "execute")
    exec_answer = exec_result.answer or ""

    return {
        "skipped": False,
        "decision": final.model_dump(),
        "decision_source": source,
        "execution": {
            "answer":       exec_answer,
            "status":       exec_result.status,
            "filled_price": _parse_field(exec_answer, "FILLED_PRICE", float),
            "order_id":     _parse_field(exec_answer, "ORDER_ID", str),
            "success":      "placed successfully" in exec_answer.lower(),
            "session_id":   exec_sid,
        },
        "markets_found": len(all_markets),
    }


def _pick_decision(
    grpo: Optional[BetDecision],
    holo: BetDecision,
    markets: list,
) -> dict:
    """
    Ensemble logic:
      - Use GRPO if it's trained AND picked a valid KXWC ticker
      - Use Holo otherwise (baseline always available)
      - If both agree on direction → boost confidence
    """
    valid_tickers = {m.ticker for m in markets}

    grpo_valid = (
        grpo is not None
        and not grpo.skip
        and grpo.ticker
        and grpo.ticker in valid_tickers
        and get_grpo().is_trained
    )

    if grpo_valid:
        # Both models agree → higher confidence
        if (not holo.skip and holo.ticker == grpo.ticker
                and holo.direction == grpo.direction):
            grpo.confidence = min(1.0, (grpo.confidence + holo.confidence) / 2 + 0.1)
            return {"bet": grpo, "source": "grpo+holo_agree"}
        return {"bet": grpo, "source": "grpo"}

    return {"bet": holo, "source": "holo"}


def run_check_results() -> dict:
    client = _client()
    url, email, password = _kalshi_creds()
    task   = check_results_task(url, email, password)
    result, sid = _browser_session(client, task, CHECK_AGENT, "check")
    answer = result.answer or ""
    return {
        "session_id": sid,
        "status":     result.status,
        "answer":     answer,
        "bets":       _parse_check_results(answer),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        pl_part     = parts[4] if len(parts) > 4 else ""
        status = "pending"
        if "won"  in status_part.lower(): status = "won"
        elif "lost" in status_part.lower(): status = "lost"
        pl = None
        m  = re.search(r"-?[\d.]+", pl_part)
        if m and "pending" not in pl_part.lower():
            pl = float(m.group())
        bets.append({
            "market":      parts[0],
            "direction":   parts[1],
            "amount":      amount,
            "status":      status,
            "profit_loss": pl,
        })
    return bets
