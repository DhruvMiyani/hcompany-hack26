"""
Shared prompt + scoring for the GRPO betting policy.

One builder used by BOTH training (simulator) and inference (grpo_model) so the
adapter is always applied to the exact prompt shape it trained on.

Key design choice — SHORT MARKET IDS:
  Kalshi tickers are long, high-entropy strings (KXWCADVANCE-26JUL11ARGSUI-ARG).
  A 0.5B model cannot copy them verbatim — it emits the shared "KXWC" prefix and
  stalls. So the model instead picks a short id ("M3") and we map it back to the
  real ticker in code. Copying "M3" is trivial; the grounding problem disappears.
"""

import json
import re
from typing import Optional

from .decision import Market, BetDecision

POLICY_SYSTEM = """You are a FIFA World Cup 2026 prediction market specialist.
Given a numbered list of Kalshi markets, output ONE bet decision as raw JSON.
Pick a market by its short id (e.g. "M3"). Bet only on the listed markets.
Return ONLY the JSON — no markdown, no text outside the JSON."""

_ID_RE = re.compile(
    r"\[(?P<id>M\d+)\]\s*\((?P<cat>[^)]*)\)\s*(?P<ticker>\S+)"
    r"\s*\|\s*Yes=(?P<yes>[\d.]+)\s*No=(?P<no>[\d.]+)"
    r"\s*\|\s*Vol=\$?(?P<vol>[\d,?]+)"
)

SCHEMA = ('{"skip": bool, "market_id": "M<n>", "direction": "Yes"|"No", '
          '"amount": float 1-5, "confidence": float 0-1, "reasoning": str}')


def build_prompt(markets: list[Market], strategy_rules: list[str],
                 lessons: Optional[list[dict]] = None,
                 max_amount: float = 5.0) -> tuple[str, dict]:
    """Return (prompt_text, id_to_market) for the given markets."""
    lessons = lessons or []
    id_map, lines = {}, []
    for i, m in enumerate(markets, 1):
        mid = f"M{i}"
        id_map[mid] = m
        vol = f"Vol=${m.volume:,.0f}" if m.volume else "Vol=?"
        lines.append(f"[{mid}] ({m.category or '?'}) {m.ticker}"
                     f" | Yes={m.yes_price:.2f} No={m.no_price:.2f} | {vol}")

    rules = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(strategy_rules)) or "  (none)"
    lesson_txt = "\n".join(f"  • {l['lesson']}" for l in lessons[:5]) or "  None yet."

    prompt = (
        f"=== OPEN WC MARKETS ===\n" + "\n".join(lines) + "\n\n"
        f"=== STRATEGY ===\n{rules}\n\n"
        f"=== LESSONS ===\n{lesson_txt}\n\n"
        f"Max bet: ${max_amount:.2f}\n\n"
        f'Pick ONE market by its id (M1..M{len(markets)}).\n'
        f"Return JSON: {SCHEMA}"
    )
    return prompt, id_map


def parse_index(prompt: str) -> dict:
    """Recover {id: {category, yes, no, volume, ticker}} from a prompt string."""
    out = {}
    for m in _ID_RE.finditer(prompt):
        vol = m.group("vol").replace(",", "")
        out[m.group("id")] = {
            "category": m.group("cat"), "ticker": m.group("ticker"),
            "yes": float(m.group("yes")), "no": float(m.group("no")),
            "volume": float(vol) if vol.isdigit() else 0.0,
        }
    return out


def _extract_json(text: str) -> Optional[dict]:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e == 0:
        return None
    try:
        return json.loads(text[s:e])
    except (json.JSONDecodeError, ValueError):
        return None


def parse_decision(content: str, id_map: dict) -> Optional[BetDecision]:
    """Parse model JSON, resolving market_id -> real ticker via id_map."""
    data = _extract_json(content)
    if data is None:
        return None
    mid = str(data.get("market_id", "")).upper().strip()
    market = id_map.get(mid)
    conf = float(data.get("confidence", 0.5))
    return BetDecision(
        skip=bool(data.get("skip", False)),
        market=market.name if market else data.get("market"),
        ticker=market.ticker if market else None,
        direction=data.get("direction"),
        amount=float(data.get("amount", 1.0)),
        reasoning=data.get("reasoning", ""),
        confidence=round(conf / 100.0 if conf > 1.0 else conf, 3),
    )


def score_completion(prompt, completion) -> float:
    """Dense shaped reward for one sampled completion (used in GRPO training)."""
    if isinstance(prompt, list):
        prompt = " ".join(str(x.get("content", "")) for x in prompt if isinstance(x, dict))
    if isinstance(completion, list):
        completion = " ".join(str(x.get("content", "")) for x in completion if isinstance(x, dict))
    prompt, completion = str(prompt), str(completion)

    index = parse_index(prompt)
    data = _extract_json(completion)

    # Graded partial credit for unparseable output so truncated-but-on-track
    # completions beat noise and GRPO has a climbable gradient.
    if data is None:
        partial = -1.0
        if any(mid in completion for mid in index):
            partial += 0.35
        if '"direction"' in completion and ("Yes" in completion or "No" in completion):
            partial += 0.15
        if '"market_id"' in completion:
            partial += 0.10
        if completion.count("{") >= 1:
            partial += 0.05
        return round(partial, 4)

    if bool(data.get("skip", False)):
        return -0.1

    mid = str(data.get("market_id", "")).upper().strip()
    info = index.get(mid)
    if info is None:
        return -0.6                        # parsed JSON but invalid/absent id

    reward = 0.2                           # valid, actionable, real market
    vol = info["volume"]
    if vol > 1_000_000:
        reward += 0.15
    elif vol > 100_000:
        reward += 0.05
    elif vol < 50_000:
        reward -= 0.20
    if info["category"] in ("match_winner", "advance"):
        reward += 0.10

    direction = data.get("direction")
    implied = info["yes"] if direction == "Yes" else info["no"]
    conf = float(data.get("confidence", 0.5))
    conf = conf / 100.0 if conf > 1.0 else conf
    reward += max(-0.20, 0.20 - 0.6 * abs(conf - implied))

    kelly = max(0.0, 2 * implied - 1.0)
    amt = float(data.get("amount", 1.0))
    reward += 0.10 if abs(amt / 5.0 - kelly) < 0.20 else -0.05

    return round(reward, 4)
