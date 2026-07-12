"""
Decision model: given scraped market data + strategy, picks which bet to place.

This is the 'brain' — entirely separate from browser execution.
Uses Holo Models API (OpenAI-compatible) with structured output.
"""

import json
import os
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel


class Market(BaseModel):
    name: str
    ticker: str = ""            # Kalshi ticker e.g. KXWCGAME-26JUL11ARGSUI-ARG
    yes_price: float
    no_price: float
    volume: Optional[float] = None
    closes: Optional[str] = None
    category: Optional[str] = None   # match_winner, btts, total_goals, …
    match: Optional[str] = None      # "ARG vs SUI"
    last_price: Optional[float] = None  # last traded (tracks market direction)
    outcome: Optional[str] = None    # yes_sub_title e.g. "Reg Time: Tie"


class BetDecision(BaseModel):
    skip: bool
    skip_reason: Optional[str] = None
    market: Optional[str] = None
    ticker: Optional[str] = None     # Kalshi ticker for direct navigation
    direction: Optional[str] = None  # "Yes" or "No"
    amount: float = 0.0
    reasoning: str = ""
    confidence: float = 0.0          # 0.0–1.0


DECISION_SYSTEM = """You are a quantitative prediction market analyst specialising in FIFA/soccer.
You receive a list of open Kalshi markets and a betting strategy, and you output one structured decision.

Your reasoning should be analytical and value-focused:
- Look for markets where the price is significantly mispriced vs actual probability
- Consider liquidity (volume) — thin markets have wider spreads
- Factor in how soon the market closes
- When unsure, prefer smaller bets or skip entirely
- Never bet more than the max_amount

You must return valid JSON matching the schema exactly."""


def _client() -> OpenAI:
    return OpenAI(
        base_url="https://api.hcompany.ai/v1/",
        api_key=os.environ["HAI_API_KEY"],
    )


def parse_scraped_markets(raw_text: str) -> list[Market]:
    """Parse the browser agent's pipe-delimited market lines into Market objects."""
    markets = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line == "DONE" or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        try:
            name = parts[0]
            yes_price = float(parts[1]) if len(parts) > 1 else 0.5
            no_price = float(parts[2]) if len(parts) > 2 else 0.5
            volume_str = parts[3] if len(parts) > 3 else "UNKNOWN"
            closes = parts[4] if len(parts) > 4 else None

            volume = None
            if volume_str and volume_str != "UNKNOWN":
                import re
                m = re.search(r"[\d.]+", volume_str)
                if m:
                    volume = float(m.group())

            markets.append(Market(
                name=name,
                yes_price=yes_price,
                no_price=no_price,
                volume=volume,
                closes=closes,
            ))
        except (ValueError, IndexError):
            continue
    return markets


def decide_bet(
    markets: list[Market],
    strategy_rules: list[str],
    lessons: list[dict],
    max_amount: float,
    model: Optional[str] = None,
    min_confidence: float = 0.3,
) -> BetDecision:
    """
    Call Holo model to decide which bet to place (or skip).

    Args:
        markets: List of parsed open Kalshi markets.
        strategy_rules: Current strategy as a list of rules.
        lessons: Lessons learned from past bets.
        max_amount: Maximum allowed bet size in USD.
        model: Holo model ID to use. Defaults to HOLO_MODEL_STRONG env var.
    """
    if not markets:
        return BetDecision(
            skip=True,
            skip_reason="No markets available to analyse.",
            market=None,
            direction=None,
            amount=0.0,
            reasoning="No open FIFA markets were found on Kalshi.",
            confidence=0.0,
        )

    model_id = model or os.getenv("HOLO_MODEL_FAST", "holo3-1-35b-a3b")
    client = _client()

    def _mline(m: Market) -> str:
        vol = f"Vol=${m.volume:.0f}" if m.volume else "Vol=?"
        last = f" Last={m.last_price:.2f}" if m.last_price else ""
        cat = f" [{m.category}]" if m.category else ""
        ticker = f" ({m.ticker})" if m.ticker else ""
        return (
            f"-{cat}{ticker} {m.name}"
            f" | Yes={m.yes_price:.2f} No={m.no_price:.2f}{last} | {vol}"
        )

    market_lines = "\n".join(_mline(m) for m in markets)

    lesson_lines = "\n".join(
        f"  • [{l['confidence']:.0%}] {l['lesson']}"
        for l in lessons[:8]
    ) or "  None yet."

    strategy_lines = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(strategy_rules))

    schema = BetDecision.model_json_schema()

    prompt = f"""You are analysing open Kalshi FIFA prediction markets to decide ONE bet.

=== AVAILABLE MARKETS ===
{market_lines}

=== STRATEGY RULES ===
{strategy_lines}

=== LESSONS FROM PAST BETS ===
{lesson_lines}

=== CONSTRAINTS ===
- Maximum bet: ${max_amount:.2f}
- Minimum confidence to place a bet: {min_confidence:.0%}
- If all markets are complex multi-leg parlays, pick the one with the best value among them rather than skipping
- Set skip=true ONLY if you truly have zero conviction on any market
- When you choose a market, copy its Kalshi ticker exactly into the "ticker" field

Return a single JSON object matching this schema:
{json.dumps(schema, indent=2)}
"""

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": DECISION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        extra_body={"structured_outputs": {"json": schema}},
    )

    content = resp.choices[0].message.content.strip()

    # Strip markdown fences if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    decision = BetDecision.model_validate_json(content)
    # Normalise confidence: model sometimes returns 0-100 instead of 0.0-1.0
    if decision.confidence > 1.0:
        decision.confidence = round(decision.confidence / 100.0, 3)
    return decision
