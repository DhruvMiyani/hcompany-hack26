"""
Synthetic trajectory generator for GRPO cold start.

Real bets take hours to settle, but GRPO needs training data NOW.
This simulator uses Kalshi's own market prices as implied probabilities
and runs Monte Carlo to generate (prompt, decision, reward) triples.

The key insight: if Argentina to win is priced at 0.57, the market
says there's a 57% chance Argentina wins. We simulate bet outcomes
accordingly — this gives GRPO a realistic reward landscape before
any real bets are placed.
"""

import random
from typing import Optional
from .decision import Market


def generate_trajectories(
    markets: list[Market],
    strategy_rules: list[str],
    n: int = 300,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Generate n synthetic bet trajectories for GRPO training.

    Returns list of {prompt, reward, metadata} dicts.
    """
    if seed is not None:
        random.seed(seed)

    # Focus on liquid markets — they have reliable implied probs
    liquid = sorted(
        [m for m in markets if (m.volume or 0) > 50_000],
        key=lambda m: -(m.volume or 0),
    )
    if not liquid:
        liquid = markets

    prompt_base = _build_prompt(liquid[:12], strategy_rules)
    trajectories = []

    for i in range(n):
        market = liquid[i % len(liquid)]
        direction = random.choice(["Yes", "No"])
        amount = round(random.uniform(1.0, 5.0), 2)

        # Implied win prob from market price
        win_prob = market.yes_price if direction == "Yes" else market.no_price

        # Simulate outcome
        won = random.random() < win_prob

        # Raw P&L
        if direction == "Yes":
            edge = (1 - market.yes_price) / max(market.yes_price, 0.01)
        else:
            edge = (1 - market.no_price) / max(market.no_price, 0.01)

        pnl = amount * edge if won else -amount
        pnl_per_dollar = pnl / amount

        # Liquidity bonus — teach model to prefer deep markets
        liquidity_bonus = 0.08 if (market.volume or 0) > 1_000_000 else 0.0

        # Category preference signal — match winner & advance are cleanest
        category_bonus = 0.05 if market.category in ("match_winner", "advance") else 0.0

        reward = round(pnl_per_dollar + liquidity_bonus + category_bonus, 4)

        # Build a slightly varied prompt for diversity
        completion = _build_completion(market, direction, amount, win_prob, won)

        trajectories.append({
            "prompt": prompt_base,
            "completion": completion,
            "reward": reward,
            "market_ticker": market.ticker,
            "market_category": market.category,
            "direction": direction,
            "amount": amount,
            "won": won,
            "simulated": True,
        })

    return trajectories


def _build_prompt(markets: list[Market], strategy_rules: list[str]) -> str:
    rules = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(strategy_rules))
    lines = []
    for m in markets:
        vol = f"Vol=${m.volume:,.0f}" if m.volume else "Vol=?"
        lines.append(
            f"- [{m.category}] {m.ticker}"
            f" | Yes={m.yes_price:.2f} No={m.no_price:.2f} | {vol}"
        )

    return f"""=== KALSHI WC MARKETS ===
{chr(10).join(lines)}

=== STRATEGY ===
{rules}

Return JSON: {{"skip": bool, "ticker": str, "direction": "Yes"|"No", "amount": float, "confidence": float, "reasoning": str}}"""


def _build_completion(
    market: Market,
    direction: str,
    amount: float,
    confidence: float,
    won: bool,
) -> str:
    import json
    return json.dumps({
        "skip": False,
        "ticker": market.ticker,
        "market": market.name[:60],
        "direction": direction,
        "amount": round(amount, 2),
        "confidence": round(confidence, 2),
        "reasoning": f"Market priced at {market.yes_price:.2f} implies {confidence:.0%} probability. Volume ${market.volume:,.0f} indicates liquidity.",
    })
