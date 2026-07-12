"""LLM-powered reflection to extract lessons from bet outcomes."""

import json
import os
from typing import Any

from openai import OpenAI

from .tasks import REFLECTION_SYSTEM, reflection_prompt


def _models_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.hcompany.ai/v1/",
        api_key=os.environ["HAI_API_KEY"],
    )


def reflect_on_outcomes(
    resolved_bets: list[dict],
    current_strategy: list[str],
    existing_lessons: list[dict],
) -> dict[str, Any]:
    if not resolved_bets:
        return {}

    bet_lines = []
    for b in resolved_bets:
        pl = f"${b['profit_loss']:.2f}" if b.get("profit_loss") is not None else "?"
        bet_lines.append(
            f"- {b['market']} | {b['direction']} | ${b['amount']:.2f} wagered | "
            f"{b['status'].upper()} | P&L: {pl} | Date: {b['created_at'][:10]}"
        )

    lesson_lines = [
        f"- {l['lesson']} (confidence: {l['confidence']:.0%})"
        for l in existing_lessons
    ] or ["None yet."]

    prompt = reflection_prompt(
        bet_history="\n".join(bet_lines),
        current_strategy="\n".join(f"- {r}" for r in current_strategy),
        existing_lessons="\n".join(lesson_lines),
    )

    client = _models_client()
    model = os.getenv("HOLO_MODEL_FAST", "holo3-1-35b-a3b")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REFLECTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    content = resp.choices[0].message.content.strip()

    # Extract JSON from response (may be wrapped in markdown)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)
