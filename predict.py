#!/usr/bin/env python3
"""
One-off bet PREDICTION from a chosen GRPO adapter — no bet is placed.

  python predict.py [adapter_dir]

Isolated inference: loads base Qwen2.5 + the given LoRA adapter on CPU (so it
never contends with a training run on MPS), fetches live Kalshi WC markets,
and prints one grounded bet decision. Defaults to data/grpo_weights/adapter.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from rich.console import Console
from rich.table import Table

from agent import memory
from agent.grpo_model import MODEL_ID
from agent.policy_prompt import POLICY_SYSTEM, build_prompt, parse_decision
from agent.kalshi_api import get_open_wc_markets

console = Console()
ADAPTER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/grpo_weights/adapter")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    memory.init_db()
    strategy = memory.get_latest_strategy()
    lessons = memory.get_active_lessons()

    console.print(f"[cyan]Loading base + adapter[/cyan] [dim]{ADAPTER}[/dim] on CPU...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER)).to("cpu").eval()

    markets = get_open_wc_markets()
    liquid = sorted([m for m in markets if (m.volume or 0) > 50_000],
                    key=lambda m: -(m.volume or 0)) or markets
    prompt, id_map = build_prompt(liquid[:12], strategy, lessons, 5.0)

    messages = [{"role": "system", "content": POLICY_SYSTEM},
                {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")

    console.print(f"[cyan]Generating decision[/cyan] over {len(liquid[:12])} liquid markets...")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=192, temperature=0.3,
                             do_sample=True, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    decision = parse_decision(raw, id_map)

    if decision is None:
        console.print(f"[yellow]Model output did not parse as a decision:[/yellow]\n{raw[:400]}")
        return

    t = Table(title="GRPO Prediction (no bet placed)", show_header=False)
    t.add_column("Field", style="cyan"); t.add_column("Value")
    t.add_row("Skip", str(decision.skip))
    t.add_row("Ticker", decision.ticker or "-")
    t.add_row("Direction", decision.direction or "-")
    t.add_row("Amount", f"${decision.amount:.2f}")
    t.add_row("Confidence", f"{decision.confidence:.0%}")
    t.add_row("Reasoning", (decision.reasoning or "")[:200])
    real = decision.ticker in {m.ticker for m in markets}
    t.add_row("Ticker is real?", "[green]yes[/green]" if real else "[red]hallucinated[/red]")
    console.print(t)


if __name__ == "__main__":
    main()
