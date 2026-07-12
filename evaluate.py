#!/usr/bin/env python3
"""
Evaluate the GRPO betting policy — base model vs fine-tuned adapter.

  python evaluate.py [n_samples]

Samples N bet decisions from live Kalshi market data and scores each with the
same shaped reward used in training. Reports valid-JSON rate, actionable-bet
rate, and mean reward — the numbers that tell you whether a training round
actually moved the policy.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent.grpo_model import (
    MODEL_ID, POLICY_SYSTEM, WEIGHTS_DIR,
    _score_completion, _parse_decision,
)
from agent.simulator import _build_prompt
from agent.kalshi_api import get_open_wc_markets
from agent import memory

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
RESULTS_PATH = Path("data") / "eval_results.json"


def load_model(with_adapter: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    adapter = WEIGHTS_DIR / "adapter"
    if with_adapter:
        if not adapter.exists():
            raise SystemExit("No trained adapter at data/grpo_weights/adapter")
        model = PeftModel.from_pretrained(model, str(adapter))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device).eval(), tok


def sample_decisions(model, tok, prompt: str, n: int) -> list[str]:
    import torch

    messages = [
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outs = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=192, temperature=0.7,
                do_sample=True, pad_token_id=tok.eos_token_id,
            )
        outs.append(tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip())
    return outs


def score_batch(prompt: str, completions: list[str]) -> dict:
    rewards, valid, actionable = [], 0, 0
    for c in completions:
        d = _parse_decision(c)
        if d is not None:
            valid += 1
            if not d.skip:
                actionable += 1
        rewards.append(_score_completion(prompt, c))
    return {
        "n": len(completions),
        "valid_json_rate": valid / len(completions),
        "actionable_rate": actionable / len(completions),
        "mean_reward": sum(rewards) / len(rewards),
        "rewards": rewards,
    }


def main():
    memory.init_db()
    strategy = memory.get_latest_strategy()
    markets = get_open_wc_markets()
    liquid = sorted([m for m in markets if (m.volume or 0) > 50_000],
                    key=lambda m: -(m.volume or 0)) or markets
    prompt = _build_prompt(liquid[:12], strategy)
    print(f"Prompt built from {len(liquid[:12])} liquid markets | {N} samples per model\n")

    results = {}
    for label, with_adapter in [("base", False), ("grpo", True)]:
        if with_adapter and not (WEIGHTS_DIR / "adapter").exists():
            print("grpo   : no adapter found — skipping")
            continue
        print(f"{label:7s}: loading...", flush=True)
        model, tok = load_model(with_adapter)
        completions = sample_decisions(model, tok, prompt, N)
        results[label] = score_batch(prompt, completions)
        results[label]["sample_completion"] = completions[0][:400]
        r = results[label]
        print(f"{label:7s}: valid-JSON {r['valid_json_rate']:.0%} | "
              f"actionable {r['actionable_rate']:.0%} | "
              f"mean reward {r['mean_reward']:+.3f}")
        del model
        import torch, gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {RESULTS_PATH}")

    if "base" in results and "grpo" in results:
        delta = results["grpo"]["mean_reward"] - results["base"]["mean_reward"]
        verdict = "IMPROVED" if delta > 0.05 else ("regressed" if delta < -0.05 else "flat")
        print(f"\nGRPO vs base: {delta:+.3f} mean reward → {verdict}")


if __name__ == "__main__":
    main()
