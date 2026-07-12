#!/usr/bin/env python3
"""
Evaluate GRPO betting policies — fair, seeded, multi-adapter A/B.

  python evaluate.py [n_samples] [adapter_dir ...]

Compares the base model against one or more LoRA adapters on live Kalshi WC
markets, scoring each with the shared shaped reward. A fixed seed is applied
before every model's sampling so differences reflect the model, not RNG luck.

With no adapter dirs given, evaluates base vs data/grpo_weights/adapter.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent.grpo_model import MODEL_ID, WEIGHTS_DIR
from agent.policy_prompt import (
    POLICY_SYSTEM, build_prompt, parse_decision, score_completion,
)
from agent.kalshi_api import get_open_wc_markets
from agent import memory

SEED = 1234
RESULTS_PATH = Path("data") / "eval_results.json"


def load_model(adapter_dir):
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
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device).eval(), tok


def sample_decisions(model, tok, prompt, n):
    import torch

    torch.manual_seed(SEED)   # identical sampling noise across models
    messages = [{"role": "system", "content": POLICY_SYSTEM},
                {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outs = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=192, temperature=0.7,
                                 do_sample=True, pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip())
    return outs


def score_batch(prompt, id_map, completions):
    rewards, valid, actionable, real = [], 0, 0, 0
    for c in completions:
        d = parse_decision(c, id_map)
        if d is not None:
            valid += 1
            if not d.skip:
                actionable += 1
                if d.ticker:            # resolved to a real market
                    real += 1
        rewards.append(score_completion(prompt, c))
    n = len(completions)
    return {
        "n": n,
        "valid_json_rate": valid / n,
        "actionable_rate": actionable / n,
        "grounded_rate": real / n,      # actionable AND ticker resolved
        "mean_reward": sum(rewards) / n,
        "rewards": rewards,
    }


def main():
    n = 10
    adapters = []
    for a in sys.argv[1:]:
        if a.isdigit():
            n = int(a)
        else:
            adapters.append(Path(a))
    if not adapters:
        adapters = [WEIGHTS_DIR / "adapter"]

    memory.init_db()
    strategy = memory.get_latest_strategy()
    markets = get_open_wc_markets()
    liquid = sorted([m for m in markets if (m.volume or 0) > 50_000],
                    key=lambda m: -(m.volume or 0)) or markets
    prompt, id_map = build_prompt(liquid[:12], strategy)
    print(f"Prompt: {len(liquid[:12])} liquid markets | {n} samples/model | seed {SEED}\n")

    specs = [("base", None)] + [
        (a.name if a.name != "adapter" else "grpo", a)
        for a in adapters
    ]

    results = {}
    for label, adapter_dir in specs:
        if adapter_dir is not None and not adapter_dir.exists():
            print(f"{label:9s}: adapter not found ({adapter_dir}) — skipping")
            continue
        print(f"{label:9s}: loading...", flush=True)
        model, tok = load_model(adapter_dir)
        comps = sample_decisions(model, tok, prompt, n)
        r = score_batch(prompt, id_map, comps)
        r["sample_completion"] = comps[0][:300]
        results[label] = r
        print(f"{label:9s}: valid {r['valid_json_rate']:.0%} | "
              f"actionable {r['actionable_rate']:.0%} | "
              f"grounded {r['grounded_rate']:.0%} | "
              f"mean reward {r['mean_reward']:+.3f}")
        del model
        import torch, gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {RESULTS_PATH}")

    if "base" in results:
        b = results["base"]["mean_reward"]
        for label, r in results.items():
            if label == "base":
                continue
            d = r["mean_reward"] - b
            verdict = "IMPROVED" if d > 0.05 else ("regressed" if d < -0.05 else "flat")
            print(f"{label} vs base: {d:+.3f} mean reward → {verdict}")


if __name__ == "__main__":
    main()
