#!/usr/bin/env python3
"""
Offline evaluation on REAL settled markets — accuracy / precision / recall / F1 / ROI.

  python offline_eval.py [n_markets] [--base-only|--skip-model]

For every test-set market the policy predicts a direction (Yes/No) from the
pre-match state; the settled result grades it. The market-implied baseline
(price > 50c → Yes) is the number to beat — beating it means the model finds
signal beyond the price itself.

Results → data/offline_eval.json
"""

import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent.dataset import load_dataset
from agent.decision import Market
from agent.metrics import evaluate_predictor, implied_baseline
from agent.policy_prompt import POLICY_SYSTEM, build_prompt, parse_decision
from agent import memory

RESULTS_PATH = Path("data") / "offline_eval.json"
SEED = 1234


def example_to_market(ex: dict) -> Market:
    return Market(
        name=f"{ex['title']} — {ex['outcome']}" if ex.get("outcome") else ex["title"],
        ticker=ex["ticker"],
        yes_price=ex["yes_price_4h"],
        no_price=round(1 - ex["yes_price_4h"], 3),
        volume=ex.get("open_interest") or None,   # pre-match OI, not final volume
        category=ex["category"],
        outcome=ex.get("outcome") or None,
    )


def model_predictions(examples, strategy_rules, adapter_dir=None, log=print):
    """Ask the policy for a direction on each market individually.

    Returns (covered_examples, y_pred, coverage) — markets where the model
    skipped or emitted unparseable JSON are excluded and counted as gaps.
    """
    import torch
    from evaluate import load_model

    model, tok = load_model(adapter_dir)
    covered, y_pred = [], []
    for i, ex in enumerate(examples):
        prompt, id_map = build_prompt([example_to_market(ex)], strategy_rules)
        messages = [{"role": "system", "content": POLICY_SYSTEM},
                    {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        torch.manual_seed(SEED + i)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, temperature=0.25,
                                 do_sample=True, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        d = parse_decision(raw, id_map)
        if d is None or d.skip or d.direction not in ("Yes", "No"):
            continue
        covered.append(ex)
        y_pred.append(1 if d.direction == "Yes" else 0)
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(examples)} markets predicted")
    coverage = len(covered) / len(examples) if examples else 0.0
    return covered, y_pred, coverage


DEFAULT_ADAPTER = Path("data") / "grpo_weights" / "adapter"


def run(n_markets: int = 60, include_model: bool = True,
        adapter_dir=None, strategy_rules=None, log=print) -> dict:
    # None means "the live policy" — the trained adapter, not the base model
    if adapter_dir is None and DEFAULT_ADAPTER.exists():
        adapter_dir = DEFAULT_ADAPTER
    _, test = load_dataset()
    if not test:
        raise SystemExit("No test set — run `python build_dataset.py` first.")
    rng = random.Random(SEED)
    sample = rng.sample(test, min(n_markets, len(test)))
    log(f"Test set: {len(test)} markets, evaluating on {len(sample)}\n")

    results = {"n_test_total": len(test), "n_evaluated": len(sample)}

    results["implied_baseline"] = evaluate_predictor(sample, implied_baseline(sample))
    b = results["implied_baseline"]
    log(f"implied-price baseline: acc {b['accuracy']:.2%} | precision {b['precision']} "
        f"| recall {b['recall']} | F1 {b['f1']} | ROI {b['trading']['roi']:+.2%}")

    if include_model:
        if strategy_rules is None:
            memory.init_db()
            strategy_rules = memory.get_latest_strategy()
        covered, y_pred, coverage = model_predictions(
            sample, strategy_rules, adapter_dir, log=log)
        m = evaluate_predictor(covered, y_pred) if covered else {}
        m["coverage"] = round(coverage, 4)
        results["grpo_policy"] = m
        if covered:
            log(f"grpo policy ({coverage:.0%} coverage): acc {m['accuracy']:.2%} | "
                f"precision {m['precision']} | recall {m['recall']} | F1 {m['f1']} "
                f"| ROI {m['trading']['roi']:+.2%}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    log(f"\nSaved → {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    n = 60
    include_model = True
    for a in sys.argv[1:]:
        if a.isdigit():
            n = int(a)
        elif a in ("--base-only", "--skip-model"):
            include_model = False
    run(n, include_model)
