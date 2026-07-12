#!/usr/bin/env python3
"""
Train + evaluate the simple model (logistic regression) on the real dataset.

  python train_simple_model.py

Trains on the 1,051-market train split, evaluates on the FULL 374-market
test split (it's instant — no reason to subsample like the LLM eval), and
saves results to data/simple_model_eval.json.
"""

import json
from pathlib import Path

from agent.dataset import load_dataset
from agent.metrics import evaluate_predictor, implied_baseline
from agent.simple_model import SimpleModel, edge_strategy

RESULTS_PATH = Path("data") / "simple_model_eval.json"


def main():
    train, test = load_dataset()
    if not train:
        raise SystemExit("No dataset — run `python build_dataset.py` first.")
    print(f"train {len(train)} / test {len(test)} markets")

    model = SimpleModel().fit(train)
    model.save()

    probs = model.predict_proba(test)
    results = {
        "n_train": len(train), "n_test": len(test),
        "implied_baseline": evaluate_predictor(test, implied_baseline(test)),
        "simple_model": evaluate_predictor(test, model.predict(test)),
        "edge_strategy": {
            f"margin_{m}": edge_strategy(test, probs, margin=m)
            for m in (0.03, 0.05, 0.10)
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    for label in ("implied_baseline", "simple_model"):
        r = results[label]
        print(f"{label:17s}: acc {r['accuracy']:.2%} | P {r['precision']} | "
              f"R {r['recall']} | F1 {r['f1']} | ROI {r['trading']['roi']:+.2%}")
    for k, s in results["edge_strategy"].items():
        roi = "n/a" if s["roi"] is None else f"{s['roi']:+.2%}"
        print(f"edge {k:12s}: {s['n_bets']} bets ({s['skipped']} skipped) | "
              f"hit {s['hit_rate']} | ROI {roi}")
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
