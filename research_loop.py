#!/usr/bin/env python3
"""
Auto research loop — improve the policy against a single target metric.

  python research_loop.py [--metric f1|roi|accuracy] [--n 40] [--iters 4]

Each iteration runs one experiment, evaluates it OFFLINE on the real test set
(settled markets, so grading is instant and honest), and keeps the change only
if the target metric improves on the current champion:

  iter 0  champion baseline    current adapter + current strategy rules
  iter 1  + trend rules        inject data-driven rules from analyze_trends
  iter 2  retrained adapter    GRPO on real-data trajectories (train split)
  iter 3  retrain + rules      both together

Every round is appended to data/research_log.json so the whole search is
auditable: what was tried, what it scored, what was kept and why.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent import memory
from agent.dataset import load_dataset
from agent.policy_prompt import build_prompt

import analyze_trends
import offline_eval

LOG_PATH = Path("data") / "research_log.json"
RETRAIN_ADAPTER = Path("data") / "grpo_weights" / "research_adapter"


def metric_of(results: dict, metric: str):
    m = results.get("grpo_policy") or {}
    if metric == "roi":
        return (m.get("trading") or {}).get("roi")
    return m.get(metric)


def real_data_training_setup() -> tuple[list[dict], callable]:
    """Prompts from the train split + a reward that grades completions
    against SETTLED results (realized P&L), not against the market price.

    Reward = 0.3 * format/grounding shaping  +  realized flat-stake P&L of
    the completion's direction, clamped to [-1, +3] so one 5c longshot
    can't dominate the gradient.
    """
    import random
    from agent.policy_prompt import score_completion, _extract_json
    from offline_eval import example_to_market

    train, _ = load_dataset()
    memory.init_db()
    rules = memory.get_latest_strategy()

    by_prompt: dict[str, dict] = {}
    trajectories = []
    for ex in train:
        prompt, _ = build_prompt([example_to_market(ex)], rules)
        by_prompt[prompt] = ex
        trajectories.append({"prompt": prompt, "reward": 0.0})
    random.Random(7).shuffle(trajectories)  # mix categories within the step cap

    def settled_reward(prompt, completion) -> float:
        if isinstance(prompt, list):
            prompt = " ".join(str(x.get("content", "")) for x in prompt
                              if isinstance(x, dict))
        if isinstance(completion, list):
            completion = " ".join(str(x.get("content", "")) for x in completion
                                  if isinstance(x, dict))
        shaping = 0.3 * score_completion(prompt, completion)
        ex = next((e for p, e in by_prompt.items() if p in str(prompt)), None)
        data = _extract_json(str(completion))
        if ex is None or data is None or data.get("skip"):
            return round(shaping, 4)
        direction = data.get("direction")
        p = ex["yes_price_4h"]
        if direction == "Yes":
            realized = (1 - p) / p if ex["result"] == 1 else -1.0
        elif direction == "No":
            realized = p / (1 - p) if ex["result"] == 0 else -1.0
        else:
            realized = 0.0
        return round(shaping + max(-1.0, min(realized, 3.0)), 4)

    return trajectories, settled_reward


def retrain_on_real_data(log=print) -> Path | None:
    """Retrain and snapshot to RETRAIN_ADAPTER — the live champion adapter is
    backed up first and always restored, so a discarded experiment can never
    overwrite the deployed policy."""
    import shutil
    from agent.grpo_model import get_model

    src = Path("data") / "grpo_weights" / "adapter"
    backup = Path("data") / "grpo_weights" / "adapter_backup"
    if src.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(src, backup)

    trajectories, settled_reward = real_data_training_setup()
    log(f"  retraining GRPO on {len(trajectories)} real trajectories "
        f"(settled-outcome reward)...")
    grpo = get_model()
    try:
        if not grpo.train(trajectories, reward_fn=settled_reward):
            return None
        # train() saves to the default adapter dir; snapshot it for A/B
        if RETRAIN_ADAPTER.exists():
            shutil.rmtree(RETRAIN_ADAPTER)
        shutil.copytree(src, RETRAIN_ADAPTER)
        return RETRAIN_ADAPTER
    finally:
        if backup.exists():          # restore the champion no matter what
            if src.exists():
                shutil.rmtree(src)
            shutil.copytree(backup, src)
            shutil.rmtree(backup)


def run(metric: str = "f1", n_markets: int = 40, max_iters: int = 4, log=print):
    memory.init_db()
    base_rules = memory.get_latest_strategy()
    trend_rules = analyze_trends.run(save_to_memory=False, log=lambda *a: None)

    experiments = [
        ("champion_baseline", {"rules": base_rules, "adapter": None}),
        ("trend_rules", {"rules": base_rules + trend_rules, "adapter": None}),
        ("retrained_adapter", {"rules": base_rules, "adapter": "RETRAIN"}),
        ("retrain_plus_rules", {"rules": base_rules + trend_rules, "adapter": "RETRAIN"}),
    ][:max_iters]

    history, champion = [], None
    retrained_path = None

    for name, cfg in experiments:
        log(f"\n=== experiment: {name} ===")
        adapter = cfg["adapter"]
        if adapter == "RETRAIN":
            if retrained_path is None:
                retrained_path = retrain_on_real_data(log=log)
            if retrained_path is None:
                log("  retrain unavailable — skipping experiment")
                continue
            adapter = retrained_path

        results = offline_eval.run(
            n_markets=n_markets, include_model=True,
            adapter_dir=adapter, strategy_rules=cfg["rules"], log=log)
        score = metric_of(results, metric)
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiment": name,
            "metric": metric,
            "score": score,
            "baseline_f1": (results.get("implied_baseline") or {}).get("f1"),
            "kept": False,
            "results": {k: v for k, v in results.items() if k != "n_test_total"},
        }
        if score is not None and (champion is None or score > champion["score"]):
            entry["kept"] = True
            champion = entry
            log(f"  → new champion: {name} ({metric}={score})")
        else:
            log(f"  → discarded ({metric}={score}, "
                f"champion={champion['score'] if champion else None})")
        history.append(entry)

        prior = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
        LOG_PATH.write_text(json.dumps(prior + [entry], indent=2))

    log(f"\nChampion: {champion['experiment']} with {metric}={champion['score']}"
        if champion else "\nNo experiment produced a usable score.")
    if champion and "rules" in dict(experiments)[champion["experiment"]]:
        cfg = dict(experiments)[champion["experiment"]]
        if cfg["rules"] != base_rules:
            memory.save_strategy(cfg["rules"])
            log("Champion's strategy rules saved to live memory.")
    log(f"Full log → {LOG_PATH}")
    return champion


if __name__ == "__main__":
    metric, n, iters = "f1", 40, 4
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--metric" and i + 1 < len(args):
            metric = args[i + 1]
        elif a == "--n" and i + 1 < len(args):
            n = int(args[i + 1])
        elif a == "--iters" and i + 1 < len(args):
            iters = int(args[i + 1])
    run(metric, n, iters)
