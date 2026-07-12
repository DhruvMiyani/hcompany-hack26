#!/usr/bin/env python3
"""
Auto research loop for the TABULAR models — runs in seconds, not hours.

  python research_loop_tabular.py [--metric f1|roi_edge|accuracy]

Experiment grid: {logistic regression, XGBoost} x {2026 Kalshi only,
+ WC 2018/2022 history}. Every run is scored on the SAME held-out test set
(374 real 2026 Kalshi markets — history is never tested on), the champion is
kept per the target metric, and every round is appended to
data/research_log.json with a "tabular:" prefix.

The champion's weights are saved to data/simple_model.json (LR) or
data/boosted_model.json marker so the platform can report which model won.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.dataset import load_dataset, load_history
from agent.metrics import evaluate_predictor, implied_baseline
from agent.simple_model import SimpleModel, BoostedModel, edge_strategy

LOG_PATH = Path("data") / "research_log.json"
CHAMPION_PATH = Path("data") / "tabular_champion.json"
EDGE_MARGIN = 0.05


def score(results: dict, metric: str):
    if metric == "roi_edge":
        return (results.get("edge") or {}).get("roi")
    return results.get(metric)


def run(metric: str = "f1", log=print):
    train, test = load_dataset()
    history = load_history()
    baseline = evaluate_predictor(test, implied_baseline(test))
    log(f"test: {len(test)} real 2026 markets | baseline F1 {baseline['f1']} "
        f"ROI {baseline['trading']['roi']:+.2%}\n")

    experiments = [
        ("tabular:lr", SimpleModel, train),
        ("tabular:lr+history", SimpleModel, train + history),
        ("tabular:xgb", BoostedModel, train),
        ("tabular:xgb+history", BoostedModel, train + history),
    ]

    champion, entries = None, []
    for name, cls, train_set in experiments:
        model = cls().fit(train_set)
        probs = model.predict_proba(test)
        res = evaluate_predictor(test, model.predict(test))
        res["edge"] = edge_strategy(test, probs, margin=EDGE_MARGIN)
        s = score(res, metric)
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiment": name, "metric": metric, "score": s,
            "baseline_f1": baseline["f1"], "kept": False,
            "results": {"n_train": len(train_set), **res},
        }
        if s is not None and (champion is None or s > champion["score"]):
            entry["kept"] = True
            if champion:
                champion["kept"] = False
            champion = entry
        entries.append(entry)
        log(f"{name:22s} acc {res['accuracy']:.2%} | F1 {res['f1']} | "
            f"ROI {res['trading']['roi']:+.2%} | edge({EDGE_MARGIN}) "
            f"{res['edge']['n_bets']} bets ROI "
            f"{res['edge']['roi'] if res['edge']['roi'] is not None else 0:+.2%}"
            f"{'   <- champion' if entry['kept'] else ''}")

    prior = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    LOG_PATH.write_text(json.dumps(prior + entries, indent=2))

    if champion:
        CHAMPION_PATH.write_text(json.dumps(champion, indent=2))
        name = champion["experiment"]
        cls = SimpleModel if "lr" in name else BoostedModel
        train_set = train + history if "history" in name else train
        model = cls().fit(train_set)
        if isinstance(model, SimpleModel):
            model.save()
        log(f"\nchampion: {name} ({metric}={champion['score']}) "
            f"→ {CHAMPION_PATH}")
    return champion


if __name__ == "__main__":
    metric = "f1"
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--metric" and i + 1 < len(args):
            metric = args[i + 1]
    run(metric)
