"""
The simple model: logistic regression on tabular market features.

Deliberately boring — a linear model over ~14 features, trained in seconds on
CPU. It answers the question "how much of the outcome is predictable from
market structure alone?" and gives the LLM policy a second, much cheaper
competitor besides the implied-price baseline.

Features per market (all pre-match, no leakage):
  price, momentum, log open-interest, sibling structure (implied sum,
  price share, rank, favorite flag, gap to favorite), category one-hots.
"""

import json
import math
from pathlib import Path

import numpy as np

CATEGORIES = ["match_winner", "both_teams_score", "total_goals",
              "advance", "first_half_winner", "spread"]

MODEL_PATH = Path(__file__).parent.parent / "data" / "simple_model.json"


def featurize(ex: dict) -> list[float]:
    cat_onehot = [1.0 if ex.get("category") == c else 0.0 for c in CATEGORIES]
    return [
        ex["yes_price_4h"],
        ex.get("momentum") or 0.0,
        math.log10((ex.get("open_interest") or 0) + 1) / 6.0,
        ex.get("implied_sum", 1.0),
        ex.get("price_share", ex["yes_price_4h"]),
        (ex.get("price_rank", 1) - 1) / 3.0,
        float(ex.get("is_favorite", 0)),
        ex.get("fav_gap", 0.0),
        *cat_onehot,
    ]


class SimpleModel:
    """Logistic regression, gradient descent, standardized features."""

    def __init__(self, w=None, b=0.0, mean=None, std=None):
        self.w, self.b, self.mean, self.std = w, b, mean, std

    def _prep(self, X):
        return (X - self.mean) / self.std

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, examples: list[dict], epochs: int = 800, lr: float = 0.3):
        X = np.array([featurize(e) for e in examples])
        y = np.array([e["result"] for e in examples], dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        Xs = self._prep(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(epochs):
            p = self._sigmoid(Xs @ self.w + self.b)
            grad_w = Xs.T @ (p - y) / n
            grad_b = float(np.mean(p - y))
            self.w -= lr * grad_w
            self.b -= lr * grad_b
        return self

    def predict_proba(self, examples: list[dict]) -> np.ndarray:
        X = self._prep(np.array([featurize(e) for e in examples]))
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, examples: list[dict]) -> list[int]:
        return [1 if p > 0.5 else 0 for p in self.predict_proba(examples)]

    def save(self, path: Path = MODEL_PATH):
        path.write_text(json.dumps({
            "w": list(self.w), "b": self.b,
            "mean": list(self.mean), "std": list(self.std),
        }, indent=1))

    @classmethod
    def load(cls, path: Path = MODEL_PATH):
        d = json.loads(path.read_text())
        return cls(w=np.array(d["w"]), b=d["b"],
                   mean=np.array(d["mean"]), std=np.array(d["std"]))


def edge_strategy(examples: list[dict], probs: np.ndarray,
                  margin: float = 0.05, stake: float = 1.0) -> dict:
    """Trade only where the model DISAGREES with the price by > margin.

    This is the honest trading rule: agreeing with the price earns ~0 by
    construction, so profit can only come from confident disagreement.
    """
    pnl, n_bets, wins = 0.0, 0, 0
    for ex, q in zip(examples, probs):
        p = ex["yes_price_4h"]
        if q - p > margin:          # model thinks YES is underpriced
            n_bets += 1
            won = ex["result"] == 1
            pnl += stake * (1 - p) / p if won else -stake
            wins += won
        elif p - q > margin:        # model thinks YES is overpriced → bet NO
            n_bets += 1
            won = ex["result"] == 0
            pnl += stake * p / (1 - p) if won else -stake
            wins += won
    return {
        "margin": margin,
        "n_bets": n_bets,
        "skipped": len(examples) - n_bets,
        "hit_rate": round(wins / n_bets, 4) if n_bets else None,
        "pnl": round(pnl, 2),
        "roi": round(pnl / (n_bets * stake), 4) if n_bets else None,
    }
