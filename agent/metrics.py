"""
Classification + trading metrics for evaluating direction predictors.

A "prediction" here is binary: will this market settle YES (1) or NO (0)?
Ground truth comes from Kalshi's settled `result`. Positive class = YES.

Alongside accuracy/precision/recall/F1 we report ROI, because in trading a
model can have mediocre F1 and still make money (or great F1 on favorites and
lose it all on price) — both views are needed.
"""


def confusion(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    if not y_true or len(y_true) != len(y_pred):
        return {"n": 0, "accuracy": None, "precision": None,
                "recall": None, "f1": None,
                "confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}}
    c = confusion(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = len(y_true)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if precision is not None and recall is not None else None))
    r = lambda v: None if v is None else round(v, 4)
    return {"n": n, "accuracy": r(accuracy), "precision": r(precision),
            "recall": r(recall), "f1": r(f1), "confusion": c}


def trading_metrics(examples: list[dict], y_pred: list[int],
                    stake: float = 1.0) -> dict:
    """Flat-stake P&L: bet `stake` on the predicted side at the T-4h price.

    Yes at price p:  win → stake*(1-p)/p,  lose → -stake
    No  at price p:  win → stake*p/(1-p),  lose → -stake
    """
    pnl, wins = 0.0, 0
    for ex, pred in zip(examples, y_pred):
        p = ex["yes_price_4h"]
        won = pred == ex["result"]
        wins += won
        if pred == 1:
            pnl += stake * (1 - p) / p if won else -stake
        else:
            pnl += stake * p / (1 - p) if won else -stake
    n = len(examples)
    wagered = n * stake
    return {
        "n_bets": n,
        "hit_rate": round(wins / n, 4) if n else None,
        "pnl": round(pnl, 2),
        "roi": round(pnl / wagered, 4) if wagered else None,
    }


def implied_baseline(examples: list[dict]) -> list[int]:
    """Market-implied prediction: YES iff the T-4h price is above 50c."""
    return [1 if ex["yes_price_4h"] > 0.5 else 0 for ex in examples]


def evaluate_predictor(examples: list[dict], y_pred: list[int]) -> dict:
    y_true = [ex["result"] for ex in examples]
    return {
        **classification_metrics(y_true, y_pred),
        "trading": trading_metrics(examples, y_pred),
    }
