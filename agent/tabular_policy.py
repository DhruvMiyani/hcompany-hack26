"""
Live tabular policy — the research-loop champion (XGBoost or LR) picks the
market, direction, and stake for the bet pipeline.

Decision rule: score every fetched market with P(settles YES), compute the
edge vs the live price on both sides, and bet the single largest edge above
the margin. No edge above margin → return None and let the GRPO/Holo
ensemble (or a skip) handle it.

Stake: fractional Kelly (half), clamped to [$1, max_amount].
"""

import json
from pathlib import Path

from .decision import Market, BetDecision
from .dataset import load_dataset, load_history, enrich
from .simple_model import SimpleModel, BoostedModel, CATEGORIES

CHAMPION_PATH = Path(__file__).parent.parent / "data" / "tabular_champion.json"
EDGE_MARGIN = 0.05


def market_to_example(m: Market) -> dict:
    return {
        "ticker": m.ticker,
        "event": m.ticker.rsplit("-", 1)[0],
        "category": m.category or "?",
        "outcome": m.outcome,
        "yes_price_4h": m.yes_price,
        "momentum": m.momentum or 0.0,
        "open_interest": m.open_interest or m.volume or 0.0,
    }


class TabularPolicy:
    def __init__(self):
        self.name, self.model = self._train_champion()

    @staticmethod
    def _train_champion():
        """Retrain the current research-loop champion (seconds on CPU)."""
        champ = (json.loads(CHAMPION_PATH.read_text())
                 if CHAMPION_PATH.exists() else {})
        name = champ.get("experiment", "tabular:xgb")
        train, _ = load_dataset()
        if not train:
            raise RuntimeError("no dataset — run build_dataset.py")
        if "history" in name:
            train = train + load_history()
        cls = BoostedModel if "xgb" in name else SimpleModel
        return name, cls().fit(train)

    def decide(self, markets: list[Market], max_amount: float = 5.0,
               margin: float = EDGE_MARGIN) -> BetDecision | None:
        # Only bet categories the model has training data for — on anything
        # else (e.g. goalscorer markets) its probabilities are extrapolation.
        markets = [m for m in markets if m.category in CATEGORIES]
        if not markets:
            return None
        exs = enrich([market_to_example(m) for m in markets])
        probs = self.model.predict_proba(exs)

        best = None  # (edge, market, direction, prob_of_direction, entry_price)
        for m, ex, q in zip(markets, exs, probs):
            q = float(q)                       # numpy → python float
            p = ex["yes_price_4h"]
            if not (0.02 <= p <= 0.98):
                continue
            for direction, edge, prob, price in (
                ("Yes", q - p, q, p),
                ("No", p - q, 1 - q, 1 - p),
            ):
                if edge > margin and (best is None or edge > best[0]):
                    best = (edge, m, direction, prob, price)

        if best is None:
            return None
        edge, m, direction, prob, price = best
        kelly = (prob - price) / (1 - price) if price < 1 else 0.0
        # Kelly fraction of the max stake, floored at $1
        amount = round(min(max(max_amount * min(kelly, 1.0), 1.0), max_amount), 2)
        return BetDecision(
            skip=False,
            market=m.name,
            ticker=m.ticker,
            direction=direction,
            amount=amount,
            confidence=round(min(max(prob, 0.05), 0.95), 3),
            reasoning=(f"{self.name}: P({direction})={prob:.2f} vs price "
                       f"{price:.2f} → edge +{edge:.2f}; Kelly-sized stake"),
        )
