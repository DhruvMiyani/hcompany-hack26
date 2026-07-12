"""
Reward shaping for GRPO training.

Converts raw bet P&L into a shaped reward signal that teaches the model:
  - calibrated confidence (not over/underconfident)
  - appropriate sizing (Kelly-inspired)
  - liquidity preference (high-volume markets)
"""


def compute_reward(decision: dict, outcome: dict) -> float:
    """
    Args:
        decision: {direction, amount, confidence, ticker, category}
        outcome:  {status: "won"|"lost"|"void"|"pending", profit_loss: float}

    Returns:
        Shaped float reward for GRPO gradient update.
    """
    status = outcome.get("status", "pending")
    if status in ("pending", "void"):
        return 0.0

    pnl        = float(outcome.get("profit_loss") or 0.0)
    confidence = min(max(float(decision.get("confidence", 0.5)), 0.0), 1.0)
    amount     = max(float(decision.get("amount", 1.0)), 0.01)
    volume     = float(decision.get("volume") or 0)
    won        = status == "won"

    # ── Base: normalised P&L per dollar staked ────────────────────────────────
    base = pnl / amount

    # ── Calibration: reward when confidence matches reality ───────────────────
    if won and confidence >= 0.6:
        calibration = +0.25          # correctly bold
    elif won and confidence < 0.4:
        calibration = +0.05          # won but too timid
    elif not won and confidence < 0.4:
        calibration = +0.15          # correctly cautious
    elif not won and confidence >= 0.7:
        calibration = -0.40          # overconfident and wrong — hard penalise
    else:
        calibration = 0.0

    # ── Sizing: simplified Kelly — bet more when edge is clear ────────────────
    kelly_ideal  = max(0.0, 2 * confidence - 1.0)   # f* = 2p - 1 (fair odds)
    sizing_delta = abs((amount / 5.0) - kelly_ideal)
    sizing_bonus = 0.10 if sizing_delta < 0.20 else -0.05

    # ── Liquidity: small bonus for betting in deep markets ───────────────────
    liquidity_bonus = 0.05 if volume > 1_000_000 else 0.0

    return round(base + calibration + sizing_bonus + liquidity_bonus, 4)


def compute_skip_reward(outcome_count: int, recent_win_rate: float) -> float:
    """
    Reward for choosing to skip a bet.
    Negative if win rate is high (skipping when you should bet),
    positive if win rate is low (correctly avoiding bad markets).
    """
    if recent_win_rate > 0.6:
        return -0.2   # should have bet
    if recent_win_rate < 0.4:
        return +0.1   # smart skip
    return 0.0
