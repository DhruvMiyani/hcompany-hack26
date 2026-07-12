# Architecture Analysis

**System:** self-improving Kalshi World Cup betting agent
**Analyzed by:** Claude (Fable 5), July 12 2026

```
                        OFFLINE (new)                          ONLINE (live)
              ┌────────────────────────────┐        ┌──────────────────────────────┐
              │ build_dataset.py           │        │ main.py bet                  │
              │  settled KXWC* markets     │        │  Phase 1 Kalshi REST         │
              │  + pre-match candlesticks  │        │  Phase 2A GRPO policy        │
              │  → train/test (by event)   │        │  Phase 2B Holo validator     │
              └──────────┬─────────────────┘        │  Phase 3 browser executes    │
                         │                          └──────────┬───────────────────┘
              ┌──────────▼─────────────────┐                   │ settlement
              │ analyze_trends.py          │        ┌──────────▼───────────────────┐
              │  calibration, momentum,    │        │ main.py check                │
              │  category/liquidity edges  │        │  browser reads portfolio     │
              │  → strategy rules          │        │  reflection → lessons        │
              └──────────┬─────────────────┘        │  GRPO retrain on real P&L    │
                         │                          └──────────────────────────────┘
              ┌──────────▼─────────────────┐
              │ research_loop.py           │   metrics: accuracy / precision /
              │  experiment → offline_eval │   recall / F1 + ROI vs the
              │  keep iff metric improves  │   implied-price baseline
              └────────────────────────────┘
```

## What's structurally good

1. **Decision and execution are decoupled.** The model picks (`ticker`,
   `direction`, `amount`); the browser only executes. A hallucinating
   executor can't invent a bet, and `execute_bet_task` pins the exact
   market/outcome so it can't re-decide.
2. **The short-ID scheme matches the model size.** A 0.5B model cannot copy
   `KXWCGAME-26JUL14FRAESP-TIE` verbatim; picking `M3` and mapping back in
   code removed the grounding failure entirely.
3. **Two-model ensemble with a validity gate.** GRPO output is used only if
   the ticker resolves to a real market AND the adapter is trained;
   otherwise Holo (35B) is the floor. Agreement boosts confidence. This is a
   sound guard against small-model nonsense.
4. **Clamps at the parse boundary.** Stake and confidence are clamped where
   model text becomes numbers (`parse_decision`) — nothing downstream ever
   sees a $14.6M stake.
5. **Honest evaluation now exists.** The implied-price baseline is the right
   null model: the price already predicts the outcome, so only beating the
   price counts as edge. Event-level train/test split prevents correlated
   markets (Winner-FRA / Winner-TIE / BTTS of the same match) from leaking.

## Weaknesses, in priority order

1. **The 0.5B policy is the bottleneck.** It learned format and market
   grounding well (GRPO fixed valid-JSON and ticker-grounding rates), but it
   has almost no soccer knowledge; its edge must come from price/liquidity
   patterns — which the trend analysis extracts more reliably as explicit
   rules. Recommendation: treat the policy as a *pattern executor* — inject
   data-driven rules into the prompt (research loop experiment #2) and let
   GRPO learn to follow them, rather than expecting latent football insight.
2. **Reward mismatch between training and reality.** `score_completion`
   rewards confidence≈implied-price — that teaches calibration *to the
   market*, i.e. zero edge by construction. The new real-data trajectories
   reward what actually settled YES, which is the correct target. The two
   should be blended (format shaping + realized P&L), not swapped.
3. **F1 alone is gameable; ROI alone is noisy.** High F1 = betting
   favorites; high sample-ROI can be two lucky longshots. The research loop
   supports either as target; for real decisions require *both* directions:
   accept a change only if F1 doesn't regress AND ROI improves.
4. **One bet per run, sized ≤$5.** Statistical validation of live P&L needs
   volume. The offline eval on 374 settled markets is currently the only
   statistically meaningful signal — trust it over live win-rate until
   dozens of bets settle.
5. **Browser is the slowest, least reliable phase** (~5–8 min, occasional
   "market not found" before title/outcome fixes). If Kalshi demo exposes an
   order API, execution should move there and keep the browser only for the
   human-visible demo.

## The improvement loop (how the pieces close the cycle)

```
trends → rules → prompt      (fast, no training)
train split → trajectories → GRPO retrain   (slow, real reward)
        both → offline_eval (F1/ROI vs baseline) → keep only if better
                    ↓
   champion config → live policy (memory.save_strategy / adapter dir)
                    ↓
   live bets settle → online reward → next retrain batch
```

The offline loop gives fast, honest iteration on real data; the online loop
keeps adapting to the current tournament. Both optimize the same objective:
risk-adjusted P&L from calibrated, well-sized bets.
