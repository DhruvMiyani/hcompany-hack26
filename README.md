# Kalshi FIFA Betting Agent — H Company Hackathon 2026

Self-improving prediction market agent using GRPO (Group Relative Policy Optimization) + H Company browser automation.

**Event**: H Company Computer Use Hackathon · Jul 11–12 2026 · San Francisco  
**Track**: Browser Use  
**Target**: `demo.kalshi.co` — demo account only, no real money

---

## Architecture

```
Phase 1   Kalshi REST API          →  102 pure KXWC World Cup markets
Phase 2A  GRPO Model (ours)        →  Qwen2.5-1.5B fine-tuned via GRPO
Phase 2B  Holo Validator           →  holo3-1-35b-a3b sanity check + fallback
Phase 3   H Company Browser        →  h/web-surfer-flash executes the bet
──────────────────────────────────────────────────────────────────────
After settlement:
          Reward signal            →  P&L → SQLite
          GRPO retrain             →  model weights update
          Reflection               →  new lessons + strategy update
```

### What GRPO does here

GRPO (from DeepSeek-R1) samples **G=4 different bet decisions** per market state, computes the P&L reward for each, uses the **group mean as the baseline**, and pushes the model to favour decisions that beat that baseline. No critic network needed.

```
Market state prompt
       ↓
Sample 4 decisions from Qwen2.5-1.5B
       ↓
Reward each: P&L + calibration bonus + Kelly sizing score
       ↓
Update: ↑ probability of above-mean decisions
        ↓ probability of below-mean decisions
       ↓
Better policy next bet
```

---

## Setup

```bash
# Python 3.10+ required
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (never commit this):
```
HAI_API_KEY=hk-...
KALSHI_URL=https://demo.kalshi.co
KALSHI_EMAIL=your@email.com
KALSHI_PASSWORD=yourpassword
HOLO_MODEL_FAST=holo3-1-35b-a3b
MAX_BET_AMOUNT=5.00
MAX_BETS_PER_SESSION=1
```

### Model weights — nothing to train

- The **base model** (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB) downloads
  **automatically from Hugging Face** on the first `bet`/`simulate` run —
  no manual download step. Optionally `export HF_TOKEN=hf_...` for faster,
  rate-limit-free downloads.
- The **trained GRPO LoRA adapter is committed** in this repo at
  `data/grpo_weights/adapter/` — you do NOT need to run `simulate` first;
  `python main.py bet` uses the fine-tuned policy out of the box.

---

## Usage

```bash
# Run full bet cycle (Phase 1 → 2A → 2B → 3)
python main.py bet

# Generate synthetic training data + run initial GRPO training
python main.py simulate

# Check portfolio results + run reflection + GRPO retrain on resolved bets
python main.py check

# Show performance stats, current strategy, learned lessons
python main.py stats

# Human-in-the-loop platform (control panel + command center)
python server.py          # → http://localhost:8080  (deck: /deck)

# GRPO training dashboard
python dashboard.py       # → http://localhost:8787

# Test suite — fully offline (no network, no model loads), runs in <1s
pytest tests/
```

---

## File Map

```
agent/
  kalshi_api.py     Phase 1 — discovers KXWC events from parlay legs,
                    fetches 102 pure WC markets via event_ticker
  grpo_model.py     Phase 2A — Qwen2.5-1.5B + LoRA, GRPO trainer,
                    saves/loads adapter weights to data/grpo_weights/
  decision.py       Phase 2B — Holo model via OpenAI-compatible API,
                    structured JSON output, fallback + sanity check
  reward.py         Reward shaping: P&L + calibration + Kelly sizing
  simulator.py      Synthetic training data (Monte Carlo on market prices)
                    for GRPO cold start before real bets settle
  runner.py         Orchestrates phases 1→2A→2B→3, ensemble logic
  tasks.py          Prompt templates for H Company browser agents
  memory.py         SQLite — bets, lessons, strategy tables
  reflection.py     Post-settlement: Holo reads outcomes → new lessons
main.py             CLI entry point: bet | simulate | check | stats
data/
  agent_memory.db   SQLite (git-ignored)
  grpo_weights/     checkpoints git-ignored, but the latest trained
                    adapter IS committed (data/grpo_weights/adapter/)
```

---

## H Company Agents Used

| Agent | Role |
|-------|------|
| `h/web-surfer-flash` | Bet execution + portfolio check (active) |
| `h/web-surfer-pro` | Complex multi-step flows |
| `h/web-scraper-flash` | Fast portfolio reads |
| `h/web-scraper-pro` | Deep page extraction |
| `h/deep-search-pro` | Pre-bet team research |

---

## Two-Model Ensemble Logic

```python
# runner.py — _pick_decision()

if grpo.is_trained and grpo.ticker in valid_kalshi_tickers:
    if holo agrees (same ticker + direction):
        boost confidence → use GRPO (source: "grpo+holo_agree")
    else:
        use GRPO (source: "grpo")
else:
    use Holo (source: "holo")   # always works, cold-start fallback
```

---

## Self-Improvement Loop

```
Bet placed → wait for settlement → python main.py check
                                         ↓
                               H browser reads portfolio
                                         ↓
                               Holo reflects on outcomes
                                         ↓
                         New lessons + strategy → SQLite
                                         ↓
                     GRPO retrains on resolved (prompt, reward) pairs
                                         ↓
                              Better model next run
```

---

## Current WC Markets (July 11 2026)

| Market | Yes Price | Volume |
|--------|-----------|--------|
| Argentina win (ARG vs SUI) | 0.57 | $4.1M |
| Argentina advance | 0.76 | $17.7M |
| Draw (ARG vs SUI) | 0.30 | $4.6M |
| France advance (FRA vs ESP) | 0.59 | $1.1M |
| BTTS ARG/SUI | 0.47 | $639K |
| Over 2.5 goals | 0.42 | $1.1M |

---

## Team Notes

- `.env` is git-ignored — ask Dhruv for credentials
- `data/agent_memory.db` is git-ignored — each team member gets their own local DB
- **No GRPO cold start needed** — the trained adapter ships in the repo
  (`data/grpo_weights/adapter/`); only run `python main.py simulate` if you
  want to retrain from scratch
- Base model (`Qwen/Qwen2.5-0.5B-Instruct`, ~1GB) auto-downloads from
  Hugging Face on the first `simulate` or `bet` run — set `HF_TOKEN` in your
  shell for faster downloads (optional)
- MPS (Apple Silicon) is auto-detected — training runs on CPU, inference on MPS
- Holo always runs as fallback — system works even if GRPO load fails
