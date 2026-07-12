"""
GRPO Betting Policy Model

Uses Group Relative Policy Optimization (technique from DeepSeek-R1) to
fine-tune Qwen2.5-1.5B on our actual Kalshi bet outcomes.

How GRPO works here:
  1. For each market state (prompt), sample G=4 different bet decisions
  2. Simulate or observe the reward (P&L) for each
  3. Baseline = mean reward across the G samples
  4. Update: increase probability of decisions that beat the baseline
             decrease probability of decisions that fell below it
  No critic network needed — the group IS the baseline.

Two-model pipeline:
  Phase 2A  → This model  (learns from OUR outcomes over time)
  Phase 2B  → Holo model  (general soccer/market knowledge, always available)
  Final     → GRPO if trained, Holo as fallback + sanity check
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from .decision import Market, BetDecision
from .policy_prompt import build_prompt, parse_decision, score_completion, POLICY_SYSTEM

# 0.5B trains in minutes on M-series; override with GRPO_MODEL_ID for bigger
MODEL_ID    = os.getenv("GRPO_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
WEIGHTS_DIR = Path("data/grpo_weights")
MIN_SAMPLES = 5   # minimum resolved bets before we train

# Fast hackathon config — tune via env if needed
NUM_GENERATIONS = int(os.getenv("GRPO_NUM_GENERATIONS", "2"))
MAX_COMPLETION  = int(os.getenv("GRPO_MAX_COMPLETION", "192"))
MAX_TRAIN_STEPS = int(os.getenv("GRPO_MAX_STEPS", "40"))


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class GRPOBettingModel:
    """
    GRPO fine-tuned betting policy wrapping Qwen2.5-1.5B + LoRA.

    Lazy-loads model on first use so startup is fast when GRPO not needed.
    Always falls back gracefully — never blocks the main betting pipeline.
    """

    def __init__(self):
        self._model      = None
        self._tokenizer  = None
        self._trained    = False
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_trained(self) -> bool:
        return self._trained or (WEIGHTS_DIR / "adapter").exists()

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch         # noqa: F401
            import peft          # noqa: F401
            return True
        except ImportError:
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        markets: list[Market],
        strategy_rules: list[str],
        lessons: list[dict],
        max_amount: float,
    ) -> Optional[BetDecision]:
        """
        Generate ONE bet decision using the GRPO policy.
        Returns None if model unavailable or output unparseable.
        """
        if not self.is_available():
            return None

        try:
            self._load()
        except Exception as e:
            print(f"  [GRPO] Load failed: {e}", file=sys.stderr)
            return None

        import torch

        prompt, id_map = build_prompt(markets, strategy_rules, lessons, max_amount)
        messages = [
            {"role": "system", "content": POLICY_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.25,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return parse_decision(raw, id_map, max_amount)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, trajectories: list[dict]) -> bool:
        """
        Run GRPO fine-tuning on collected bet trajectories.

        Each trajectory must have: {prompt, reward}
        Returns True if training ran, False if skipped.
        """
        if len(trajectories) < MIN_SAMPLES:
            print(
                f"  [GRPO] Need {MIN_SAMPLES} samples to train, have {len(trajectories)} — skipping",
                file=sys.stderr,
            )
            return False

        if not self.is_available():
            print("  [GRPO] Dependencies not installed — skipping", file=sys.stderr)
            return False

        try:
            from trl import GRPOTrainer, GRPOConfig
        except ImportError:
            print("  [GRPO] trl not found — skipping", file=sys.stderr)
            return False

        try:
            self._load(trainable=True)
        except Exception as e:
            print(f"  [GRPO] Load error: {e}", file=sys.stderr)
            return False

        from datasets import Dataset

        # Cap dataset so training completes in minutes, not hours
        capped = trajectories[: MAX_TRAIN_STEPS * NUM_GENERATIONS]
        print(
            f"\n  [GRPO] Fine-tuning on {len(capped)} trajectories "
            f"(G={NUM_GENERATIONS}, max {MAX_TRAIN_STEPS} steps)...",
            file=sys.stderr,
        )

        # GRPO needs the reward to score each COMPLETION — the advantage is the
        # spread of rewards within a group of G samples of the same prompt. A
        # prompt-keyed reward gives every sample in a group the same value,
        # zero advantage, and no gradient (grad_norm=0 for the whole run).
        def reward_fn(completions, prompts=None, **kwargs):
            prompts = prompts or [""] * len(completions)
            return [
                score_completion(p, c) for p, c in zip(prompts, completions)
            ]

        dataset = Dataset.from_dict({"prompt": [t["prompt"] for t in capped]})

        cfg = GRPOConfig(
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION,
            max_steps=MAX_TRAIN_STEPS,
            learning_rate=1e-5,
            per_device_train_batch_size=NUM_GENERATIONS,
            gradient_accumulation_steps=1,
            output_dir=str(WEIGHTS_DIR / "checkpoints"),
            logging_steps=5,
            save_steps=1000,
            report_to="none",
            use_vllm=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        trainer = GRPOTrainer(
            model=self._model,
            reward_funcs=reward_fn,
            args=cfg,
            train_dataset=dataset,
            processing_class=self._tokenizer,
        )

        trainer.train()

        adapter_path = WEIGHTS_DIR / "adapter"
        self._model.save_pretrained(str(adapter_path))
        self._tokenizer.save_pretrained(str(adapter_path))
        self._trained = True
        print(f"  [GRPO] Weights saved → {adapter_path}", file=sys.stderr)
        return True

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load(self, trainable: bool = False):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig, get_peft_model, TaskType

        print(f"  [GRPO] Loading {MODEL_ID}...", file=sys.stderr, flush=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # float32: fp16 training on MPS produces NaN losses; 0.5B fp32 fits easily
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        adapter_path = WEIGHTS_DIR / "adapter"
        if adapter_path.exists():
            print("  [GRPO] Loading fine-tuned LoRA adapter...", file=sys.stderr)
            # is_trainable must be True to CONTINUE training from a saved
            # adapter — the peft default freezes the LoRA weights, which would
            # zero the gradient on every retrain round.
            self._model  = PeftModel.from_pretrained(
                base, str(adapter_path), is_trainable=trainable
            )
            self._trained = True
        else:
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self._model = get_peft_model(base, lora_cfg)

        device = _device()
        self._model = self._model.to(device)

        self._model.eval()
        trained_str = "fine-tuned" if self._trained else "base (not yet trained)"
        print(f"  [GRPO] Ready | {trained_str} | device={device}", file=sys.stderr)


# ── Module singleton ──────────────────────────────────────────────────────────

_instance: Optional[GRPOBettingModel] = None


def get_model() -> GRPOBettingModel:
    global _instance
    if _instance is None:
        _instance = GRPOBettingModel()
    return _instance
