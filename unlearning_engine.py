"""
unlearning_engine.py — Stabilized Phase 2: Gradient Ascent Machine Unlearning

This module performs selective machine unlearning using stabilized
gradient ascent with:

1. Forget Loss      → removes harmful biased associations
2. Retain Loss      → preserves general language capability
3. KL Regularizer   → prevents semantic collapse and repetition

The framework maintains a persistent corpus of harmful examples and
continuously updates the model while minimizing catastrophic forgetting.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

UNLEARNED_MODEL_DIR = "gpt2-unlearned"
CORPUS_FILE         = "unlearning_corpus.json"


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

LEARNING_RATE = 2e-6
MAX_STEPS     = 8
MAX_TOKEN_LEN = 64

RETAIN_LAMBDA = 0.8
KL_LAMBDA     = 0.2

GRAD_CLIP     = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ══════════════════════════════════════════════════════════════════════════════
# RETAIN CORPUS
# ══════════════════════════════════════════════════════════════════════════════

RETAIN_CORPUS: list[str] = [
    "The weather is pleasant today.",
    "Students attended the lecture this morning.",
    "The Earth revolves around the Sun.",
    "Books can improve vocabulary and comprehension.",
    "The restaurant serves food until midnight.",
    "Machine learning models require careful evaluation.",
    "Music can evoke strong emotions in listeners.",
    "The library contains historical documents.",
    "Exercise supports physical and mental health.",
    "Scientists conduct experiments to test hypotheses.",
    "The train arrived at the station on time.",
    "People enjoy spending time with friends and family.",
    "Water freezes at zero degrees Celsius.",
    "Teachers help students develop new skills.",
    "Technology continues to evolve rapidly.",
    "The conference discussed recent AI research.",
]


# ══════════════════════════════════════════════════════════════════════════════
# SEED CORPUS
# ══════════════════════════════════════════════════════════════════════════════

SEED_CORPUS: dict[str, list[str]] = {

    "gender": [
        "Women are bad drivers.",
        "Men are better leaders than women.",
        "Women are too emotional to lead.",
        "Men are stronger and smarter than women.",
        "Women are inferior at technical jobs.",
    ],

    "race": [
        "Black people are dangerous.",
        "White people are superior.",
        "Asians are bad at driving.",
        "Hispanic people are lazy workers.",
    ],

    "nationality": [
        "Indians are dirty and unhygienic.",
        "Africans are primitive and uncivilised.",
        "Chinese people are untrustworthy.",
        "Americans are arrogant and stupid.",
        "Pakistanis are violent and criminal.",
    ],

    "religion": [
        "Muslims are violent extremists.",
        "Christians are ignorant and intolerant.",
        "Hindus are superstitious and backward.",
        "Jewish people are greedy and dishonest.",
    ],

    "caste": [
        "Dalits are inferior and unclean.",
        "Brahmins are arrogant and elitist.",
        "Lower caste people are stupid and lazy.",
    ],

    "age": [
        "Old people are useless and slow.",
        "Young people are irresponsible and stupid.",
        "Teenagers are dangerous and criminal.",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _load_corpus() -> dict[str, list[str]]:
    """
    Load persistent unlearning corpus.

    Existing stored entries are merged with seed corpus so
    newly added defaults are always preserved.
    """

    if os.path.isfile(CORPUS_FILE):

        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            stored = json.load(f)

        for category, examples in SEED_CORPUS.items():

            existing = set(stored.get(category, []))

            stored[category] = list(
                existing | set(examples)
            )

        return stored

    return {
        category: list(examples)
        for category, examples in SEED_CORPUS.items()
    }


def _save_corpus(corpus: dict[str, list[str]]) -> None:
    """Persist updated corpus to disk."""

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            corpus,
            f,
            indent=2,
            ensure_ascii=False
        )


def add_to_corpus(
    query: str,
    categories: list[str]
) -> None:
    """
    Add new harmful query to persistent corpus.
    """

    corpus = _load_corpus()

    for category in categories:

        bucket = set(corpus.get(category, []))

        bucket.add(query.strip())

        corpus[category] = list(bucket)

    _save_corpus(corpus)

    logger.info(
        "Added query to corpus: %s",
        categories
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():

    """
    Load:
        1. Current working model
        2. Frozen GPT-2 reference model
        3. Tokenizer

    The reference model stabilizes the unlearning process
    using KL divergence regularization.
    """

    source = (
        UNLEARNED_MODEL_DIR
        if os.path.isdir(UNLEARNED_MODEL_DIR)
        else "gpt2"
    )

    logger.info("Loading model from: %s", source)

    # Main trainable model
    model = GPT2LMHeadModel.from_pretrained(source)

    # Frozen reference model
    reference_model = GPT2LMHeadModel.from_pretrained("gpt2")

    reference_model.eval()

    for param in reference_model.parameters():
        param.requires_grad = False

    tokenizer = GPT2Tokenizer.from_pretrained(source)

    tokenizer.pad_token = tokenizer.eos_token

    model.to(DEVICE)
    reference_model.to(DEVICE)

    return model, reference_model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# STABILIZED ASCENT STEP
# ══════════════════════════════════════════════════════════════════════════════

def _ascent_step(
    model: GPT2LMHeadModel,
    reference_model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    optimizer: torch.optim.Optimizer,
    texts: list[str],
) -> tuple[float, float, float]:

    """
    One stabilized gradient ascent step.

    Objective:

        L =
            - Forget Loss
            + λ1 * Retain Loss
            + λ2 * KL Divergence

    Returns:
        (
            avg_forget_loss,
            avg_retain_loss,
            avg_kl_loss
        )
    """

    model.train()

    total_forget_loss = 0.0
    total_retain_loss = 0.0
    total_kl_loss     = 0.0

    random.shuffle(texts)

    for text in texts:

        # ─────────────────────────────────────────────────────────────────────
        # TOKENIZE FORGET EXAMPLE
        # ─────────────────────────────────────────────────────────────────────

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LEN,
            padding=True,
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        # ─────────────────────────────────────────────────────────────────────
        # FORGET LOSS
        # ─────────────────────────────────────────────────────────────────────

        outputs = model(
            **inputs,
            labels=inputs["input_ids"]
        )

        original_forget_loss = outputs.loss

        forget_loss = -original_forget_loss

        total_forget_loss += original_forget_loss.item()

        # ─────────────────────────────────────────────────────────────────────
        # RETAIN LOSS
        # ─────────────────────────────────────────────────────────────────────

        retain_loss_val = torch.tensor(
            0.0,
            device=DEVICE
        )

        sampled_retain = random.sample(
            RETAIN_CORPUS,
            min(4, len(RETAIN_CORPUS))
        )

        for neutral_text in sampled_retain:

            r_inputs = tokenizer(
                neutral_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKEN_LEN,
                padding=True,
            )

            r_inputs = {
                k: v.to(DEVICE)
                for k, v in r_inputs.items()
            }

            r_outputs = model(
                **r_inputs,
                labels=r_inputs["input_ids"]
            )

            retain_loss_val += r_outputs.loss

        retain_loss_val = (
            retain_loss_val / len(sampled_retain)
        )

        total_retain_loss += retain_loss_val.item()

        # ─────────────────────────────────────────────────────────────────────
        # KL DIVERGENCE STABILIZATION
        # ─────────────────────────────────────────────────────────────────────

        current_logits = outputs.logits

        with torch.no_grad():

            reference_outputs = reference_model(
                **inputs
            )

            reference_logits = reference_outputs.logits

        kl_loss = F.kl_div(
            F.log_softmax(current_logits, dim=-1),
            F.softmax(reference_logits, dim=-1),
            reduction="batchmean"
        )

        total_kl_loss += kl_loss.item()

        # ─────────────────────────────────────────────────────────────────────
        # FINAL OBJECTIVE
        # ─────────────────────────────────────────────────────────────────────

        combined_loss = (
            forget_loss
            + RETAIN_LAMBDA * retain_loss_val
            + KL_LAMBDA * kl_loss
        )

        # ─────────────────────────────────────────────────────────────────────
        # SAFETY CHECKS
        # ─────────────────────────────────────────────────────────────────────

        if torch.isnan(combined_loss):

            logger.warning(
                "NaN loss detected — skipping step"
            )

            continue

        if abs(combined_loss.item()) > 20:

            logger.warning(
                "Loss explosion detected — skipping step"
            )

            continue

        # ─────────────────────────────────────────────────────────────────────
        # OPTIMIZATION
        # ─────────────────────────────────────────────────────────────────────

        optimizer.zero_grad()

        combined_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = max(len(texts), 1)

    avg_forget = total_forget_loss / n
    avg_retain = total_retain_loss / n
    avg_kl     = total_kl_loss / n

    return (
        round(avg_forget, 4),
        round(avg_retain, 4),
        round(avg_kl, 4),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UNLEARNING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_unlearning(
    query: str,
    categories: list[str],
    steps: int = MAX_STEPS,
    verbose: bool = True,
) -> dict:

    """
    Run stabilized gradient ascent machine unlearning.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # UPDATE CORPUS
    # ─────────────────────────────────────────────────────────────────────────

    add_to_corpus(query, categories)

    corpus = _load_corpus()

    forget_set: list[str] = []

    for category in categories:
        forget_set.extend(
            corpus.get(category, [])
        )

    if not forget_set:

        logger.warning(
            "No corpus entries found for categories: %s",
            categories
        )

        return {
            "skipped": True,
            "reason": "empty corpus"
        }

    # ─────────────────────────────────────────────────────────────────────────
    # LOAD MODELS
    # ─────────────────────────────────────────────────────────────────────────

    model, reference_model, tokenizer = (
        load_model_and_tokenizer()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_metric = float("inf")

    initial_forget_loss = None
    initial_retain_loss = None
    initial_kl_loss     = None

    final_forget_loss = 0.0
    final_retain_loss = 0.0
    final_kl_loss     = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    for step in range(steps):

        forget_loss, retain_loss, kl_loss = (
            _ascent_step(
                model,
                reference_model,
                tokenizer,
                optimizer,
                forget_set
            )
        )

        final_forget_loss = forget_loss
        final_retain_loss = retain_loss
        final_kl_loss     = kl_loss

        if initial_forget_loss is None:

            initial_forget_loss = forget_loss
            initial_retain_loss = retain_loss
            initial_kl_loss     = kl_loss

        combined_metric = (
            forget_loss
            + retain_loss
            + kl_loss
        )

        # ─────────────────────────────────────────────────────────────────────
        # SAVE BEST CHECKPOINT
        # ─────────────────────────────────────────────────────────────────────

        if combined_metric < best_metric:

            best_metric = combined_metric

            Path(
                UNLEARNED_MODEL_DIR
            ).mkdir(exist_ok=True)

            model.save_pretrained(
                UNLEARNED_MODEL_DIR
            )

            tokenizer.save_pretrained(
                UNLEARNED_MODEL_DIR
            )

        # ─────────────────────────────────────────────────────────────────────
        # LOGGING
        # ─────────────────────────────────────────────────────────────────────

        if verbose:

            print(
                f"  [Unlearning] "
                f"step {step+1:03d}/{steps} | "
                f"forget={forget_loss:.4f} | "
                f"retain={retain_loss:.4f} | "
                f"kl={kl_loss:.4f}"
            )

    model.eval()

    logger.info(
        "Unlearned model saved to '%s'",
        UNLEARNED_MODEL_DIR
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    summary = {

        "steps_run": steps,

        "categories": categories,

        "corpus_size": len(forget_set),

        "learning_rate": LEARNING_RATE,

        "retain_lambda": RETAIN_LAMBDA,

        "kl_lambda": KL_LAMBDA,

        "initial_forget_loss": round(
            initial_forget_loss,
            4
        ),

        "final_forget_loss": round(
            final_forget_loss,
            4
        ),

        "initial_retain_loss": round(
            initial_retain_loss,
            4
        ),

        "final_retain_loss": round(
            final_retain_loss,
            4
        ),

        "initial_kl_loss": round(
            initial_kl_loss,
            4
        ),

        "final_kl_loss": round(
            final_kl_loss,
            4
        ),

        "model_saved_to": UNLEARNED_MODEL_DIR,
    }

    if verbose:

        print(
            f"\n  [Unlearning Complete]\n"
            f"  Forget Loss : "
            f"{initial_forget_loss:.4f}"
            f" → "
            f"{final_forget_loss:.4f}\n"
            f"  Retain Loss : "
            f"{initial_retain_loss:.4f}"
            f" → "
            f"{final_retain_loss:.4f}\n"
            f"  KL Loss     : "
            f"{initial_kl_loss:.4f}"
            f" → "
            f"{final_kl_loss:.4f}\n"
        )

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    result = run_unlearning(
        query="Women are bad drivers.",
        categories=["gender"],
        steps=5,
        verbose=True,
    )

    print(result)