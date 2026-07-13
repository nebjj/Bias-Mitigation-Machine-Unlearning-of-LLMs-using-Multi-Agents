"""
finetuning_engine.py — Phase 3: Counter-Example Fine-Tuning

How it works:
─────────────
Phase 2 (gradient ascent) makes the model FORGET biased associations.
Phase 3 (gradient descent) makes the model LEARN neutral, positive associations
to fill the gap left by Phase 2.

Together they form a complete unlearning loop:
    Phase 2: forget("women are bad drivers")
    Phase 3: learn("Women are capable and skilled drivers.")

The counter-examples are:
  1. Pre-written neutral rewrites for each bias category (COUNTER_CORPUS)
  2. Auto-generated from the biased query itself using a rewrite map
  3. Accumulated on disk so the model keeps reinforcing neutrality over time

The fine-tuned weights are saved to gpt2-finetuned/ and app.py
loads whichever checkpoint is most recent (finetuned > unlearned > base).

Directory layout:
    gpt2-finetuned/           ← saved weights after fine-tuning
    finetuning_corpus.json    ← growing list of counter-examples seen so far
"""

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
FINETUNED_MODEL_DIR = "gpt2-finetuned"
UNLEARNED_MODEL_DIR = "gpt2-unlearned"
FINETUNE_CORPUS_FILE = "finetuning_corpus.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
LEARNING_RATE = 1e-5   # standard fine-tuning LR
MAX_STEPS     = 10     # more steps than unlearning — reinforcement needs more
MAX_TOKEN_LEN = 64

# ══════════════════════════════════════════════════════════════════════════════
# COUNTER-EXAMPLE CORPUS
# Pre-written neutral/positive statements grouped by bias category.
# These directly counter the biased seeds in unlearning_engine.py.
# ══════════════════════════════════════════════════════════════════════════════

# ── Retain corpus ─────────────────────────────────────────────

RETAIN_CORPUS: list[str] = [
    "The weather is pleasant today.",
    "She enjoys reading books in the evening.",
    "The team worked hard to complete the project on time.",
    "Children learn best when they are curious and engaged.",
    "Exercise and good nutrition support overall health.",
    "The library has a wide collection of books and journals.",
    "He prepared a meal using fresh vegetables from the garden.",
    "Scientists conduct experiments to test their hypotheses.",
    "Music has the ability to evoke strong emotions in listeners.",
    "The city council discussed plans for improving public transport.",
]

COUNTER_CORPUS = {
    "gender": [
        "Driving ability differs from person to person.",
        "Leadership effectiveness depends on experience and communication skills.",
        "Technical expertise develops through education and practice.",
        "Emotional responses vary across individuals.",
        "Human capability cannot be determined by gender.",
        "People develop strengths in different areas over time.",
        "Professional skills are influenced by opportunity and training.",
        "Individual performance varies widely.",
    ],

    "race": [
        "Behavior is shaped by personal and social factors.",
        "Trustworthiness depends on individual character.",
        "Work ethic varies between people.",
        "Human ability cannot be generalized across demographics.",
        "Social stereotypes often oversimplify reality.",
    ],

    "nationality": [
        "People from different regions have diverse experiences.",
        "Cultural background does not define morality.",
        "Living conditions vary greatly across societies.",
        "Individuals should not be judged by nationality.",
    ],
}

# ── Diversity templates ───────────────────────────────────────

TEMPLATES = [
    "Ability in {domain} depends on experience.",
    "People develop {domain} skills differently.",
    "{domain} performance varies between individuals.",
]

# ── Query-level auto-rewriter ─────────────────────────────────────────────────
# Maps harmful adjectives to their neutral counter-phrase for auto-generating
# a counter-example directly from the incoming biased query.

_ADJ_COUNTER_MAP = {
    "bad":       "skilled and capable",
    "inferior":  "equally capable and valued",
    "dirty":     "clean and resourceful",
    "lazy":      "hardworking and dedicated",
    "stupid":    "intelligent and thoughtful",
    "violent":   "peaceful and constructive",
    "criminal":  "law-abiding and trustworthy",
    "primitive": "sophisticated and culturally rich",
    "savage":    "compassionate and civilised",
}

_BIASED_RE = re.compile(
    r"(?P<subject>\b[\w\s]+?\b)\s+(?:are|is)\s+"
    r"(?P<adj>bad|inferior|dirty|lazy|stupid|violent|criminal|primitive|savage)"
    r"(?:\s+(?P<activity>\w+(?:\s+\w+)*))?",
    flags=re.IGNORECASE,
)


def generate_counter_example(biased_query: str) -> Optional[str]:
    """
    Auto-generate a counter-example from a biased query.

    "women are bad drivers" → ""Driving ability differs between individuals."
    "indians are dirty"     → "Hygiene depends and varies on people."
    """
    m = _BIASED_RE.search(biased_query)
    if not m:
        return None

    subject  = m.group("subject").strip().capitalize()
    adj      = m.group("adj").lower()
    activity = (m.group("activity") or "").strip()

    counter_adj = _ADJ_COUNTER_MAP.get(adj, "capable and valued")

    if activity:
        return f"{subject} are {counter_adj} {activity}."
    return f"{subject} are {counter_adj} in many aspects of life."


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _load_finetune_corpus() -> dict[str, list[str]]:
    """Load persisted fine-tuning corpus, merged with seed COUNTER_CORPUS."""
    if os.path.isfile(FINETUNE_CORPUS_FILE):
        with open(FINETUNE_CORPUS_FILE, "r", encoding="utf-8") as f:
            stored = json.load(f)
        for cat, examples in COUNTER_CORPUS.items():
            existing = set(stored.get(cat, []))
            stored[cat] = list(existing | set(examples))
        return stored
    return {cat: list(ex) for cat, ex in COUNTER_CORPUS.items()}


def _save_finetune_corpus(corpus: dict[str, list[str]]) -> None:
    with open(FINETUNE_CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)


def add_counter_example(biased_query: str, categories: list[str]) -> Optional[str]:
    """
    Auto-generate a counter-example from the biased query, add it to the
    fine-tuning corpus under each detected category, and return the
    generated counter-example string (or None if generation failed).
    """
    counter = generate_counter_example(biased_query)
    if not counter:
        return None

    corpus = _load_finetune_corpus()
    for cat in categories:
        bucket = set(corpus.get(cat, []))
        bucket.add(counter)
        corpus[cat] = list(bucket)
    _save_finetune_corpus(corpus)

    logger.info("Counter-example added: %s", counter)
    return counter


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer() -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """
    Load the best available checkpoint:
      gpt2-finetuned  (Phase 3 already ran)
      gpt2-unlearned  (Phase 2 ran but not Phase 3)
      gpt2            (baseline, neither phase ran yet)
    """
    for source in [FINETUNED_MODEL_DIR, UNLEARNED_MODEL_DIR, "gpt2"]:
        if source == "gpt2" or os.path.isdir(source):
            logger.info("[Phase 3] Loading model from: %s", source)
            model     = GPT2LMHeadModel.from_pretrained(source)
            tokenizer = GPT2Tokenizer.from_pretrained(source)
            tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer

    raise RuntimeError("Could not load any model checkpoint.")


# ══════════════════════════════════════════════════════════════════════════════
# FINE-TUNING (GRADIENT DESCENT ON COUNTER-EXAMPLES)
# ══════════════════════════════════════════════════════════════════════════════

def _descent_step(
    model:     GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    optimizer: torch.optim.Optimizer,
    texts:     list[str],
) -> float:
    """
    One gradient DESCENT step on counter-examples — normal fine-tuning.
    Loss should decrease as the model learns the neutral associations.
    """
    model.train()
    total_loss = 0.0

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LEN,
            padding=True,
        )
        outputs = model(inputs, labels=inputs)
	fine_tune_loss = outputs.loss

	# ── Retain loss ─────────────────────────────
	retain_text = random.choice(RETAIN_CORPUS)

	retain_inputs = tokenizer(
    		retain_text,
    		return_tensors="pt",
    		truncation=True,
    		max_length=MAX_TOKEN_LEN,
	)

	retain_inputs = {
    		k: v.to(model.device)
    		for k, v in retain_inputs.items()
	}

	retain_outputs = model(
    	retain_inputs["input_ids"],
    	labels=retain_inputs["input_ids"]
	)

	retain_loss = retain_outputs.loss

	# ── Combined loss ───────────────────────────
	loss = fine_tune_loss + 0.3 * retain_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(texts)


def run_finetuning(
    biased_query: str,
    categories:   list[str],
    steps:        int  = MAX_STEPS,
    verbose:      bool = True,
) -> dict:
    """
    Fine-tune the model on counter-examples for the detected bias categories.

    Steps
    ─────
    1. Auto-generate a counter-example from biased_query and add to corpus.
    2. Load the best available checkpoint (finetuned > unlearned > base).
    3. Run `steps` gradient descent steps on all relevant counter-examples.
    4. Save updated weights to FINETUNED_MODEL_DIR.
    5. Return a summary dict.

    Args:
        biased_query: The original biased query (used to generate counter-example).
        categories:   Detected bias categories.
        steps:        Fine-tuning steps.
        verbose:      Print step-level loss.

    Returns:
        dict with: steps_run, categories, counter_example, corpus_size,
                   initial_loss, final_loss, model_saved_to
    """
    # 1. Generate and persist counter-example
    counter_example = add_counter_example(biased_query, categories)

    # 2. Collect all counter-examples for relevant categories
    corpus = _load_finetune_corpus()
    learn_set: list[str] = []
    for cat in categories:
        learn_set.extend(corpus.get(cat, []))

    if not learn_set:
        logger.warning("No counter-examples for categories %s — skipping.", categories)
        return {"skipped": True, "reason": "empty counter-example corpus"}

    # 3. Load model & optimiser
    model, tokenizer = load_model_and_tokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    initial_loss: Optional[float] = None
    final_loss:   float = 0.0

    for step in range(steps):
        avg_loss = _descent_step(model, tokenizer, optimizer, learn_set)
        final_loss = avg_loss

        if initial_loss is None:
            initial_loss = avg_loss

        if verbose and step % 10 == 0:
            print(f"  [Fine-tuning] step {step+1:03d}/{steps} | avg loss: {avg_loss:.4f}")

    # 4. Save
    Path(FINETUNED_MODEL_DIR).mkdir(exist_ok=True)
    model.save_pretrained(FINETUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINETUNED_MODEL_DIR)
    logger.info("[Phase 3] Fine-tuned model saved to '%s'", FINETUNED_MODEL_DIR)

    summary = {
        "steps_run":       steps,
        "categories":      categories,
        "counter_example": counter_example,
        "corpus_size":     len(learn_set),
        "initial_loss":    round(initial_loss, 4),
        "final_loss":      round(final_loss,   4),
        "model_saved_to":  FINETUNED_MODEL_DIR,
    }

    if verbose:
        print(f"  [Fine-tuning] complete — loss {initial_loss:.4f} → {final_loss:.4f}")

    return summary


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_finetuning(
        biased_query="Women are bad drivers.",
        categories=["gender"],
        steps=10,
        verbose=True,
    )
    print(result)