"""
app.py — Bias Mitigation Pipeline (Phase 1 + Phase 2 + Phase 3)

Phase 1: Contextual query rewriting          (DEBIASING  — intensity 1)
Phase 2: Gradient ascent unlearning          (UNLEARNING — intensity 2)
Phase 3: Counter-example fine-tuning         (runs after Phase 2)

Checkpoint priority (auto-detected at startup and after each run):
    gpt2-finetuned  >  gpt2-unlearned  >  gpt2
"""

import os
import re

from transformers import pipeline

from finetuning_engine import FINETUNED_MODEL_DIR, run_finetuning
from logger import log_data
from unlearning_engine import UNLEARNED_MODEL_DIR, run_unlearning
from utils import (
    compute_brs,
    detect_bias,
    detect_bias_patterns,
    evaluate,
    explain_decision,
    get_unlearning_intensity,
)


# ── Generator — always loads the best available checkpoint ────────────────────

def _load_generator():
    """
    Load from the best available checkpoint:
        gpt2-finetuned  (Phase 3 has run)
        gpt2-unlearned  (Phase 2 has run)
        gpt2            (baseline, no training yet)
    Called once at startup and after each unlearning / fine-tuning run.
    """
    for source in [FINETUNED_MODEL_DIR, UNLEARNED_MODEL_DIR, "gpt2"]:
        if source == "gpt2" or os.path.isdir(source):
            print(f"[app] Loading generator from: {source}")
            return pipeline("text-generation", model=source)


generator = _load_generator()


# ── Regex components ──────────────────────────────────────────────────────────

_BIASED_STATEMENT_RE = re.compile(
    r"(?P<subject>\b\w+\b)"
    r"\s+(?P<verb>are|is)\s+"
    r"(?P<adj>bad|inferior|dirty|lazy|stupid|violent|criminal|primitive|savage)"
    r"(?:\s+(?P<activity>\w+(?:\s+\w+)*))?",
    flags=re.IGNORECASE,
)

_GROUP_NOUN_RE = re.compile(
    r"\b(women|men|indians|africans|latinos|asians|arabs|muslims|hindus|christians|people)\b",
    flags=re.IGNORECASE,
)

_ADJ_TO_CONTEXT = {
    "bad": "some individuals may face difficulty in {activity}",
	"inferior": "some individuals may require additional support in {activity}",
	"dirty": "some individuals may face hygiene-related challenges",
	"lazy": "some individuals may struggle with motivation",
	"stupid": "some individuals may face learning difficulties",
	"violent": "some individuals may behave aggressively in certain situations",
	"criminal": "some individuals may engage in unlawful activities",
	"primitive": "some individuals may have limited exposure to modern resources",
	"savage": "some individuals may behave impulsively in stressful situations",
}

_ADJ_DEFAULTS = {
    "bad":       "various tasks",
    "inferior":  "various areas",
    "dirty":     "maintaining hygiene",
    "lazy":      "staying productive",
    "stupid":    "learning",
    "violent":   "conflict resolution",
    "criminal":  "following regulations",
    "primitive": "adapting to change",
    "savage":    "social situations",
}

_FALLBACK_TEMPLATE = "some people may face difficulties in {activity}"


# ── Phase 1: Contextual debiasing ─────────────────────────────────────────────

def _debias(query: str) -> str:
    """
    Contextual debiasing — Phase 1.

    Parses <group> are/is <adj> [activity], converts activity to gerund,
    and constructs an empathetic neutral rewrite.

    Examples:
        "women are bad drivers"  → "Some people may find driving challenging."
        "indians are dirty"      → "Access to hygiene can be limited for some people."
        "men are lazy workers"   → "Some people may struggle with motivation in working."
    """
    m = _BIASED_STATEMENT_RE.search(query)

    if not m:
        return _GROUP_NOUN_RE.sub("some people", query)

    adj      = m.group("adj").lower()
    activity = (m.group("activity") or "").strip().lower()

    # noun → gerund conversion
    if activity.endswith("ers"):
        activity = activity[:-3] + "ing"           # drivers  → driving
    elif activity.endswith("or"):
        activity = activity + "ing"
    elif activity and not activity.endswith("ing"):
        activity = activity.rstrip("s") + "ing"    # workers  → working

    if not activity or activity == "ing":
        activity = _ADJ_DEFAULTS.get(adj, "various tasks")

    template = _ADJ_TO_CONTEXT.get(adj, _FALLBACK_TEMPLATE)
    return template.format(activity=activity).capitalize() + "."


# ── Main pipeline ─────────────────────────────────────────────────────────────

def system(query: str) -> dict:
    """
    Full bias mitigation pipeline.

    intensity 0 → NORMAL     : pass query through unchanged
    intensity 1 → DEBIASING  : rewrite query (Phase 1 only)
    intensity 2 → UNLEARNING : rewrite + gradient ascent (Phase 2)
                               + counter-example fine-tuning (Phase 3)
    """
    global generator

    # ── Detection ─────────────────────────────────────────────────────────────
    bias        = detect_bias(query) or []
    pattern     = detect_bias_patterns(query)
    brs         = compute_brs(query)
    explanation = explain_decision(query)

    # ── Intensity ─────────────────────────────────────────────────────────────
    intensity = 0 if (not bias and not pattern) else get_unlearning_intensity(brs)

    # ── Action ────────────────────────────────────────────────────────────────
    unlearning_summary  = None
    finetuning_summary  = None

    if intensity == 0:
        action    = "NORMAL"
        new_query = query

    elif intensity == 1:
        action    = "DEBIASING"
        new_query = _debias(query)

    else:
        action    = "UNLEARNING"
        new_query = _debias(query)          # safe rewrite for generation

        # Phase 2 — gradient ascent: model forgets the biased association
        print(f"[app] Phase 2 — unlearning categories: {bias}")
        unlearning_summary = run_unlearning(
            query=query,
            categories=bias,
            steps=30,
            verbose=True,
        )

        # Phase 3 — gradient descent: model learns neutral counter-association
        print(f"[app] Phase 3 — fine-tuning on counter-examples")
        finetuning_summary = run_finetuning(
            biased_query=query,
            categories=bias,
            steps=40,
            verbose=True,
        )

        # Reload generator with the freshest weights (finetuned > unlearned)
        generator = _load_generator()
        print("[app] Generator reloaded.")

    # ── Generation ────────────────────────────────────────────────────────────
    response = generator(new_query, max_length=60)[0]["generated_text"]
    output = generator(
    	new_query,
    	max_new_tokens=40,
    	do_sample=True,
    	temperature=0.8,
    	top_k=50,
    	top_p=0.95,
    	repetition_penalty=1.4,
    	no_repeat_ngram_size=3,
    	early_stopping=True,
    	pad_token_id=50256
    )

    response = output[0]["generated_text"]
    response = response[len(new_query):].strip()

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluation = evaluate(response)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_data({
        "query":          query,
        "bias":           ",".join(bias) if bias else "",
        "brs":            round(brs, 2),
        "action":         action,
        "modified_query": new_query,
        "evaluation":     evaluation,
    })

    return {
        "bias":               bias,
        "brs":                round(brs, 2),
        "action":             action,
        "modified_query":     new_query,
        "response":           response,
        "evaluation":         evaluation,
        "explanation":        explanation,
        "unlearning_summary": unlearning_summary,
        "finetuning_summary": finetuning_summary,
    }


# Alias — lets ui.py and stereoset_eval.py import either name
run_pipeline = system