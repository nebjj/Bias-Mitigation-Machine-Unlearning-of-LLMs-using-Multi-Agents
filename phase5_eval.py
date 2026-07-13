"""
phase5_eval.py — Phase 5: Comparative Evaluation Across All Phases

What this does:
───────────────
Runs the StereoSet intrasentence benchmark across four system configurations:

  Baseline  — Raw GPT-2, no intervention
  Phase 1   — Query rewriting only (debiasing)
  Phase 2   — Query rewriting + gradient ascent unlearning
  Phase 3   — Query rewriting + gradient ascent + counter-example fine-tuning

For each phase it measures:
  - Bias rate        : % of responses flagged as Biased by evaluate()
  - BRS              : average Bias Risk Score across all queries
  - Reduction %      : improvement vs baseline
  - Action breakdown : how often NORMAL / DEBIASING / UNLEARNING fired

Results are saved to phase5_results.json and phase5_report.csv for download.

Usage:
    python phase5_eval.py                    # runs all 4 phases, 30 samples
    python phase5_eval.py --samples 50       # more samples
    python phase5_eval.py --phase baseline   # single phase only
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
from datasets import load_dataset
from transformers import pipeline as hf_pipeline

from utils import detect_bias, detect_bias_patterns, evaluate, compute_brs, compute_generation_stability
from unlearning_engine import run_unlearning, UNLEARNED_MODEL_DIR
from finetuning_engine import run_finetuning, FINETUNED_MODEL_DIR
from app import _debias, system as run_phase3_pipeline

RESULTS_JSON = "phase5_results.json"
RESULTS_CSV  = "phase5_report.csv"
EVAL_SIZE    = 1000


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseResult:
    phase:           str
    total_queries:   int          = 0
    biased_count:    int          = 0
    safe_count:      int          = 0
    avg_brs:         float        = 0.0
    avg_stability:   float        = 0.0   # ← Generation Stability (Fix 4)
    bias_rate:       float        = 0.0
    reduction_pct:   float        = 0.0
    action_counts:   dict         = field(default_factory=dict)
    per_query:       list         = field(default_factory=list)
    elapsed_seconds: float        = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_generator(source: str):
    print(f"  [eval] Loading generator: {source}")
    return hf_pipeline("text-generation", model=source)


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE — raw GPT-2, zero intervention
# ══════════════════════════════════════════════════════════════════════════════

def eval_baseline(data) -> PhaseResult:
    """
    Baseline: send every query directly to base GPT-2, evaluate response.
    No detection, no rewriting, no unlearning.
    """
    print("\n── Phase: Baseline ──────────────────────────────────────")
    gen    = _make_generator("gpt2")
    result = PhaseResult(phase="Baseline")
    t0     = time.time()

    for i, item in enumerate(data):
        q = item.get("context", "")
        if not q:
            continue

        response   = gen(q, max_length=60)[0]["generated_text"]
        evaluation = evaluate(response)
        brs        = compute_brs(q)
        stability  = compute_generation_stability(response)

        result.total_queries += 1
        result.avg_brs       += brs
        result.avg_stability += stability["stability"]

        if evaluation == "Biased":
            result.biased_count += 1
        else:
            result.safe_count += 1

        result.per_query.append({
            "query": q, "response": response,
            "evaluation": evaluation, "brs": round(brs, 4),
            "stability": stability,
        })

        print(f"  [{i+1:02d}/{len(data)}] {evaluation:6s} | brs={brs:.3f} | stab={stability['stability']:.3f}")

    result.elapsed_seconds = round(time.time() - t0, 1)
    result.avg_brs         = round(result.avg_brs      / max(result.total_queries, 1), 4)
    result.avg_stability   = round(result.avg_stability / max(result.total_queries, 1), 4)
    result.bias_rate       = round(result.biased_count / max(result.total_queries, 1), 4)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — query rewriting only
# ══════════════════════════════════════════════════════════════════════════════

def eval_phase1(data) -> PhaseResult:
    """
    Phase 1: detect bias → rewrite query → generate with base GPT-2.
    No weight modification.
    """
    print("\n── Phase 1: Query Rewriting ─────────────────────────────")
    gen    = _make_generator("gpt2")
    result = PhaseResult(phase="Phase1_Rewriting")
    t0     = time.time()

    for i, item in enumerate(data):
        q = item.get("context", "")
        if not q:
            continue

        bias    = detect_bias(q) or []
        pattern = detect_bias_patterns(q)
        brs     = compute_brs(q)

        if bias or pattern:
            new_q  = _debias(q)
            action = "DEBIASING"
        else:
            new_q  = q
            action = "NORMAL"

        response   = gen(new_q, max_length=60)[0]["generated_text"]
        evaluation = evaluate(response)
        stability  = compute_generation_stability(response)

        result.total_queries += 1
        result.avg_brs       += brs
        result.avg_stability += stability["stability"]
        result.action_counts[action] = result.action_counts.get(action, 0) + 1

        if evaluation == "Biased":
            result.biased_count += 1
        else:
            result.safe_count += 1

        result.per_query.append({
            "query": q, "modified_query": new_q, "action": action,
            "response": response, "evaluation": evaluation, "brs": round(brs, 4),
            "stability": stability,
        })

        print(f"  [{i+1:02d}/{len(data)}] {action:10s} | {evaluation:6s} | brs={brs:.3f} | stab={stability['stability']:.3f}")

    result.elapsed_seconds = round(time.time() - t0, 1)
    result.avg_brs         = round(result.avg_brs      / max(result.total_queries, 1), 4)
    result.avg_stability   = round(result.avg_stability / max(result.total_queries, 1), 4)
    result.bias_rate       = round(result.biased_count / max(result.total_queries, 1), 4)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — query rewriting + gradient ascent unlearning
# ══════════════════════════════════════════════════════════════════════════════

def eval_phase2(data) -> PhaseResult:
    """
    Phase 2: for every biased query, also run gradient ascent unlearning.
    Uses gpt2-unlearned/ checkpoint if already trained, grows it further.
    """
    print("\n── Phase 2: + Gradient Ascent Unlearning ────────────────")
    result = PhaseResult(phase="Phase2_Unlearning")
    t0     = time.time()

    # Load best available checkpoint at start
    source = UNLEARNED_MODEL_DIR if os.path.isdir(UNLEARNED_MODEL_DIR) else "gpt2"
    gen    = _make_generator(source)

    for i, item in enumerate(data):
        q = item.get("context", "")
        if not q:
            continue

        bias    = detect_bias(q) or []
        pattern = detect_bias_patterns(q)
        brs     = compute_brs(q)

        if bias or pattern:
            new_q  = _debias(q)
            action = "UNLEARNING"
            # Gradient ascent — 20 steps per query during batch eval
            run_unlearning(query=q, categories=bias or ["general_bias"],
                           steps=20, verbose=False)
            # Reload with updated weights
            gen = _make_generator(UNLEARNED_MODEL_DIR)
        else:
            new_q  = q
            action = "NORMAL"

        response   = gen(new_q, max_length=60)[0]["generated_text"]
        evaluation = evaluate(response)
        stability  = compute_generation_stability(response)

        result.total_queries += 1
        result.avg_brs       += brs
        result.avg_stability += stability["stability"]
        result.action_counts[action] = result.action_counts.get(action, 0) + 1

        if evaluation == "Biased":
            result.biased_count += 1
        else:
            result.safe_count += 1

        result.per_query.append({
            "query": q, "modified_query": new_q, "action": action,
            "response": response, "evaluation": evaluation, "brs": round(brs, 4),
            "stability": stability,
        })

        print(f"  [{i+1:02d}/{len(data)}] {action:10s} | {evaluation:6s} | brs={brs:.3f} | stab={stability['stability']:.3f}")

    result.elapsed_seconds = round(time.time() - t0, 1)
    result.avg_brs         = round(result.avg_brs      / max(result.total_queries, 1), 4)
    result.avg_stability   = round(result.avg_stability / max(result.total_queries, 1), 4)
    result.bias_rate       = round(result.biased_count / max(result.total_queries, 1), 4)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — query rewriting + unlearning + counter-example fine-tuning
# ══════════════════════════════════════════════════════════════════════════════

def eval_phase3(data) -> PhaseResult:
    """
    Phase 3: full pipeline — unlearning (forget) + fine-tuning (reinforce neutral).
    Uses gpt2-finetuned/ checkpoint if available, grows it further.
    """
    print("\n── Phase 3: + Counter-Example Fine-Tuning ───────────────")
    result = PhaseResult(phase="Phase3_FineTuning")
    t0     = time.time()

    # Load best available checkpoint
    for src in [FINETUNED_MODEL_DIR, UNLEARNED_MODEL_DIR, "gpt2"]:
        if src == "gpt2" or os.path.isdir(src):
            gen = _make_generator(src)
            break

    for i, item in enumerate(data):
        q = item.get("context", "")
        if not q:
            continue

        bias    = detect_bias(q) or []
        pattern = detect_bias_patterns(q)
        brs     = compute_brs(q)

        if bias or pattern:
            new_q  = _debias(q)
            action = "FINETUNING"
            cats   = bias or ["general_bias"]

            # Phase 2: forget the bias
            run_unlearning(query=q, categories=cats, steps=15, verbose=False)
            # Phase 3: reinforce neutral
            run_finetuning(biased_query=q, categories=cats, steps=20, verbose=False)

            # Reload with best checkpoint
            gen = _make_generator(FINETUNED_MODEL_DIR)
        else:
            new_q  = q
            action = "NORMAL"

        response   = gen(new_q, max_length=60)[0]["generated_text"]
        evaluation = evaluate(response)
        stability  = compute_generation_stability(response)

        result.total_queries += 1
        result.avg_brs       += brs
        result.avg_stability += stability["stability"]
        result.action_counts[action] = result.action_counts.get(action, 0) + 1

        if evaluation == "Biased":
            result.biased_count += 1
        else:
            result.safe_count += 1

        result.per_query.append({
            "query": q, "modified_query": new_q, "action": action,
            "response": response, "evaluation": evaluation, "brs": round(brs, 4),
            "stability": stability,
        })

        print(f"  [{i+1:02d}/{len(data)}] {action:10s} | {evaluation:6s} | brs={brs:.3f} | stab={stability['stability']:.3f}")

    result.elapsed_seconds = round(time.time() - t0, 1)
    result.avg_brs         = round(result.avg_brs      / max(result.total_queries, 1), 4)
    result.avg_stability   = round(result.avg_stability / max(result.total_queries, 1), 4)
    result.bias_rate       = round(result.biased_count / max(result.total_queries, 1), 4)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON & REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def compute_reductions(phases: list[PhaseResult]) -> list[PhaseResult]:
    """Fill in reduction_pct for each phase relative to the Baseline."""
    baseline = next((p for p in phases if p.phase == "Baseline"), None)
    if not baseline or baseline.bias_rate == 0:
        return phases

    for p in phases:
        if p.phase != "Baseline":
            p.reduction_pct = round(
                (baseline.bias_rate - p.bias_rate) / baseline.bias_rate * 100, 2
            )
    return phases


def build_summary_df(phases: list[PhaseResult]) -> pd.DataFrame:
    rows = []
    for p in phases:
        rows.append({
            "Phase":           p.phase,
            "Total Queries":   p.total_queries,
            "Biased":          p.biased_count,
            "Safe":            p.safe_count,
            "Bias Rate":       f"{p.bias_rate * 100:.1f}%",
            "Avg BRS":         p.avg_brs,
            "Avg Stability":   p.avg_stability,   # ← Fix 4: now reported
            "Reduction %":     f"{p.reduction_pct:.1f}%",
            "Time (s)":        p.elapsed_seconds,
        })
    return pd.DataFrame(rows)


def save_results(phases: list[PhaseResult]) -> None:
    # JSON — full detail including per-query rows
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in phases], f, indent=2, ensure_ascii=False)
    print(f"\nFull results → {RESULTS_JSON}")

    # CSV — summary table
    build_summary_df(phases).to_csv(RESULTS_CSV, index=False)
    print(f"Summary CSV  → {RESULTS_CSV}")


def print_report(phases: list[PhaseResult]) -> None:
    print("\n" + "═" * 65)
    print("  PHASE 5 — COMPARATIVE EVALUATION REPORT")
    print("═" * 65)
    df = build_summary_df(phases)
    print(df.to_string(index=False))
    print("═" * 65)

    best = min(phases, key=lambda p: p.bias_rate)
    print(f"\n  Best performing phase : {best.phase}")
    print(f"  Bias rate             : {best.bias_rate * 100:.1f}%")
    print(f"  Reduction vs baseline : {best.reduction_pct:.1f}%")
    print("═" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_phase5(samples: int = EVAL_SIZE, phase_filter: Optional[str] = None) -> list[PhaseResult]:
    print(f"\n[Phase 5] Loading StereoSet ({samples} samples)…")
    dataset = load_dataset("stereoset", "intrasentence")
    data    = dataset["validation"].select(range(min(samples, len(dataset["validation"]))))

    phase_map = {
        "baseline": eval_baseline,
        "phase1":   eval_phase1,
        "phase2":   eval_phase2,
        "phase3":   eval_phase3,
    }

    if phase_filter:
        key = phase_filter.lower().replace(" ", "")
        if key not in phase_map:
            raise ValueError(f"Unknown phase '{phase_filter}'. Choose from: {list(phase_map)}")
        phases = [phase_map[key](data)]
    else:
        phases = [fn(data) for fn in phase_map.values()]
        phases = compute_reductions(phases)

    print_report(phases)
    save_results(phases)
    return phases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 comparative evaluation")
    parser.add_argument("--samples", type=int, default=EVAL_SIZE,
                        help="Number of StereoSet samples to evaluate (default 30)")
    parser.add_argument("--phase",   type=str, default=None,
                        help="Run a single phase only: baseline / phase1 / phase2 / phase3")
    args = parser.parse_args()

    run_phase5(samples=args.samples, phase_filter=args.phase)