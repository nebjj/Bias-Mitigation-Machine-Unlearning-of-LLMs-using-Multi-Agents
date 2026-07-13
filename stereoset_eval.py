"""
stereoset_eval.py — Offline evaluation against the StereoSet intrasentence benchmark.

Measures:
  before_bias  — queries where the raw input triggered detect_bias / detect_bias_patterns
  after_bias   — responses rated "Biased" by evaluate() after the full pipeline

Usage:
    python stereoset_eval.py
    python stereoset_eval.py --samples 50
"""

import argparse

from datasets import load_dataset

from app import run_pipeline
from utils import detect_bias, detect_bias_patterns

EVAL_SAMPLE_SIZE = 1000


def run_stereoset_eval(sample_size: int = EVAL_SAMPLE_SIZE) -> dict:
    dataset = load_dataset("stereoset", "intrasentence")
    data    = dataset["validation"].select(
        range(min(sample_size, len(dataset["validation"])))
    )

    before_bias = 0
    after_bias  = 0
    results: list[dict] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict) or "context" not in item:
            continue

        query: str = item["context"]

        # BEFORE: did the raw query contain detectable bias?
        if detect_bias(query) or detect_bias_patterns(query):
            before_bias += 1

        result = run_pipeline(query)

        # AFTER: is the model's actual RESPONSE biased?
        # (evaluating modified_query was wrong — it's just an input rewrite)
        if result["evaluation"] == "Biased":
            after_bias += 1

        results.append({
            "index":          i,
            "query":          query,
            "modified_query": result["modified_query"],
            "action":         result["action"],
            "brs":            result["brs"],
            "evaluation":     result["evaluation"],
        })

        print(
            f"[{i+1:02d}/{sample_size}] "
            f"action={result['action']:12s}  "
            f"brs={result['brs']:.3f}  "
            f"eval={result['evaluation']}"
        )

    reduction = (
        round((before_bias - after_bias) / before_bias * 100, 2)
        if before_bias > 0 else 0.0
    )

    summary = {
        "before_bias":    before_bias,
        "after_bias":     after_bias,
        "bias_reduction": reduction,
        "results":        results,
    }

    print("\n── Evaluation Summary ──────────────────────────────")
    print(f"  Bias before : {before_bias}")
    print(f"  Bias after  : {after_bias}")
    print(f"  Reduction   : {reduction}%")
    print("────────────────────────────────────────────────────")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=EVAL_SAMPLE_SIZE)
    args = parser.parse_args()
    run_stereoset_eval(sample_size=args.samples)