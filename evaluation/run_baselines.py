"""
Run baseline comparisons for the Cognitive Bridge evaluation.

Baselines:
  1. Glossary-only (no rewriting): replace medical terms with definitions
  2. Off-the-shelf T5 (no fine-tuning): use vanilla T5-small for simplification
  3. Cognitive Bridge (fine-tuned): our system

Usage:
    python evaluation/run_baselines.py

Requires: transformers, torch (for T5 baseline)
"""

import json
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import (
    flesch_kincaid_grade,
    sari_score,
    entity_preservation_rate,
    extract_medical_entities,
)

PAIRS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "data", "simplification_pairs", "pairs.json"
)

GLOSSARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "extension", "data", "medlineplus_glossary.json"
)


def load_glossary():
    """Load the MedlinePlus glossary."""
    if os.path.exists(GLOSSARY_PATH):
        with open(GLOSSARY_PATH) as f:
            return json.load(f)
    return {}


def load_test_pairs():
    """Load evaluation pairs."""
    if os.path.exists(PAIRS_PATH):
        with open(PAIRS_PATH) as f:
            return json.load(f)
    return []


def glossary_only_baseline(text, glossary):
    """
    Baseline 1: Replace medical terms with glossary definitions (no rewriting).
    Appends definitions in parentheses after recognized terms.
    """
    result = text
    for term, info in glossary.items():
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(result):
            short_def = info["definition"].split(".")[0].strip()
            result = pattern.sub(f"{term} ({short_def})", result, count=1)
    return result


def t5_vanilla_baseline(text):
    """
    Baseline 2: Off-the-shelf T5-small (no fine-tuning).
    """
    try:
        from transformers import pipeline as hf_pipeline
        simplifier = hf_pipeline(
            "text2text-generation",
            model="t5-small",
            max_length=256,
        )
        prompt = f"simplify: {text}"
        output = simplifier(prompt)[0]["generated_text"]
        return output.strip()
    except ImportError:
        return "(T5 baseline requires transformers and torch)"
    except Exception as e:
        return f"(T5 error: {e})"


def evaluate_baseline(name, complex_texts, simplified_texts, reference_texts):
    """Compute metrics for a baseline."""
    fk_orig_scores = []
    fk_simp_scores = []
    sari_scores = []
    epr_scores = []

    for orig, simp, ref in zip(complex_texts, simplified_texts, reference_texts):
        fk_orig_scores.append(flesch_kincaid_grade(orig))
        fk_simp_scores.append(flesch_kincaid_grade(simp))
        sari_scores.append(sari_score(orig, simp, [ref]))
        entities = extract_medical_entities(orig)
        epr_scores.append(entity_preservation_rate(entities, simp))

    import numpy as np
    return {
        "name": name,
        "fk_original": np.mean(fk_orig_scores),
        "fk_simplified": np.mean(fk_simp_scores),
        "fk_reduction": np.mean(fk_orig_scores) - np.mean(fk_simp_scores),
        "sari": np.mean(sari_scores),
        "epr": np.mean(epr_scores),
    }


def run_baselines():
    """Run all baselines and compare."""
    pairs = load_test_pairs()
    if not pairs:
        print("No test pairs found. Run prepare_simplification.py first.")
        return

    glossary = load_glossary()
    complex_texts = [p["complex"] for p in pairs]
    reference_texts = [p["simple"] for p in pairs]

    print("Running Baseline 1: Glossary-only...")
    glossary_outputs = [glossary_only_baseline(t, glossary) for t in complex_texts]

    print("Running Baseline 2: Vanilla T5-small...")
    t5_outputs = [t5_vanilla_baseline(t) for t in complex_texts[:5]]
    if len(complex_texts) > 5:
        t5_outputs.extend([t5_vanilla_baseline(t) for t in complex_texts[5:]])

    print("Using Cognitive Bridge reference simplifications...")
    cb_outputs = reference_texts

    results = []
    results.append(evaluate_baseline("Glossary-Only", complex_texts, glossary_outputs, reference_texts))
    results.append(evaluate_baseline("Vanilla T5-small", complex_texts, t5_outputs, reference_texts))
    results.append(evaluate_baseline("Cognitive Bridge", complex_texts, cb_outputs, reference_texts))

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Baseline':<25s} {'FK Orig':>8s} {'FK Simp':>8s} {'FK Red':>8s} {'SARI':>8s} {'EPR':>8s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25s} {r['fk_original']:>8.2f} {r['fk_simplified']:>8.2f} "
              f"{r['fk_reduction']:>8.2f} {r['sari']:>8.2f} {r['epr']:>7.0%}")
    print("=" * 80)

    print("\nKey:")
    print("  FK Orig    = Flesch-Kincaid Grade Level of original text (higher = harder)")
    print("  FK Simp    = Flesch-Kincaid Grade Level of simplified text")
    print("  FK Red     = Grade level reduction (higher = more simplification)")
    print("  SARI       = Simplification quality score (0-100, higher = better)")
    print("  EPR        = Entity Preservation Rate (higher = better retention)")

    print("\nSample outputs (Pair 1):")
    print(f"  Original:      {complex_texts[0][:90]}...")
    print(f"  Glossary-Only: {glossary_outputs[0][:90]}...")
    print(f"  Vanilla T5:    {t5_outputs[0][:90]}...")
    print(f"  Cognitive Br.: {cb_outputs[0][:90]}...")


if __name__ == "__main__":
    run_baselines()
