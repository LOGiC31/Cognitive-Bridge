"""
Evaluation metrics for the Cognitive Bridge simplification system.

Computes:
  - Flesch-Kincaid Grade Level (readability)
  - SARI Score (simplification quality)
  - Entity Preservation Rate (EPR) — clinical entity retention

Usage:
    python evaluation/evaluate.py

Reads test data from training/data/simplification_pairs/pairs.json
"""

import json
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAIRS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "data", "simplification_pairs", "pairs.json"
)


def count_syllables(word):
    """Estimate syllable count for a word."""
    word = word.lower().strip()
    if len(word) <= 2:
        return 1

    word = re.sub(r'(?:es|ed|e)$', '', word) or word
    vowels = re.findall(r'[aeiouy]+', word)
    count = max(1, len(vowels))
    return count


def flesch_kincaid_grade(text):
    """
    Compute Flesch-Kincaid Grade Level.
    Lower = easier to read. Target: 8th grade or below.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0

    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(w) for w in words)

    grade = (
        0.39 * (num_words / num_sentences)
        + 11.8 * (num_syllables / num_words)
        - 15.59
    )
    return max(0, grade)


def sari_score(source, prediction, references):
    """
    Compute SARI (System output Against References and against the Input sentence).
    Measures how well the simplification adds, deletes, and keeps appropriate words.
    Score is between 0 and 100.
    """
    def ngrams(text, n):
        words = text.lower().split()
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

    def ngram_counts(text, n):
        counts = {}
        for ng in ngrams(text, n):
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    sari_scores = []

    for n in range(1, 5):  # 1-grams to 4-grams
        src_ng = set(ngrams(source, n))
        pred_ng = ngram_counts(prediction, n)
        ref_ng = ngram_counts(references[0] if references else "", n)
        ref_set = set(ref_ng.keys())

        # Keep: n-grams in both source and prediction that are also in reference
        keep_correct = sum(1 for ng in pred_ng if ng in src_ng and ng in ref_set)
        keep_total = sum(1 for ng in src_ng if ng in ref_set)
        keep_precision = keep_correct / max(sum(pred_ng.values()), 1)
        keep_recall = keep_correct / max(keep_total, 1)
        keep_f1 = (2 * keep_precision * keep_recall / max(keep_precision + keep_recall, 1e-8))

        # Delete: n-grams in source but not in prediction, that are also not in reference
        del_correct = sum(1 for ng in src_ng if ng not in pred_ng and ng not in ref_set)
        del_total = sum(1 for ng in src_ng if ng not in ref_set)
        del_precision = del_correct / max(len([ng for ng in pred_ng if ng not in src_ng]) + del_correct, 1)
        del_recall = del_correct / max(del_total, 1)

        # Add: n-grams in prediction but not in source, that are in reference
        add_correct = sum(1 for ng in pred_ng if ng not in src_ng and ng in ref_set)
        add_total = sum(1 for ng in ref_set if ng not in src_ng)
        add_precision = add_correct / max(sum(1 for ng in pred_ng if ng not in src_ng), 1)
        add_recall = add_correct / max(add_total, 1)

        sari_n = (keep_f1 + del_precision + add_precision) / 3
        sari_scores.append(sari_n)

    return np.mean(sari_scores) * 100


def entity_preservation_rate(original_entities, simplified_text):
    """
    Compute Entity Preservation Rate (EPR).
    Checks what fraction of original medical entities are still present
    (or explained) in the simplified text.
    """
    if not original_entities:
        return 1.0

    simplified_lower = simplified_text.lower()
    preserved = 0

    for entity in original_entities:
        entity_lower = entity.lower()
        if entity_lower in simplified_lower:
            preserved += 1
        else:
            words = entity_lower.split()
            if any(w in simplified_lower for w in words if len(w) > 3):
                preserved += 0.5

    return preserved / len(original_entities)


def extract_medical_entities(text):
    """Simple regex-based entity extraction for evaluation purposes."""
    patterns = [
        r'\b\w+(?:itis|ectomy|emia|osis|pathy|algia|plasty|scopy)\b',
        r'\b(?:hypertension|diabetes|carcinoma|lymphoma|pneumonia|anemia)\b',
        r'\b(?:hemoglobin|creatinine|bilirubin|troponin|albumin|platelet)\b',
        r'\b(?:metformin|lisinopril|atorvastatin|warfarin|aspirin|clopidogrel)\b',
        r'\b(?:myocardial|infarction|fibrillation|thrombosis|embolism|stenosis)\b',
    ]

    entities = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities.update(m.lower() for m in matches)

    return list(entities)


def evaluate_pair(complex_text, simplified_text):
    """Evaluate a single complex/simplified pair across all metrics."""
    fk_original = flesch_kincaid_grade(complex_text)
    fk_simplified = flesch_kincaid_grade(simplified_text)

    sari = sari_score(complex_text, simplified_text, [simplified_text])

    entities = extract_medical_entities(complex_text)
    epr = entity_preservation_rate(entities, simplified_text)

    return {
        "fk_original": fk_original,
        "fk_simplified": fk_simplified,
        "fk_reduction": fk_original - fk_simplified,
        "sari": sari,
        "epr": epr,
        "entities_found": len(entities),
    }


def run_evaluation():
    """Run evaluation on all test pairs."""
    if os.path.exists(PAIRS_PATH):
        with open(PAIRS_PATH) as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} evaluation pairs from {PAIRS_PATH}")
    else:
        print(f"No pairs file found at {PAIRS_PATH}, using built-in examples.")
        pairs = [
            {
                "complex": "The patient presents with acute myocardial infarction with ST-segment elevation.",
                "simple": "The patient is having a heart attack with a specific pattern on the heart test."
            },
            {
                "complex": "Labs reveal elevated creatinine at 2.8 suggestive of acute kidney injury.",
                "simple": "A blood test shows high creatinine levels, suggesting the kidneys are not working properly."
            },
        ]

    results = []
    for pair in pairs:
        result = evaluate_pair(pair["complex"], pair["simple"])
        results.append(result)

    avg_fk_orig = np.mean([r["fk_original"] for r in results])
    avg_fk_simp = np.mean([r["fk_simplified"] for r in results])
    avg_fk_red = np.mean([r["fk_reduction"] for r in results])
    avg_sari = np.mean([r["sari"] for r in results])
    avg_epr = np.mean([r["epr"] for r in results])

    print("\n" + "=" * 60)
    print("COGNITIVE BRIDGE — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Number of test pairs:          {len(pairs)}")
    print(f"  Avg FK Grade (Original):       {avg_fk_orig:.2f}")
    print(f"  Avg FK Grade (Simplified):     {avg_fk_simp:.2f}")
    print(f"  Avg FK Grade Reduction:        {avg_fk_red:.2f}")
    print(f"  Avg SARI Score:                {avg_sari:.2f}")
    print(f"  Avg Entity Preservation Rate:  {avg_epr:.2%}")
    print("=" * 60)

    print("\nPer-pair breakdown:")
    for i, (pair, result) in enumerate(zip(pairs, results)):
        print(f"\n--- Pair {i+1} ---")
        print(f"  Complex:  {pair['complex'][:80]}...")
        print(f"  Simple:   {pair['simple'][:80]}...")
        print(f"  FK: {result['fk_original']:.1f} -> {result['fk_simplified']:.1f} "
              f"(reduction: {result['fk_reduction']:.1f})")
        print(f"  SARI: {result['sari']:.1f}  |  EPR: {result['epr']:.0%} "
              f"({result['entities_found']} entities)")

    return results


if __name__ == "__main__":
    run_evaluation()
