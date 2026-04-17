#!/usr/bin/env python3
"""
Quick sanity-check for candidate simplification datasets.

Usage (recommended, matches SLURM env paths):
  module purge
  module load GCCcore/13.3.0
  module load Python/3.12.3
  source /scratch/user/vinaysingh/cb-venv/bin/activate
  export HF_HOME=/scratch/user/vinaysingh/.cache/huggingface
  export PIP_CACHE_DIR=/scratch/user/vinaysingh/PIP_CACHE
  export NLTK_DATA=/scratch/user/vinaysingh/Cognitive-Bridge/.nltk_data
  python scripts/check_datasets.py

What it does:
- For each dataset, prints splits, columns, and N sample rows (truncated).
- Runs simple heuristics to estimate "bad target" rate (review/meta-text, etc.).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset
from huggingface_hub import HfApi


DATASETS = [
    "GuyDor007/medisimplifier-dataset",
    "liliya-makhmutova/medical_texts_simplification",
    "starmpcc/Asclepius-Synthetic-Clinical-Notes",
]

SAMPLES_PER_SPLIT = 6
MAX_CHARS = 260

BAD_TARGET_PATTERNS = [
    r"\bthis (is an update|review)\b",
    r"\bupdate of a previous\b",
    r"\bmedical literature\b",
    r"\bevidence (is|was) current to\b",
    r"\bsearch(es)? (are|were) (up[- ]to[- ]date|current)\b",
    r"\bwe (found|searched|included|identified)\b",
    r"\bsystematic review\b",
    r"\bcochrane\b",
    r"\bplain language summary\b",
    r"\brandomi[sz]ed\b",
    r"\bparticipants?\b",
    r"\btrials?\b",
    r"\bstudies?\b",
]


def trunc(x: Any, n: int = MAX_CHARS) -> str:
    s = str(x)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def is_bad_target(text: str) -> bool:
    t = text.lower()
    for pat in BAD_TARGET_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def list_splits(dsid: str) -> List[str]:
    # Works for most datasets; falls back to builder dict.
    builder = load_dataset(dsid, streaming=True)
    if isinstance(builder, dict):
        return list(builder.keys())
    # Sometimes returns an IterableDataset directly (rare)
    return ["train"]


def peek_streaming(dsid: str, split: str) -> Optional[Dict[str, Any]]:
    try:
        ds = load_dataset(dsid, split=split, streaming=True)
        return next(iter(ds))
    except Exception:
        try:
            builder = load_dataset(dsid, streaming=True)
            if isinstance(builder, dict) and split in builder:
                return next(iter(builder[split]))
        except Exception:
            return None
    return None


def try_get_text_pair_fields(columns: List[str]) -> Optional[Tuple[str, str]]:
    """
    Heuristic mapping to (source_field, target_field) for simplification.
    """
    cols = [c.lower() for c in columns]

    # Common instruction datasets
    for a, b in [
        ("input", "output"),
        ("source", "target"),
        ("complex", "simple"),
        ("text", "simplified_text"),
        ("original", "simplified"),
        ("original_text", "simplified_text"),
    ]:
        if a in cols and b in cols:
            return (columns[cols.index(a)], columns[cols.index(b)])

    # liliya dataset likely has only label + maybe complex/simple in nested structure; handled in per-dataset notes.
    return None


def estimate_bad_rate(dsid: str, split: str, target_field: str, max_rows: int = 400) -> Tuple[int, int]:
    """
    Streams up to max_rows and counts how many targets match bad meta-text patterns.
    """
    bad = 0
    total = 0
    ds = load_dataset(dsid, split=split, streaming=True)
    for row in ds:
        t = row.get(target_field)
        if t is None:
            continue
        total += 1
        if isinstance(t, dict):
            # if target is nested, stringify
            t = str(t)
        if is_bad_target(str(t)):
            bad += 1
        if total >= max_rows:
            break
    return bad, total


def print_env_paths():
    print("=== ENV ===")
    for k in ["HF_HOME", "PIP_CACHE_DIR", "NLTK_DATA"]:
        print(f"{k}={os.environ.get(k)}")
    print()


def main() -> int:
    print_env_paths()
    api = HfApi()

    for dsid in DATASETS:
        print("=" * 88)
        print(f"DATASET: {dsid}")
        print("-" * 88)

        # HF metadata
        try:
            info = api.dataset_info(dsid)
            print(f"downloads={info.downloads}  likes={info.likes}")
            license_val = info.cardData.get("license") if info.cardData else None
            print(f"license={license_val}")
        except Exception as e:
            print(f"(could not fetch dataset_info) {e}")

        # Splits + columns
        try:
            splits = list_splits(dsid)
            print(f"splits={splits}")
        except Exception as e:
            print(f"(could not list splits) {e}")
            splits = ["train"]

        # Peek first row to get columns
        first_row = None
        first_split_used = None
        for sp in splits:
            first_row = peek_streaming(dsid, sp)
            if first_row is not None:
                first_split_used = sp
                break

        if first_row is None:
            print("Could not load any split.")
            print()
            continue

        columns = list(first_row.keys())
        print(f"peek_split={first_split_used}")
        print(f"columns={columns}")

        pair_fields = try_get_text_pair_fields(columns)
        if pair_fields:
            src_f, tgt_f = pair_fields
            print(f"detected_pair_fields: source='{src_f}' target='{tgt_f}'")
        else:
            print("detected_pair_fields: (none)")

        print()

        # Print samples
        for sp in splits[:3]:
            try:
                ds = load_dataset(dsid, split=sp, streaming=True)
            except Exception as e:
                print(f"[{sp}] load error: {e}")
                continue

            print(f"--- SPLIT: {sp} (showing {SAMPLES_PER_SPLIT} samples) ---")
            it = iter(ds)
            for i in range(SAMPLES_PER_SPLIT):
                try:
                    row = next(it)
                except StopIteration:
                    break
                print(f"[{sp} #{i}]")
                # show important fields first if detected
                if pair_fields:
                    src_f, tgt_f = pair_fields
                    if src_f in row:
                        print(f"  {src_f}: {trunc(row.get(src_f))}")
                    if tgt_f in row:
                        print(f"  {tgt_f}: {trunc(row.get(tgt_f))}")
                # then show the rest (truncated), skipping images/big blobs
                for k, v in row.items():
                    kl = k.lower()
                    if kl in ("image", "images"):
                        continue
                    if pair_fields and k in pair_fields:
                        continue
                    print(f"  {k}: {trunc(v)}")
                print()

            # Estimate bad target rate if we found a target field
            if pair_fields:
                _, tgt_f = pair_fields
                try:
                    bad, total = estimate_bad_rate(dsid, sp, tgt_f, max_rows=400)
                    if total:
                        print(f"[{sp}] bad_target_rate (first {total} rows): {bad}/{total} ({100*bad/total:.1f}%)")
                except Exception as e:
                    print(f"[{sp}] bad_target_rate error: {e}")
            print()

        # Special note for Asclepius: it’s source-only, not paired
        if "Asclepius-Synthetic-Clinical-Notes" in dsid:
            print("NOTE: Asclepius is a SOURCE corpus (not paired simplification).")
            print("      Use it to generate targets via an LLM or rules, then fine-tune.")
            print()

    print("=" * 88)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())