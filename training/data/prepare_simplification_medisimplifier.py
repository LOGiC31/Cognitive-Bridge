"""
Prepare parallel medical simplification training data from:
  - GuyDor007/medisimplifier-dataset

Instead of using full discharge summaries as single training examples, this
script splits each document into sections and extracts only the sections where
actual medical-jargon simplification occurs. Boilerplate (patient headers,
identical lines) is discarded.

Why section-level?
  - Full discharge summaries average ~2000 chars — too long for T5-small's
    512-token input limit without heavy truncation.
  - Most of each document is unchanged boilerplate (patient name, dates, etc.)
    that teaches the model to copy, not simplify.
  - The extension calls the model on short clinical passages, so training on
    section-length text (~200-400 chars) better matches inference distribution.

Usage:
  python training/data/prepare_simplification_medisimplifier.py

Output:
  training/data/simplification_pairs_medisimplifier/ — HuggingFace DatasetDict on disk
"""

from __future__ import annotations

import difflib
import os
import random
import re
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ID = "GuyDor007/medisimplifier-dataset"

# Must match extension inference prefix (and finetune_t5_medisimplifier.py).
PREFIX = "Simplify this medical text for a patient: "

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "simplification_pairs_medisimplifier",
)

# --- Section extraction thresholds ---
# Sections with similarity above this are considered unchanged (boilerplate).
SIM_MAX = 0.88
# Sections with similarity below this are too different to be a reliable parallel pair
# (e.g. the output restructured or merged multiple input sections).
SIM_MIN = 0.05
# Minimum character length for extracted section body (input side, after stripping prefix).
MIN_SECTION_CHARS = 60
# Maximum character length for extracted section body — keeps examples within
# T5-small's ~512-token budget (at ~4 chars/token, 450 chars ≈ 112 tokens, leaving
# headroom for the prefix).
MAX_SECTION_CHARS = 1800

# Regex that matches a line that looks like a section header:
# - Starts with a capital letter
# - Ends with a colon (optionally followed by whitespace)
# - Short enough to be a heading, not a sentence
_HEADER_RE = re.compile(r'^[A-Z][A-Za-z0-9\s/\-]{0,60}:\s*$')


def _clean_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split text into (header, body) pairs.
    Lines that match _HEADER_RE start a new section.
    Text before the first header is collected under '__preamble__'.
    """
    sections: List[Tuple[str, str]] = []
    current_header = "__preamble__"
    current_body: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if _HEADER_RE.match(stripped):
            body = "\n".join(current_body).strip()
            if body:
                sections.append((current_header, body))
            current_header = stripped
            current_body = []
        else:
            current_body.append(line)

    body = "\n".join(current_body).strip()
    if body:
        sections.append((current_header, body))

    return sections


def _extract_section_pairs(
    inp: str, out: str
) -> List[Tuple[str, str]]:
    """
    Align sections between input and output by header name, then return only
    the (inp_body, out_body) pairs where meaningful simplification occurred.
    """
    inp_sections = _split_sections(inp)
    out_by_header: Dict[str, str] = {}
    for h, b in _split_sections(out):
        out_by_header[h] = b

    pairs: List[Tuple[str, str]] = []
    for header, inp_body in inp_sections:
        # Skip preamble — usually just "Discharge Summary / Patient Name: [Redacted]" boilerplate.
        if header == "__preamble__":
            continue

        out_body = out_by_header.get(header, "")
        if not out_body:
            continue

        # Skip if the section body is too short to be meaningful.
        if len(inp_body) < MIN_SECTION_CHARS or len(out_body) < MIN_SECTION_CHARS:
            continue

        # Skip if the section is too long (would exceed token budget even after truncation).
        if len(inp_body) > MAX_SECTION_CHARS:
            continue

        sim = _similarity(inp_body, out_body)

        # Drop boilerplate (unchanged) and completely unrelated sections.
        if sim > SIM_MAX or sim < SIM_MIN:
            continue

        pairs.append((inp_body.strip(), out_body.strip()))

    return pairs


def _make_pairs(
    rows: Iterable[Dict],
) -> Tuple[List[str], List[str], Dict[str, int]]:
    inputs: List[str] = []
    targets: List[str] = []
    stats: Dict[str, int] = {
        "rows_seen": 0,
        "rows_no_pairs": 0,
        "sections_kept": 0,
        "sections_dropped_boilerplate": 0,
        "sections_dropped_short": 0,
        "sections_dropped_toolong": 0,
        "sections_dropped_unrelated": 0,
    }

    for row in rows:
        stats["rows_seen"] += 1

        inp = _clean_text(row.get("input", ""))
        out = _clean_text(row.get("output", ""))

        if not inp or not out:
            stats["rows_no_pairs"] += 1
            continue

        pairs = _extract_section_pairs(inp, out)

        if not pairs:
            stats["rows_no_pairs"] += 1
            continue

        for inp_body, out_body in pairs:
            inputs.append(PREFIX + inp_body)
            targets.append(out_body)
            stats["sections_kept"] += 1

    return inputs, targets, stats


def _basic_report(ds: DatasetDict) -> None:
    print("\n=== Dataset report ===")
    for split in ds.keys():
        d = ds[split]
        n = min(len(d), 1000)
        in_lens = [len(x) for x in d["input_text"][:n]]
        out_lens = [len(x) for x in d["target_text"][:n]]
        print(f"\n--- {split}: {len(d)} examples ---")
        print(f"  input chars:  avg={sum(in_lens)/len(in_lens):.0f}  "
              f"min={min(in_lens)}  max={max(in_lens)}")
        print(f"  target chars: avg={sum(out_lens)/len(out_lens):.0f}  "
              f"min={min(out_lens)}  max={max(out_lens)}")

    rng = random.Random(7)
    for split in ds.keys():
        d = ds[split]
        if not d:
            continue
        print(f"\n--- Samples: {split} ---")
        for idx in rng.sample(range(len(d)), k=min(3, len(d))):
            ex = d[int(idx)]
            print("INPUT :", ex["input_text"][:320].replace("\n", " ") +
                  ("…" if len(ex["input_text"]) > 320 else ""))
            print("TARGET:", ex["target_text"][:320].replace("\n", " ") +
                  ("…" if len(ex["target_text"]) > 320 else ""))
            print("---")


def main() -> None:
    if os.path.isfile(os.path.join(OUTPUT_DIR, "dataset_dict.json")):
        print(f"Dataset already exists at {OUTPUT_DIR}, skipping preparation.")
        print("Delete the directory to re-prepare:\n  rm -rf " + OUTPUT_DIR)
        return

    print(f"Loading {DATASET_ID} …")
    offline = os.environ.get("HF_DATASETS_OFFLINE") == "1"
    try:
        raw = load_dataset(DATASET_ID, local_files_only=offline)
    except ValueError as e:
        if offline and "Couldn't find cache" in str(e):
            print("Offline cache config mismatch; retrying without local_files_only…")
            raw = load_dataset(DATASET_ID)
        else:
            raise

    splits = {}
    for split in ["train", "validation", "test"]:
        if split not in raw:
            continue
        inputs, targets, stats = _make_pairs(raw[split])
        splits[split] = Dataset.from_dict({"input_text": inputs, "target_text": targets})
        print(f"\n[{split}]")
        print(f"  rows seen:          {stats['rows_seen']}")
        print(f"  rows with no pairs: {stats['rows_no_pairs']}")
        print(f"  section pairs kept: {stats['sections_kept']}")

    ds = DatasetDict(splits)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds.save_to_disk(OUTPUT_DIR)
    print(f"\nDataset saved to {OUTPUT_DIR}")
    _basic_report(ds)


if __name__ == "__main__":
    main()

