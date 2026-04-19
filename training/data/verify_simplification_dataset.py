"""
Verify a saved simplification DatasetDict and check whether its content is
suitable for training the T5 simplification model.

Checks performed
----------------
1.  Schema — expected columns present in every split.
2.  Size — minimum row counts per split.
3.  Character-length distributions — avg / p50 / p90 / p99 / max for both
    input and target, flagged against configurable thresholds.
4.  Token-length distributions — tokenizes a sample and reports counts;
    flags examples that would be truncated at MAX_INPUT_LENGTH /
    MAX_TARGET_LENGTH.
5.  Prefix presence — every input must start with the training prefix.
6.  Simplification signal — similarity between input (minus prefix) and
    target must be in a healthy range (not identical = no simplification,
    not completely different = unreliable pair).
7.  Length ratio — target / input character ratio should stay in a
    reasonable band (not a summary, not an expansion).
8.  Qualitative samples — prints random (input, target) pairs so you can
    eyeball quality.

Usage
-----
  python training/data/verify_simplification_dataset.py \\
      --path training/data/simplification_pairs_medisimplifier \\
      --tokenizer google/flan-t5-small \\
      --max_input_tokens 512 \\
      --max_target_tokens 256 \\
      --samples 5
"""

from __future__ import annotations

import argparse
import difflib
import os
import random
import sys
from typing import List

# ── optional tqdm ────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):   # type: ignore[misc]
        return it

from datasets import load_from_disk

# ── thresholds (all overridable via CLI) ─────────────────────────────────────
DEFAULT_MAX_INPUT_TOKENS  = 512
DEFAULT_MAX_TARGET_TOKENS = 256
DEFAULT_MIN_INPUT_CHARS   = 40
DEFAULT_MAX_INPUT_CHARS   = 1800
DEFAULT_MIN_TARGET_CHARS  = 30
# Similarity bounds: if (input_text minus prefix) is too similar to target the
# pair is a copy; if too dissimilar it may be misaligned.
DEFAULT_SIM_WARN_HIGH     = 0.92   # above → likely boilerplate copy
DEFAULT_SIM_WARN_LOW      = 0.05   # below → likely misaligned pair
# Target/input char-length ratio bounds
DEFAULT_RATIO_MIN         = 0.4    # target much shorter → probably a summary, not simplification
DEFAULT_RATIO_MAX         = 2.5    # target much longer  → suspicious expansion

PREFIX = "Simplify this medical text for a patient: "

FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
OK   = "\033[92m OK \033[0m"


def _tag(condition: bool, warn: bool = False) -> str:
    if condition:
        return OK
    return WARN if warn else FAIL


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def _check_tokenizer(tokenizer_name: str):
    """Load tokenizer, return None (with a warning) if unavailable."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"  Tokenizer loaded: {tokenizer_name}")
        return tok
    except Exception as e:
        print(f"  [{WARN}] Could not load tokenizer '{tokenizer_name}': {e}")
        print("         Token-length checks will be skipped.")
        return None


def check_split(
    split_name: str,
    d,
    tokenizer,
    args: argparse.Namespace,
    rng: random.Random,
) -> int:
    """Run all checks on one split. Returns number of FAIL conditions."""
    failures = 0
    n = len(d)

    print(f"\n{'='*60}")
    print(f"  SPLIT: {split_name}  ({n} examples)")
    print(f"{'='*60}")

    # ── 1. Schema ────────────────────────────────────────────────────────────
    for col in ("input_text", "target_text"):
        ok = col in d.column_names
        print(f"  [{_tag(ok)}] column '{col}' present")
        if not ok:
            failures += 1
    if "input_text" not in d.column_names or "target_text" not in d.column_names:
        print("  Cannot continue checks — missing columns.")
        return failures

    # ── 2. Size ──────────────────────────────────────────────────────────────
    min_rows = {"train": 1000, "validation": 100, "test": 100}.get(split_name, 50)
    ok = n >= min_rows
    print(f"  [{_tag(ok)}] row count {n} >= {min_rows}")
    if not ok:
        failures += 1

    # ── 3. Character lengths ─────────────────────────────────────────────────
    sample_n = min(n, 2000)
    indices = rng.sample(range(n), k=sample_n)

    in_chars:  List[float] = []
    out_chars: List[float] = []
    sims:      List[float] = []
    ratios:    List[float] = []
    bad_prefix = 0
    bad_sim_high = 0
    bad_sim_low  = 0
    bad_ratio    = 0

    for i in indices:
        ex       = d[int(i)]
        inp_full = ex["input_text"]
        tgt      = ex["target_text"]

        in_chars.append(len(inp_full))
        out_chars.append(len(tgt))

        # Prefix check
        if not inp_full.startswith(PREFIX):
            bad_prefix += 1

        # Similarity (strip the prefix for a fair comparison)
        inp_body = inp_full[len(PREFIX):] if inp_full.startswith(PREFIX) else inp_full
        sim = _similarity(inp_body, tgt)
        sims.append(sim)
        if sim > args.sim_warn_high:
            bad_sim_high += 1
        if sim < args.sim_warn_low:
            bad_sim_low += 1

        # Length ratio
        ratio = len(tgt) / max(len(inp_body), 1)
        ratios.append(ratio)
        if not (args.ratio_min <= ratio <= args.ratio_max):
            bad_ratio += 1

    # Char-length summary
    print(f"\n  Input character lengths (sample={sample_n}):")
    print(f"    avg={sum(in_chars)/len(in_chars):.0f}  "
          f"p50={_percentile(in_chars,50):.0f}  "
          f"p90={_percentile(in_chars,90):.0f}  "
          f"p99={_percentile(in_chars,99):.0f}  "
          f"max={max(in_chars):.0f}")
    ok_min = sum(1 for c in in_chars if c < args.min_input_chars) == 0
    ok_max = sum(1 for c in in_chars if c > args.max_input_chars) == 0
    short_count = sum(1 for c in in_chars if c < args.min_input_chars)
    long_count  = sum(1 for c in in_chars if c > args.max_input_chars)
    print(f"  [{_tag(ok_min, warn=True)}] inputs < {args.min_input_chars} chars: {short_count}/{sample_n}")
    print(f"  [{_tag(ok_max, warn=True)}] inputs > {args.max_input_chars} chars: {long_count}/{sample_n}")

    print(f"\n  Target character lengths (sample={sample_n}):")
    print(f"    avg={sum(out_chars)/len(out_chars):.0f}  "
          f"p50={_percentile(out_chars,50):.0f}  "
          f"p90={_percentile(out_chars,90):.0f}  "
          f"p99={_percentile(out_chars,99):.0f}  "
          f"max={max(out_chars):.0f}")
    short_tgt = sum(1 for c in out_chars if c < args.min_target_chars)
    print(f"  [{_tag(short_tgt == 0, warn=True)}] targets < {args.min_target_chars} chars: {short_tgt}/{sample_n}")

    # ── 4. Token lengths ─────────────────────────────────────────────────────
    if tokenizer is not None:
        print(f"\n  Token lengths (sample={min(sample_n, 500)}, "
              f"max_input={args.max_input_tokens}, max_target={args.max_target_tokens}):")

        tok_indices = indices[:500]
        in_tokens:  List[int] = []
        out_tokens: List[int] = []

        batch_size = 64
        inp_texts = [d[int(i)]["input_text"]  for i in tok_indices]
        tgt_texts = [d[int(i)]["target_text"] for i in tok_indices]

        for start in range(0, len(inp_texts), batch_size):
            enc = tokenizer(
                inp_texts[start:start+batch_size],
                truncation=False, padding=False,
            )
            in_tokens.extend(len(ids) for ids in enc["input_ids"])

        for start in range(0, len(tgt_texts), batch_size):
            enc = tokenizer(
                tgt_texts[start:start+batch_size],
                truncation=False, padding=False,
            )
            out_tokens.extend(len(ids) for ids in enc["input_ids"])

        trunc_in  = sum(1 for t in in_tokens  if t > args.max_input_tokens)
        trunc_out = sum(1 for t in out_tokens if t > args.max_target_tokens)
        pct_in  = 100 * trunc_in  / len(in_tokens)
        pct_out = 100 * trunc_out / len(out_tokens)

        print(f"    input tokens:  avg={sum(in_tokens)/len(in_tokens):.0f}  "
              f"p50={_percentile(in_tokens,50):.0f}  "
              f"p90={_percentile(in_tokens,90):.0f}  "
              f"p99={_percentile(in_tokens,99):.0f}  "
              f"max={max(in_tokens)}")
        print(f"    target tokens: avg={sum(out_tokens)/len(out_tokens):.0f}  "
              f"p50={_percentile(out_tokens,50):.0f}  "
              f"p90={_percentile(out_tokens,90):.0f}  "
              f"p99={_percentile(out_tokens,99):.0f}  "
              f"max={max(out_tokens)}")

        ok_in  = pct_in  < 5.0
        ok_out = pct_out < 5.0
        print(f"  [{_tag(ok_in,  warn=True)}] inputs  truncated at {args.max_input_tokens}t:  "
              f"{trunc_in}/{len(in_tokens)} ({pct_in:.1f}%)")
        print(f"  [{_tag(ok_out, warn=True)}] targets truncated at {args.max_target_tokens}t: "
              f"{trunc_out}/{len(out_tokens)} ({pct_out:.1f}%)")

        if not ok_in:
            failures += 1
            suggested = int(_percentile(in_tokens, 99)) + 16
            print(f"           → consider raising MAX_INPUT_LENGTH to ~{suggested}")
        if not ok_out:
            failures += 1
            suggested = int(_percentile(out_tokens, 99)) + 16
            print(f"           → consider raising MAX_TARGET_LENGTH to ~{suggested}")

    # ── 5. Prefix presence ───────────────────────────────────────────────────
    ok = bad_prefix == 0
    print(f"\n  [{_tag(ok)}] inputs missing prefix: {bad_prefix}/{sample_n}")
    if not ok:
        failures += 1

    # ── 6. Simplification signal ─────────────────────────────────────────────
    avg_sim = sum(sims) / len(sims)
    ok_high = bad_sim_high / sample_n < 0.10
    ok_low  = bad_sim_low  / sample_n < 0.10
    print(f"\n  Similarity (input_body vs target):")
    print(f"    avg={avg_sim:.3f}  "
          f"p10={_percentile(sims,10):.3f}  "
          f"p50={_percentile(sims,50):.3f}  "
          f"p90={_percentile(sims,90):.3f}")
    print(f"  [{_tag(ok_high, warn=True)}] pairs with sim > {args.sim_warn_high} (copies):    "
          f"{bad_sim_high}/{sample_n} ({100*bad_sim_high/sample_n:.1f}%)")
    print(f"  [{_tag(ok_low,  warn=True)}] pairs with sim < {args.sim_warn_low} (misaligned): "
          f"{bad_sim_low}/{sample_n}  ({100*bad_sim_low/sample_n:.1f}%)")

    # ── 7. Length ratio ──────────────────────────────────────────────────────
    avg_ratio = sum(ratios) / len(ratios)
    ok_ratio  = bad_ratio / sample_n < 0.05
    print(f"\n  Target/input length ratio:")
    print(f"    avg={avg_ratio:.2f}  "
          f"p10={_percentile(ratios,10):.2f}  "
          f"p50={_percentile(ratios,50):.2f}  "
          f"p90={_percentile(ratios,90):.2f}")
    print(f"  [{_tag(ok_ratio, warn=True)}] pairs outside ratio [{args.ratio_min},{args.ratio_max}]: "
          f"{bad_ratio}/{sample_n} ({100*bad_ratio/sample_n:.1f}%)")

    # ── 8. Qualitative samples ───────────────────────────────────────────────
    print(f"\n  --- {args.samples} random samples ---")
    for idx in rng.sample(range(n), k=min(args.samples, n)):
        ex  = d[int(idx)]
        inp = ex["input_text"]
        tgt = ex["target_text"]
        inp_body = inp[len(PREFIX):] if inp.startswith(PREFIX) else inp
        sim = _similarity(inp_body, tgt)
        ratio = len(tgt) / max(len(inp_body), 1)
        print(f"\n  [idx={idx}  sim={sim:.2f}  ratio={ratio:.2f}"
              f"  in={len(inp)}chars  out={len(tgt)}chars]")
        print(f"  INPUT : {inp[:240].replace(chr(10), ' ')}"
              + ("…" if len(inp) > 240 else ""))
        print(f"  TARGET: {tgt[:240].replace(chr(10), ' ')}"
              + ("…" if len(tgt) > 240 else ""))

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify a simplification DatasetDict for T5 training readiness."
    )
    ap.add_argument("--path", required=True,
                    help="Path to DatasetDict saved with save_to_disk()")
    ap.add_argument("--tokenizer", default="google/flan-t5-small",
                    help="HuggingFace tokenizer name or local path (default: google/flan-t5-small)")
    ap.add_argument("--max_input_tokens",  type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    ap.add_argument("--max_target_tokens", type=int, default=DEFAULT_MAX_TARGET_TOKENS)
    ap.add_argument("--min_input_chars",   type=int, default=DEFAULT_MIN_INPUT_CHARS)
    ap.add_argument("--max_input_chars",   type=int, default=DEFAULT_MAX_INPUT_CHARS)
    ap.add_argument("--min_target_chars",  type=int, default=DEFAULT_MIN_TARGET_CHARS)
    ap.add_argument("--sim_warn_high",     type=float, default=DEFAULT_SIM_WARN_HIGH)
    ap.add_argument("--sim_warn_low",      type=float, default=DEFAULT_SIM_WARN_LOW)
    ap.add_argument("--ratio_min",         type=float, default=DEFAULT_RATIO_MIN)
    ap.add_argument("--ratio_max",         type=float, default=DEFAULT_RATIO_MAX)
    ap.add_argument("--samples",           type=int, default=5,
                    help="Number of qualitative samples to print per split")
    ap.add_argument("--seed",              type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"ERROR: path not found: {args.path}")

    ds = load_from_disk(args.path)
    print(f"Loaded: {args.path}")
    print(f"Splits: {list(ds.keys())}")
    print(f"\nConfig:")
    print(f"  tokenizer:        {args.tokenizer}")
    print(f"  max_input_tokens: {args.max_input_tokens}")
    print(f"  max_target_tokens:{args.max_target_tokens}")
    print(f"  max_input_chars:  {args.max_input_chars}")
    print(f"  sim range:        [{args.sim_warn_low}, {args.sim_warn_high}]")
    print(f"  ratio range:      [{args.ratio_min}, {args.ratio_max}]")

    tokenizer = _check_tokenizer(args.tokenizer)
    rng = random.Random(args.seed)

    total_failures = 0
    for split in ds.keys():
        total_failures += check_split(split, ds[split], tokenizer, args, rng)

    print(f"\n{'='*60}")
    if total_failures == 0:
        print(f"  [{OK}] All checks passed — dataset looks good for training.")
    else:
        print(f"  [{FAIL}] {total_failures} check(s) failed — review output above.")
    print(f"{'='*60}\n")

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
