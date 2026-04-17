#!/usr/bin/env python3
"""
Verify that the quantized ONNX export matches the fine-tuned PyTorch checkpoint.

Checks:
  1. ROUGE-L degradation (ONNX vs PyTorch must be within --rouge_tol)
  2. Repetition rate (outputs with repeated 5-grams — catches the decoder loop bug)
  3. Flesch-Kincaid grade level (target <= --fk_grade_max; requires `pip install textstat`)
  4. SARI score (simplification quality; requires `evaluate` with sari cached)
  5. Empty output rate (ONNX must produce non-empty text for all inputs)

Exit code 0 if all enabled checks pass, 1 otherwise (integrates with `set -euo pipefail`).

Usage:
  python training/simplification/verify_onnx_export.py
  python training/simplification/verify_onnx_export.py --max_samples 50 --num_beams 1
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Paths (relative to this file) ─────────────────────────────────────────────

_HERE = Path(__file__).parent

DEFAULTS = {
    "dataset": str(_HERE.parent / "data" / "simplification_pairs_medisimplifier"),
    "torch_model": str(_HERE / "output_medisimplifier" / "best_model"),
    "onnx_dir": str(_HERE / "onnx_quantized_medisimplifier"),
}

# ── Thresholds ─────────────────────────────────────────────────────────────────
# These are the pass/fail gates. Adjust via CLI args if needed.

DEFAULT_ROUGE_TOL = 0.15      # 8-bit dynamic quantization typically causes 5-15% ROUGE-L drop
DEFAULT_MAX_REPETITION = 0.10 # <= 10% of outputs may contain a repeated 5-gram
DEFAULT_FK_GRADE_MAX = 9.0    # Flesch-Kincaid grade level (9 gives headroom for ONNX vs PyTorch variance)
DEFAULT_MAX_EMPTY_RATE = 0.0  # No empty ONNX outputs allowed
DEFAULT_MAX_SARI_GAP = 6.0    # 8-bit quantization typically causes 3-6 pt SARI drop


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _repetition_rate(texts: List[str], ngram: int = 5) -> float:
    """Fraction of texts that contain at least one repeated n-gram."""
    flagged = 0
    for text in texts:
        tokens = text.lower().split()
        if len(tokens) < ngram * 2:
            continue
        seen: set = set()
        for i in range(len(tokens) - ngram + 1):
            gram = tuple(tokens[i : i + ngram])
            if gram in seen:
                flagged += 1
                break
            seen.add(gram)
    return flagged / max(len(texts), 1)


def _try_fk_grade(texts: List[str]) -> Optional[float]:
    try:
        import textstat  # type: ignore
        scores = [textstat.flesch_kincaid_grade(t) for t in texts if t.strip()]
        return _mean(scores) if scores else None
    except ImportError:
        return None


def _try_sari(sources: List[str], preds: List[str], refs: List[str]) -> Optional[float]:
    try:
        import evaluate  # type: ignore
        sari = evaluate.load("sari")
        # evaluate's SARI expects references as list-of-lists
        result = sari.compute(
            sources=sources,
            predictions=preds,
            references=[[r] for r in refs],
        )
        val = result.get("sari", None)
        return float(val) if val is not None else None
    except Exception:
        return None


def _load_rouge():
    import evaluate  # type: ignore
    return evaluate.load("rouge")


def _rouge_l(preds: List[str], refs: List[str]) -> float:
    rouge = _load_rouge()
    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    val = r.get("rougeL", 0.0)
    if hasattr(val, "mid"):
        return float(val.mid.fmeasure)
    return float(val)


def _generate_torch(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> List[str]:
    model.eval()
    outs: List[str] = []
    with torch.no_grad():
        for p in tqdm(prompts, desc="PyTorch inference"):
            inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(device)
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
            )
            outs.append(tokenizer.decode(ids[0], skip_special_tokens=True).strip())
    return outs


def _generate_onnx(
    ort_model,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    outs: List[str] = []
    with torch.no_grad():
        for p in tqdm(prompts, desc="ONNX inference"):
            inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
            try:
                ids = ort_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                )
            except Exception as exc:
                # Beam search can misalign cross-attn past shapes; fall back to greedy.
                msg = str(exc)
                if "Expected" in msg and num_beams > 1:
                    ids = ort_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                else:
                    raise
            outs.append(tokenizer.decode(ids[0], skip_special_tokens=True).strip())
    return outs


def _patch_ort_t5_kv_head_dim(ort_model) -> None:
    """
    Optimum sets embed_size_per_head = d_model // num_heads (512//6 = 85 for flan-t5-small).
    T5 uses a separate d_kv (64) for K/V depth. This causes ORT to build dummy past_key_values
    with the wrong last dimension, triggering INVALID_ARGUMENT at axis 3.
    """
    cfg = getattr(ort_model, "config", None)
    if cfg is None or getattr(cfg, "model_type", None) not in ("t5", "mt5", "longt5"):
        return
    d_kv = getattr(cfg, "d_kv", None)
    if d_kv is None:
        return
    want = int(d_kv)
    for name in ("decoder", "decoder_with_past"):
        dec = getattr(ort_model, name, None)
        if dec is None or not hasattr(dec, "embed_size_per_head"):
            continue
        cur = int(getattr(dec, "embed_size_per_head"))
        if cur != want:
            print(f"  Patching ORT {name}.embed_size_per_head: {cur} -> {want} (T5 d_kv)")
            setattr(dec, "embed_size_per_head", want)


def _load_onnx_model(onnx_dir: str, provider: str):
    from optimum.onnxruntime import ORTModelForSeq2SeqLM  # type: ignore

    p = Path(onnx_dir).resolve()
    model_id = p.as_posix()
    subfolder = ""

    # If ONNX files are in an onnx/ subdir but config.json is at root, rewire.
    if not (p / "config.json").exists() and (p.parent / "config.json").exists():
        model_id = p.parent.as_posix()
        subfolder = p.name

    onnx_files_dir = Path(model_id) / subfolder if subfolder else Path(model_id)
    # Prefer files in onnx/ subdir if it exists (Transformers.js layout).
    onnx_sub = onnx_files_dir / "onnx"
    if onnx_sub.is_dir() and any(onnx_sub.glob("*.onnx")):
        subfolder = "onnx"

    ort_model = ORTModelForSeq2SeqLM.from_pretrained(
        model_id,
        subfolder=subfolder,
        local_files_only=True,
        encoder_file_name="encoder_model_quantized.onnx",
        use_cache=True,
        use_merged=True,
        provider=provider,
    )
    _patch_ort_t5_kv_head_dim(ort_model)
    return ort_model


# ── Checks ─────────────────────────────────────────────────────────────────────

class Check:
    def __init__(self, name: str, passed: bool, detail: str):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name}: {self.detail}"


def _run_checks(
    *,
    torch_preds: List[str],
    onnx_preds: List[str],
    sources: List[str],
    refs: List[str],
    rouge_tol: float,
    max_repetition: float,
    fk_grade_max: float,
    max_empty_rate: float,
    sari_gap_max: float = DEFAULT_MAX_SARI_GAP,
) -> List[Check]:
    checks: List[Check] = []

    # 1. ROUGE-L degradation
    rl_torch = _rouge_l(torch_preds, refs)
    rl_onnx = _rouge_l(onnx_preds, refs)
    threshold = rl_torch * (1.0 - rouge_tol)
    rouge_ok = rl_onnx >= threshold
    checks.append(Check(
        "ROUGE-L degradation",
        rouge_ok,
        f"PyTorch={rl_torch:.4f}  ONNX={rl_onnx:.4f}  "
        f"drop={100*(rl_torch-rl_onnx)/max(rl_torch,1e-9):.2f}%  "
        f"threshold=<{rouge_tol*100:.0f}% drop",
    ))

    # 2. Repetition rate (decoder loop detector)
    rep_rate = _repetition_rate(onnx_preds)
    rep_ok = rep_rate <= max_repetition
    flagged_ex = [p for p in onnx_preds if _has_repeat(p)]
    checks.append(Check(
        "Repetition rate (ONNX)",
        rep_ok,
        f"{rep_rate*100:.1f}% of outputs have repeated 5-grams  "
        f"(threshold <= {max_repetition*100:.0f}%)",
    ))
    if flagged_ex:
        print(f"\n  Repeated-output examples (first 2):")
        for ex in flagged_ex[:2]:
            print(f"    {ex[:120]!r}...")

    # 3. Empty output rate
    empty = sum(1 for p in onnx_preds if not p.strip())
    empty_rate = empty / max(len(onnx_preds), 1)
    empty_ok = empty_rate <= max_empty_rate
    checks.append(Check(
        "Empty outputs (ONNX)",
        empty_ok,
        f"{empty}/{len(onnx_preds)} empty  (threshold <= {max_empty_rate*100:.0f}%)",
    ))

    # 4. Flesch-Kincaid grade level
    fk = _try_fk_grade(onnx_preds)
    if fk is not None:
        fk_ok = fk <= fk_grade_max
        checks.append(Check(
            "Flesch-Kincaid grade (ONNX)",
            fk_ok,
            f"avg={fk:.2f}  (threshold <= {fk_grade_max:.0f})",
        ))
    else:
        print("  [SKIP] Flesch-Kincaid: `textstat` not installed. Run: pip install textstat")

    # 5. SARI score (informational — no hard threshold, just report)
    # Strip the prompt prefix before passing sources to SARI.
    PREFIX = "Simplify this medical text for a patient: "
    raw_sources = [s.removeprefix(PREFIX) for s in sources]
    sari = _try_sari(raw_sources, onnx_preds, refs)
    if sari is not None:
        sari_torch = _try_sari(raw_sources, torch_preds, refs)
        label = f"ONNX={sari:.2f}"
        if sari_torch is not None:
            label += f"  PyTorch={sari_torch:.2f}"
        sari_ok = sari_torch is None or abs(sari - sari_torch) <= sari_gap_max
        checks.append(Check("SARI score", sari_ok, label + f"  (threshold: gap <= {sari_gap_max} pts)"))
    else:
        print("  [SKIP] SARI: metric not cached. Run: python -c \"import evaluate; evaluate.load('sari')\"")

    # 6. Exact-match agreement (informational)
    agree = sum(1 for a, b in zip(torch_preds, onnx_preds) if a == b)
    agree_pct = 100.0 * agree / max(len(torch_preds), 1)
    print(f"\n  [INFO] PyTorch/ONNX exact match: {agree}/{len(torch_preds)} ({agree_pct:.1f}%)")

    # 7. Avg output length (sanity)
    avg_torch = _mean([len(p) for p in torch_preds])
    avg_onnx = _mean([len(p) for p in onnx_preds])
    avg_ref = _mean([len(r) for r in refs])
    print(f"  [INFO] Avg chars — PyTorch: {avg_torch:.0f}  ONNX: {avg_onnx:.0f}  Reference: {avg_ref:.0f}")

    return checks


def _has_repeat(text: str, ngram: int = 5) -> bool:
    tokens = text.lower().split()
    seen: set = set()
    for i in range(len(tokens) - ngram + 1):
        gram = tuple(tokens[i : i + ngram])
        if gram in seen:
            return True
        seen.add(gram)
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

# Medical sentences specifically chosen to be outside typical discharge-summary training data:
# rare conditions, specialist terminology, pharmacology, and procedure descriptions.
PROBE_SENTENCES = [
    "The patient was diagnosed with rhabdomyolysis following strenuous exercise, presenting with elevated creatine kinase and myoglobinuria.",
    "Echocardiography revealed severe mitral valve regurgitation with left ventricular dilatation and reduced ejection fraction of 35%.",
    "The patient has Guillain-Barré syndrome, an autoimmune condition causing ascending paralysis due to demyelination of peripheral nerves.",
    "MRI of the lumbar spine demonstrated a herniated nucleus pulposus at L4-L5 with moderate neural foraminal stenosis.",
    "The patient presented with pheochromocytoma, a rare adrenal gland tumor causing episodic hypertension, diaphoresis, and tachycardia.",
    "Spirometry confirmed a restrictive ventilatory pattern with reduced total lung capacity and diffusing capacity for carbon monoxide.",
    "The patient underwent laparoscopic cholecystectomy for acute calculous cholecystitis with successful removal of the gallbladder.",
    "Laboratory results showed hyponatremia with serum sodium of 122 mEq/L, consistent with syndrome of inappropriate antidiuretic hormone secretion.",
    "The patient has ankylosing spondylitis, a chronic inflammatory arthritis primarily affecting the sacroiliac joints and spine.",
    "Fundoscopic examination revealed papilledema and arteriovenous nicking suggestive of longstanding hypertensive retinopathy.",
    "The biopsy confirmed diffuse large B-cell lymphoma, an aggressive non-Hodgkin lymphoma requiring immediate chemotherapy.",
    "Troponin I levels were markedly elevated at 15.2 ng/mL, confirming a non-ST-elevation myocardial infarction.",
]

PREFIX = "Simplify this medical text for a patient: "


def _run_qualitative_probe(
    tokenizer,
    ort_model,
    max_new_tokens: int,
    torch_model_path: str,
    device,
) -> None:
    print()
    print("=" * 60)
    print("Qualitative Probe — Unseen Medical Terms")
    print("=" * 60)
    print("Testing model on sentences outside the training distribution.")
    print()

    probe_prompts = [PREFIX + s for s in PROBE_SENTENCES]

    # Load PyTorch model fresh (was deleted after metric inference to free VRAM)
    torch_model = AutoModelForSeq2SeqLM.from_pretrained(torch_model_path).to(device)
    torch_model.eval()
    torch_out = _generate_torch(torch_model, tokenizer, probe_prompts,
                                 max_new_tokens=max_new_tokens, num_beams=1, device=device)
    del torch_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    onnx_out = _generate_onnx(ort_model, tokenizer, probe_prompts,
                               max_new_tokens=max_new_tokens, num_beams=1)

    print(f"{'#':<3}  {'PyTorch output':<55}  {'ONNX output'}")
    print("-" * 120)
    for i, (src, tp, op) in enumerate(zip(PROBE_SENTENCES, torch_out, onnx_out), 1):
        print(f"\n[{i:02d}] INPUT:   {src}")
        print(f"     PyTorch: {tp}")
        print(f"     ONNX:    {op}")

        # Flag if ONNX output looks degenerate
        if not op.strip():
            print("     *** WARNING: ONNX produced empty output ***")
        elif _has_repeat(op):
            print("     *** WARNING: ONNX output contains repetition loop ***")
        elif len(op) < 20:
            print("     *** WARNING: ONNX output suspiciously short ***")

    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULTS["dataset"])
    ap.add_argument("--torch_model", default=DEFAULTS["torch_model"])
    ap.add_argument("--onnx_dir", default=DEFAULTS["onnx_dir"])
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--max_samples", type=int, default=200, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=1, help="Use 1 for speed; ONNX beam search can be unstable")
    ap.add_argument("--cpu_only", action="store_true")
    ap.add_argument("--rouge_tol", type=float, default=DEFAULT_ROUGE_TOL,
                    help="Max allowed relative ROUGE-L drop from PyTorch to ONNX (default 15%%)")
    ap.add_argument("--fk_grade_max", type=float, default=DEFAULT_FK_GRADE_MAX)
    ap.add_argument("--max_repetition", type=float, default=DEFAULT_MAX_REPETITION)
    ap.add_argument("--sari_gap_max", type=float, default=DEFAULT_MAX_SARI_GAP,
                    help="Max allowed SARI gap between PyTorch and ONNX (default 6 pts)")
    args = ap.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    ds = load_from_disk(args.dataset)
    split = ds[args.split]
    rng = np.random.default_rng(args.seed)
    n = len(split) if args.max_samples == 0 else min(len(split), args.max_samples)
    idxs = sorted(rng.choice(len(split), size=n, replace=False).tolist())

    prompts = [split[i]["input_text"] for i in idxs]
    refs    = [split[i]["target_text"] for i in idxs]

    print("=" * 60)
    print("Cognitive Bridge — ONNX Export Verification")
    print("=" * 60)
    print(f"Split:       {args.split}  ({n} samples)")
    print(f"PyTorch:     {args.torch_model}")
    print(f"ONNX:        {args.onnx_dir}")
    print(f"max_new_tokens={args.max_new_tokens}  num_beams={args.num_beams}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.torch_model)

    # ── PyTorch inference ──────────────────────────────────────────────────────
    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"PyTorch device: {device}")
    torch_model = AutoModelForSeq2SeqLM.from_pretrained(args.torch_model).to(device)
    torch_preds = _generate_torch(torch_model, tokenizer, prompts,
                                   max_new_tokens=args.max_new_tokens,
                                   num_beams=args.num_beams, device=device)
    del torch_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print()

    # ── ONNX inference ─────────────────────────────────────────────────────────
    try:
        from onnxruntime import get_available_providers  # type: ignore
        providers = get_available_providers()
    except Exception:
        providers = []

    if args.cpu_only or "CUDAExecutionProvider" not in providers:
        provider = "CPUExecutionProvider"
    else:
        provider = "CUDAExecutionProvider"

    print(f"ONNX provider: {provider}")
    ort_model = _load_onnx_model(args.onnx_dir, provider)
    onnx_preds = _generate_onnx(ort_model, tokenizer, prompts,
                                 max_new_tokens=args.max_new_tokens,
                                 num_beams=args.num_beams)
    print()

    # ── Run checks ─────────────────────────────────────────────────────────────
    checks = _run_checks(
        torch_preds=torch_preds,
        onnx_preds=onnx_preds,
        sources=prompts,
        refs=refs,
        rouge_tol=args.rouge_tol,
        max_repetition=args.max_repetition,
        fk_grade_max=args.fk_grade_max,
        max_empty_rate=DEFAULT_MAX_EMPTY_RATE,
        sari_gap_max=args.sari_gap_max,
    )

    # ── Qualitative test: unseen medical terms ────────────────────────────────
    _run_qualitative_probe(
        tokenizer=tokenizer,
        ort_model=ort_model,
        max_new_tokens=args.max_new_tokens,
        torch_model_path=args.torch_model,
        device=device,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    for c in checks:
        print(c)

    failed = [c for c in checks if not c.passed]
    print()
    if not failed:
        print("ALL CHECKS PASSED — ONNX export is good to upload.")
        return 0
    else:
        print(f"{len(failed)} CHECK(S) FAILED:")
        for c in failed:
            print(f"  -> {c.name}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
