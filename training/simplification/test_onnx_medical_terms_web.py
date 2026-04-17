#!/usr/bin/env python3
"""
Qualitative smoke test: fetch ~10 medical condition names from the web (NLM Clinical Tables API),
run the **ONNX** medisimplifier model on short prompts, and print outputs.

Does not use the local simplification dataset.

Uses the same prompt prefix as training (see prepare_simplification_medisimplifier.py).

Example:
  cd /path/to/Cognitive-Bridge
  source /path/to/cb-venv/bin/activate
  python training/simplification/test_onnx_medical_terms_web.py \\
    --onnx_dir training/simplification/onnx_quantized_medisimplifier \\
    --torch_model training/simplification/output_medisimplifier/best_model

  # Offline / no HTTP:
  python training/simplification/test_onnx_medical_terms_web.py --no_network
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

import torch
from transformers import AutoTokenizer

# Same directory as evaluate_t5_medisimplifier.py
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import evaluate_t5_medisimplifier as med_eval  # noqa: E402

PREFIX = "Simplify this medical text for a patient: "

# Used when --no_network or API fails (common clinical terms / phrases).
FALLBACK_TERMS: List[str] = [
    "Hypertension",
    "Type 2 diabetes mellitus",
    "Acute myocardial infarction",
    "Atrial fibrillation",
    "Chronic obstructive pulmonary disease",
    "Deep vein thrombosis",
    "Hyperlipidemia",
    "Pneumonia",
    "Acute kidney injury",
    "Gastroesophageal reflux disease",
]


def fetch_terms_nlm_clinical_tables(
    n: int,
    timeout: float,
) -> List[str]:
    """
    Fetch condition primary names from the NIH/NLM Clinical Tables API (public, no API key).
    https://clinicaltables.nlm.nih.gov/apidoc/conditions/v3/
    """
    base = "https://clinicaltables.nlm.nih.gov/api/conditions/v3/search"
    # Short letter prefixes yield diverse ICD-style condition names.
    seeds = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
    seen: set[str] = set()
    out: List[str] = []

    for seed in seeds:
        if len(out) >= n:
            break
        params = urllib.parse.urlencode(
            {
                "sf": "primary_name",
                "df": "primary_name",
                "terms": seed,
                "count": str(max(5, n)),
            }
        )
        url = f"{base}?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Cognitive-Bridge-medterm-test/1.0 (educational)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        # Typical shape: [ total, null, null, [ ["Name1"], ["Name2"], ... ] ]
        rows = payload[3] if isinstance(payload, list) and len(payload) > 3 else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if len(out) >= n:
                break
            cell = row[0] if isinstance(row, (list, tuple)) and row else row
            if not isinstance(cell, str):
                continue
            name = cell.strip()
            if len(name) < 4:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)

    return out[:n]


def _build_prompt(term: str) -> str:
    # Short, patient-facing request that includes the term (matches training style).
    return (
        PREFIX
        + f'Explain what "{term}" means in simple, non-technical language a patient can understand.'
    )


def _generate_one(
    ort_model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    num_beams: int,
    encoder_max_length: Optional[int],
) -> str:
    tok_kw: dict = {"return_tensors": "pt", "truncation": True}
    if encoder_max_length is not None:
        tok_kw["max_length"] = encoder_max_length
    inputs = tokenizer(prompt, **tok_kw)
    with torch.no_grad():
        gen_ids = ort_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
        )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="ONNX medisimplifier qualitative test with web-fetched terms.")
    ap.add_argument(
        "--torch_model",
        default=str(_SCRIPT_DIR / "output_medisimplifier" / "best_model"),
        help="Tokenizer + config (same as training checkpoint).",
    )
    ap.add_argument(
        "--onnx_dir",
        default=str(_SCRIPT_DIR / "onnx_quantized_medisimplifier"),
        help="ONNX bundle root (contains config.json; onnx/*.onnx).",
    )
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument(
        "--onnx_provider",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    ap.add_argument("--cpu_only", action="store_true")
    ap.add_argument("--n_terms", type=int, default=10, help="How many terms to test.")
    ap.add_argument("--no_network", action="store_true", help="Skip web fetch; use built-in fallback terms.")
    ap.add_argument("--fetch_timeout", type=float, default=20.0)
    ap.add_argument(
        "--onnx_max_input_length",
        type=int,
        default=None,
        help="Optional tokenizer max_length for ONNX (same meaning as evaluate_t5_medisimplifier.py).",
    )
    args = ap.parse_args()

    n = max(1, min(args.n_terms, len(FALLBACK_TERMS)))
    terms: List[str]
    if args.no_network:
        terms = FALLBACK_TERMS[:n]
        print("Using built-in fallback terms (--no_network).\n")
    else:
        try:
            terms = fetch_terms_nlm_clinical_tables(n=n, timeout=args.fetch_timeout)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as e:
            print(f"Web fetch failed ({e}); using built-in fallback terms.\n")
            terms = FALLBACK_TERMS[:n]
        if len(terms) < n:
            for t in FALLBACK_TERMS:
                if len(terms) >= n:
                    break
                if t not in terms:
                    terms.append(t)
            terms = terms[:n]

    print("Terms to test:")
    for i, t in enumerate(terms, 1):
        print(f"  {i}. {t}")
    print("")

    from optimum.onnxruntime import ORTModelForSeq2SeqLM  # type: ignore  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(args.torch_model)
    provider = med_eval._pick_execution_provider(
        onnx_provider=args.onnx_provider,
        cpu_only=args.cpu_only,
    )
    ort_model_id, ort_sub = med_eval._resolve_optimum_onnx_bundle(args.onnx_dir)

    ort_model = ORTModelForSeq2SeqLM.from_pretrained(
        ort_model_id,
        subfolder=ort_sub,
        local_files_only=med_eval._is_offline_mode(),
        encoder_file_name="encoder_model_quantized.onnx",
        use_cache=True,
        use_merged=True,
        provider=provider,
    )
    med_eval._patch_optimum_ort_t5_kv_head_dim(ort_model)

    inferred = med_eval._infer_onnx_encoder_max_length(ort_model)
    enc_cap = args.onnx_max_input_length if args.onnx_max_input_length is not None else inferred
    if enc_cap is not None:
        print(f"ONNX encoder token cap: {enc_cap}\n")

    print(f"ONNX provider: {provider}")
    print(f"num_beams: {args.num_beams}")
    print("")
    print("=" * 72)

    for i, term in enumerate(terms, 1):
        prompt = _build_prompt(term)
        text = _generate_one(
            ort_model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            encoder_max_length=enc_cap,
        )
        print(f"\n[{i}/{len(terms)}] Term: {term}")
        print("-" * 72)
        print("Prompt:")
        print(prompt)
        print("-" * 72)
        print("ONNX output:")
        print(text)
        print("=" * 72)

    print("\nNote: This is a qualitative check only; have a clinician review real patient-facing text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
