#!/usr/bin/env python3
"""
Evaluate Medisimplifier T5 simplification:
  - PyTorch checkpoint (best_model)
  - Quantized ONNX bundle produced by export_onnx_t5_medisimplifier.py

Metrics (offline-friendly):
  - ROUGE (via `evaluate`, cached by download_assets.sh in typical setups)
  - Simple length stats (avg chars)

Example:
  module purge
  module load GCCcore/13.3.0 Python/3.12.3
  source /scratch/user/vinaysingh/cb-venv/bin/activate
  export HF_HOME=/scratch/user/vinaysingh/.cache/huggingface

  python training/simplification/evaluate_t5_medisimplifier.py \
    --split test \
    --max_samples 200 \
    --torch_model training/simplification/output_medisimplifier/best_model \
    --onnx_dir training/simplification/onnx_quantized_medisimplifier
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _try_load_rouge():
    import evaluate  # type: ignore

    # Offline-friendly: relies on cached metric scripts under HF_HOME.
    return evaluate.load("rouge")


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _get_ort_available_providers() -> List[str]:
    try:
        import onnxruntime as ort  # type: ignore

        ps = ort.get_available_providers()
        return list(ps) if isinstance(ps, (list, tuple)) else [str(ps)]
    except Exception:
        return []


def _pick_execution_provider(onnx_provider: str, cpu_only: bool) -> str:
    """
    Decide which ONNX Runtime execution provider to request.

    Important: PyTorch CUDA availability does NOT imply ONNX Runtime has CUDA EP installed.
    We must query onnxruntime.get_available_providers().
    """
    if cpu_only:
        return "CPUExecutionProvider"

    available = _get_ort_available_providers()
    preferred_cuda = "CUDAExecutionProvider"
    preferred_cpu = "CPUExecutionProvider"

    if onnx_provider == "cpu":
        return preferred_cpu
    if onnx_provider == "cuda":
        if preferred_cuda in available:
            return preferred_cuda
        raise SystemExit(
            "ONNX provider was forced to CUDA but CUDAExecutionProvider is not available.\n"
            f"Available providers: {available}\n"
            "Fix: install a CUDA-enabled onnxruntime build (commonly `onnxruntime-gpu`) in this venv, "
            "or run with --onnx_provider auto/--onnx_provider cpu."
        )

    # auto
    if preferred_cuda in available:
        return preferred_cuda
    return preferred_cpu


def _onnx_dim_int(dim: Any) -> Optional[int]:
    """Positive static dimension from ONNX Runtime type shape, or None if symbolic/dynamic."""
    if isinstance(dim, int) and dim > 0:
        return dim
    if isinstance(dim, str) and dim.isdigit():
        return int(dim)
    return None


def _infer_onnx_encoder_max_length(ort_model: Any) -> Optional[int]:
    """
    Some ONNX bundles were exported with a *fixed* encoder sequence length (e.g. 64). If the tokenizer
    emits longer encoder inputs, the merged decoder's cross-attention past inputs won't match and ORT
    raises INVALID_ARGUMENT (e.g. ... encoder.value ... Got: 85 Expected: 64).

    Prefer static shapes from encoder `input_ids` (sequence axis), else from merged-decoder
    `past_key_values.*.encoder.value` at axis 2 (typical T5 layout [B, H, enc_seq, d_kv]).
    """
    try:
        enc_sess = ort_model.encoder.session
        for inp in enc_sess.get_inputs():
            if inp.name != "input_ids":
                continue
            sh = inp.shape
            if len(sh) >= 2:
                d1 = _onnx_dim_int(sh[1])
                if d1 is not None:
                    return d1
    except Exception:
        pass

    try:
        dwp = ort_model.decoder_with_past
        if dwp is not None and getattr(dwp, "session", None) is not None:
            for inp in dwp.session.get_inputs():
                nm = inp.name
                if ".encoder." not in nm or not nm.endswith(".value"):
                    continue
                sh = inp.shape
                if len(sh) >= 4:
                    ds = _onnx_dim_int(sh[2])
                    if ds is not None:
                        return ds
    except Exception:
        pass

    return None


def _parse_ort_dim_mismatch(exc: Exception) -> Optional[Tuple[int, int, int]]:
    """
    Parse ORT errors like:
      ... index: 3 Got: 85 Expected: 64
    Returns (axis_index, got, expected) or None.

    For T5 cross-attn past values shaped [B, H, S, d_kv], axis 3 is **d_kv**. A common cause of 85 vs 64 is
    Optimum using d_model//num_heads (85) instead of config.d_kv (64) when building dummy past tensors;
    see `_patch_optimum_ort_t5_kv_head_dim`.
    """
    msg = str(exc)
    m = re.search(r"index:\s*(\d+)\s+Got:\s*(\d+)\s+Expected:\s*(\d+)", msg)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    except Exception:
        return None


def _patch_optimum_ort_t5_kv_head_dim(ort_model: Any) -> None:
    """
    Optimum's `ORTDecoderForSeq2Seq` sets `embed_size_per_head = hidden_size // num_attention_heads`.
    T5 uses a separate `d_kv` for K/V depth; e.g. d_model=512, num_heads=6 gives 85, but `d_kv` is 64.
    Dummy `past_key_values` tensors then use the wrong last dim vs ONNX exported with `d_kv`.
    """
    cfg = getattr(ort_model, "config", None)
    if cfg is None:
        return
    if getattr(cfg, "model_type", None) not in ("t5", "mt5", "longt5"):
        return
    dkv = getattr(cfg, "d_kv", None)
    if dkv is None:
        return
    want = int(dkv)
    for name in ("decoder", "decoder_with_past"):
        dec = getattr(ort_model, name, None)
        if dec is None or not hasattr(dec, "embed_size_per_head"):
            continue
        cur = int(getattr(dec, "embed_size_per_head"))
        if cur != want:
            print(
                f"ONNX: patching {name}.embed_size_per_head {cur} -> {want} "
                "(T5 `d_kv`; Optimum defaults to d_model//num_heads for dummy K/V tensors).",
                flush=True,
            )
            setattr(dec, "embed_size_per_head", want)


def _is_offline_mode() -> bool:
    return (
        os.environ.get("HF_HUB_OFFLINE", "") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"
        or os.environ.get("HF_DATASETS_OFFLINE", "") == "1"
        or os.environ.get("HF_EVALUATE_OFFLINE", "") == "1"
    )


def _resolve_optimum_onnx_bundle(onnx_dir: str) -> Tuple[str, str]:
    """
    Optimum's `ORTModel*.from_pretrained()` infers the backend library from `config.json`
    in the *model_id directory* (see `optimum.modeling_base.OptimizedModel.from_pretrained`).

    Our export script copies ONNX files into `<bundle>/onnx/` for Transformers.js, but keeps
    `config.json` at `<bundle>/`. If the user points `--onnx_dir` at `<bundle>/onnx`, library
    inference fails with:
      ValueError: The library name could not be automatically inferred ...

    Returns:
      (model_id, subfolder) where ONNX files live under model_id/subfolder (subfolder may be "").
    """
    p = Path(onnx_dir).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"--onnx_dir does not exist: {p}")

    # Case A: user points at bundle root (contains config.json; ONNX may be top-level or in onnx/)
    if (p / "config.json").is_file():
        onnx_only_dir = p / "onnx"
        if onnx_only_dir.is_dir() and any(onnx_only_dir.glob("*.onnx")):
            return p.as_posix(), "onnx"
        return p.as_posix(), ""

    # Case B: user points at `<bundle>/onnx` but config lives in parent `<bundle>/`
    parent = p.parent
    if p.name.lower() == "onnx" and (parent / "config.json").is_file():
        return parent.as_posix(), "onnx"

    # Case C: ONNX-only directory without an obvious config.json nearby
    raise SystemExit(
        "Could not locate a Transformers `config.json` next to the ONNX bundle.\n"
        f"Got --onnx_dir={onnx_dir}\n"
        "Expected either:\n"
        "  - <bundle>/ containing config.json (and optionally onnx/*.onnx), or\n"
        "  - <bundle>/onnx/ containing *.onnx with <bundle>/config.json one level up.\n"
        "Tip: for this repo's export layout, use:\n"
        "  --onnx_dir training/simplification/onnx_quantized_medisimplifier\n"
        "or keep the default (same as above) instead of .../onnx_quantized_medisimplifier/onnx"
    )


def _generate_torch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> List[str]:
    outs: List[str] = []
    model.eval()
    with torch.no_grad():
        for p in tqdm(prompts, desc="torch.generate"):
            inputs = tokenizer(p, return_tensors="pt", truncation=True).to(device)
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
            )
            text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            outs.append(text)
    return outs


def _generate_onnx(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    num_beams: int,
    encoder_max_length: Optional[int],
    d_kv: int,
) -> List[str]:
    outs: List[str] = []
    # ORTModelForSeq2SeqLM is not a torch.nn.Module; it has no .eval().
    active_encoder_max_length = encoder_max_length
    active_beams = num_beams
    with torch.no_grad():
        for p in tqdm(prompts, desc="onnx.generate"):
            tok_kw: Dict[str, Any] = {"return_tensors": "pt", "truncation": True}
            if active_encoder_max_length is not None:
                tok_kw["max_length"] = active_encoder_max_length
            inputs = tokenizer(p, **tok_kw)
            try:
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=active_beams,
                )
            except Exception as exc:
                parsed = _parse_ort_dim_mismatch(exc)
                if not parsed:
                    raise
                idx, _got, exp = parsed
                # Axis 3 == d_kv for standard [B,H,S,d_kv] — beam search + merged ORT decoder often breaks this.
                if idx == 3 and exp == d_kv and active_beams > 1:
                    active_beams = 1
                    print(
                        "ONNX: ORT cross-attn past shape mismatch on head-dim axis (likely beam search). "
                        f"Switching to num_beams=1 for this and remaining samples. ({exc.__class__.__name__})",
                        flush=True,
                    )
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                # Axis 2 is usually encoder sequence length when static export is too small.
                elif idx == 2 and exp != d_kv:
                    active_encoder_max_length = exp
                    print(
                        f"ONNX: retrying with encoder max_length={active_encoder_max_length} (static export cap).",
                        flush=True,
                    )
                    inputs = tokenizer(
                        p,
                        return_tensors="pt",
                        truncation=True,
                        max_length=active_encoder_max_length,
                    )
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=active_beams,
                    )
                else:
                    raise
            text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            outs.append(text)
    return outs


def _score(preds: List[str], refs: List[str]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    rouge = _try_load_rouge()
    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    stats = {
        "avg_pred_chars": _mean([len(x) for x in preds]),
        "avg_ref_chars": _mean([len(x) for x in refs]),
    }
    return r, stats


def _print_rouge_block(title: str, rouge: Dict[str, Any]) -> None:
    print(f"=== ROUGE: {title} ===")
    # `evaluate` ROUGE returns a dict of floats OR nested dicts depending on version.
    # Print everything in a stable, copy/paste friendly way.
    def _to_jsonable(x: Any) -> Any:
        # numpy scalars / torch scalars
        if hasattr(x, "item"):
            try:
                return float(x.item())
            except Exception:
                return str(x)

        # rouge_score AggregateScore-like objects often expose low/mid/high with fmeasure/precision/recall
        mid = getattr(x, "mid", None)
        low = getattr(x, "low", None)
        high = getattr(x, "high", None)
        if mid is not None or low is not None or high is not None:
            out: Dict[str, Any] = {}
            for band_name, band in (("low", low), ("mid", mid), ("high", high)):
                if band is None:
                    continue
                band_out: Dict[str, Any] = {}
                for m in ("precision", "recall", "fmeasure"):
                    mv = getattr(band, m, None)
                    if mv is None:
                        continue
                    try:
                        band_out[m] = float(mv)
                    except Exception:
                        band_out[m] = str(mv)
                out[band_name] = band_out
            return out

        # plain dict/list/tuple
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]

        # floats/ints/strings
        if isinstance(x, (float, int, str)) or x is None:
            return x

        return str(x)

    printable = {str(k): _to_jsonable(v) for k, v in rouge.items()}
    print(json.dumps(printable, indent=2, sort_keys=True))

    # Also print common headline metrics as percentages when they look like scalars in [0,1].
    for k in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
        if k not in printable:
            continue
        v = printable[k]
        x: Optional[float] = None
        if isinstance(v, float):
            x = v
        elif isinstance(v, dict):
            # Prefer mid fmeasure when present.
            mid = v.get("mid") if isinstance(v.get("mid"), dict) else None
            if isinstance(mid, dict) and "fmeasure" in mid:
                try:
                    x = float(mid["fmeasure"])  # type: ignore[arg-type]
                except Exception:
                    x = None
        if x is None:
            continue
        if 0.0 <= x <= 1.0:
            print(f"{k} (mid fmeasure %): {x * 100:.4f}")
    print("")


def _choose_qualitative_indices(
    prompts: List[str],
    n: int,
    seed: int,
) -> List[int]:
    if n <= 0:
        return []

    # Heuristic: prefer examples containing common clinical abbreviations/terms.
    patterns = [
        r"\bHTN\b",
        r"\bDM\b|\bT2DM\b|\bT1DM\b",
        r"\bCOPD\b",
        r"\bCHF\b",
        r"\bCAD\b",
        r"\bMI\b",
        r"\bCVA\b|\bstroke\b",
        r"\bSOB\b|\bdyspnea\b",
        r"\bN/V\b|\bnausea\b|\bvomit",
        r"\bPRN\b",
        r"\bBID\b|\bTID\b|\bQID\b|\bqhs\b|\bq\d+h\b",
        r"\bPO\b|\bIV\b|\bIM\b|\bSC\b",
        r"\bmg\b|\bmcg\b|\bml\b|\bunits\b",
        r"\bA1c\b|\bHbA1c\b",
        r"\bcreatinine\b|\bBUN\b|\bGFR\b",
        r"\bWBC\b|\bHgb\b|\bPLT\b",
        r"\bCT\b|\bMRI\b|\bCXR\b",
        r"\bEKG\b|\bECG\b",
        r"\banticoag\b|\bwarfarin\b|\bheparin\b|\bapixaban\b",
    ]
    rx = re.compile("|".join(f"(?:{p})" for p in patterns), flags=re.IGNORECASE)

    candidates = [i for i, p in enumerate(prompts) if rx.search(p)]
    rng = random.Random(seed)
    rng.shuffle(candidates)

    picked: List[int] = []
    for i in candidates:
        picked.append(i)
        if len(picked) >= n:
            return sorted(set(picked))

    # If not enough “clinical-looking” prompts, fill with random examples.
    rest = list(range(len(prompts)))
    rng.shuffle(rest)
    for i in rest:
        if i in picked:
            continue
        picked.append(i)
        if len(picked) >= n:
            break
    return sorted(set(picked))


def _dump_qualitative(
    *,
    indices: List[int],
    prompts: List[str],
    refs: List[str],
    torch_preds: List[str],
    onnx_preds: List[str],
    out_path: Optional[str],
) -> None:
    if not indices:
        return

    rows: List[Dict[str, Any]] = []
    for j, i in enumerate(indices, start=1):
        row = {
            "i": int(i),
            "input_text": prompts[i],
            "reference": refs[i],
            "torch_pred": torch_preds[i],
            "onnx_pred": onnx_preds[i],
        }
        rows.append(row)

        print("")
        print(f"=== Qualitative example {j}/{len(indices)} (row={i}) ===")
        print("INPUT:")
        print(prompts[i])
        print("")
        print("REFERENCE:")
        print(refs[i])
        print("")
        print("TORCH:")
        print(torch_preds[i])
        print("")
        print("ONNX:")
        print(onnx_preds[i])
        print("")

    if out_path:
        op = Path(out_path).expanduser().resolve()
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote qualitative examples to: {op}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "simplification_pairs_medisimplifier",
        ),
        help="DatasetDict path produced by prepare_simplification_medisimplifier.py",
    )
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--max_samples", type=int, default=200, help="0 means all rows in split")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument(
        "--torch_model",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_medisimplifier", "best_model"),
    )
    ap.add_argument(
        "--onnx_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_quantized_medisimplifier"),
        help="ONNX bundle directory. Prefer the bundle root (contains config.json). "
        "If you pass .../onnx, the script will auto-rewire to the parent bundle when config.json is there.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--cpu_only", action="store_true", help="Force CPU even if CUDA is available")
    ap.add_argument(
        "--onnx_provider",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Which ONNX Runtime execution provider to request. "
        "'auto' picks CUDAExecutionProvider only if onnxruntime actually has it; otherwise CPUExecutionProvider.",
    )
    ap.add_argument(
        "--qualitative_samples",
        type=int,
        default=0,
        help="If >0, print N side-by-side examples (input/reference/torch/onnx).",
    )
    ap.add_argument(
        "--qualitative_out",
        default="",
        help="Optional path to write qualitative examples as JSONL (one example per line).",
    )
    ap.add_argument(
        "--onnx_max_input_length",
        type=int,
        default=None,
        help="Max tokenizer length for ONNX inference only. If unset, inferred from static ONNX shapes "
        "when possible; otherwise no extra cap (full-dynamic graphs). Use when ORT errors on encoder "
        "past / cross-attn dimensions.",
    )
    ap.add_argument(
        "--onnx_num_beams",
        type=int,
        default=None,
        help="Override --num_beams for ONNX only (e.g. 1 for debugging). Default: same as --num_beams.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = load_from_disk(args.dataset)
    if args.split not in ds:
        raise SystemExit(f"Split not found: {args.split}. Available: {list(ds.keys())}")

    split = ds[args.split]
    n = len(split) if args.max_samples == 0 else min(len(split), args.max_samples)

    # Deterministic subsample for speed (first N after shuffle of indices)
    idxs = list(range(len(split)))
    random.shuffle(idxs)
    idxs = sorted(idxs[:n])

    prompts = [split[i]["input_text"] for i in idxs]
    refs = [split[i]["target_text"] for i in idxs]

    print(f"Dataset: {args.dataset}")
    print(f"Split:   {args.split}")
    print(f"Rows:    {n} / {len(split)}")
    print(f"Torch:   {args.torch_model}")
    print(f"ONNX:    {args.onnx_dir}")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(args.torch_model)

    # ---- PyTorch ----
    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_model = AutoModelForSeq2SeqLM.from_pretrained(args.torch_model)
    torch_model.to(device)
    torch_preds = _generate_torch(
        torch_model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )
    rouge_t, stats_t = _score(torch_preds, refs)
    print("=== PyTorch ===")
    print(f"device: {device}")
    _print_rouge_block("PyTorch vs reference", rouge_t)
    print(f"avg_pred_chars: {stats_t['avg_pred_chars']:.1f}")
    print(f"avg_ref_chars:  {stats_t['avg_ref_chars']:.1f}")
    print("")

    # ---- ONNX (quantized) ----
    from optimum.onnxruntime import ORTModelForSeq2SeqLM  # local import (optional dependency path)

    provider = _pick_execution_provider(onnx_provider=args.onnx_provider, cpu_only=args.cpu_only)
    ort_model_id, ort_onnx_subfolder = _resolve_optimum_onnx_bundle(args.onnx_dir)
    requested_onnx_dir = Path(args.onnx_dir).expanduser().resolve()
    effective_onnx_dir = Path(ort_model_id) / ort_onnx_subfolder if ort_onnx_subfolder else Path(ort_model_id)
    if requested_onnx_dir != effective_onnx_dir:
        print(
            "Note: resolved Optimum ONNX load paths for library inference + file discovery:\n"
            f"  requested --onnx_dir: {requested_onnx_dir}\n"
            f"  optimum model_id:     {ort_model_id}\n"
            f"  optimum subfolder:    {ort_onnx_subfolder!r}\n"
            f"  effective onnx dir:   {effective_onnx_dir}\n"
        )

    ort_model = ORTModelForSeq2SeqLM.from_pretrained(
        ort_model_id,
        subfolder=ort_onnx_subfolder,
        local_files_only=_is_offline_mode(),
        # NOTE: Optimum's seq2seq loader has a subtle branch: if you pass *any* decoder file name while
        # `use_merged=True`, it can fall through to the non-merged decoder selection path. We only pin the
        # encoder filename (usually unique); merged decoder discovery should pick `decoder_model_merged_*.onnx`.
        encoder_file_name="encoder_model_quantized.onnx",
        use_cache=True,
        use_merged=True,
        provider=provider,
    )

    _patch_optimum_ort_t5_kv_head_dim(ort_model)

    inferred_enc = _infer_onnx_encoder_max_length(ort_model)
    onnx_enc_cap = args.onnx_max_input_length if args.onnx_max_input_length is not None else inferred_enc
    onnx_beams = args.onnx_num_beams if args.onnx_num_beams is not None else args.num_beams
    if onnx_enc_cap is not None:
        print(
            "ONNX encoder token cap (truncation): "
            f"{onnx_enc_cap}"
            + (f" (inferred static limit={inferred_enc})" if inferred_enc is not None else "")
            + "\n  Tip: re-export ONNX with a larger `sequence_length` in export_onnx_t5_medisimplifier.py "
            "to avoid truncating long clinical inputs."
        )
        print("")
    else:
        print(
            "ONNX: no static encoder length inferred; tokenizer uses truncation=True without max_length.\n"
            "  If ORT fails on cross-attention / past_key_values shapes, pass --onnx_max_input_length N.\n"
        )

    d_kv = int(getattr(ort_model.config, "d_kv", 64))
    onnx_preds = _generate_onnx(
        ort_model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        num_beams=onnx_beams,
        encoder_max_length=onnx_enc_cap,
        d_kv=d_kv,
    )
    rouge_o, stats_o = _score(onnx_preds, refs)
    print("=== ONNX (quantized) ===")
    print(f"provider: {provider}")
    _print_rouge_block("ONNX vs reference", rouge_o)
    print(f"avg_pred_chars: {stats_o['avg_pred_chars']:.1f}")
    print(f"avg_ref_chars:  {stats_o['avg_ref_chars']:.1f}")
    print("")

    # Agreement between torch vs onnx on identical prompts (sanity)
    agree = sum(1 for a, b in zip(torch_preds, onnx_preds) if a == b)
    print("=== Torch vs ONNX (exact string match rate) ===")
    print(f"exact_match: {agree}/{n} ({100.0 * agree / max(n, 1):.2f}%)")

    # Qualitative examples
    qn = int(args.qualitative_samples or 0)
    q_out = args.qualitative_out.strip() or None
    q_idxs = _choose_qualitative_indices(prompts, n=qn, seed=args.seed)
    _dump_qualitative(
        indices=q_idxs,
        prompts=prompts,
        refs=refs,
        torch_preds=torch_preds,
        onnx_preds=onnx_preds,
        out_path=q_out,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
