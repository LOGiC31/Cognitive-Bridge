"""
Export the fine-tuned T5 medisimplifier model to ONNX format with 8-bit quantization.

This script is a sibling of `export_onnx_t5.py` but uses:
  - training/simplification/output_medisimplifier/best_model

Usage:
  python training/simplification/finetune_t5_medisimplifier.py
  python training/simplification/export_onnx_t5_medisimplifier.py

Output:
  training/simplification/onnx_quantized_medisimplifier/ — quantized ONNX model
"""

import json
import os
import shutil
from pathlib import Path

from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer


def _configure_runtime_thread_env() -> None:
    """
    Slurm CPU affinity masks can make ONNX Runtime's pthread_setaffinity_np attempts fail,
    producing huge stderr spam during quantization. Setting explicit thread counts reduces this.
    """
    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ORT_INTRA_OP_NUM_THREADS": "1",
        "ORT_INTER_OP_NUM_THREADS": "1",
        # Avoid ORT trying to pin threads to CPUs outside the Slurm cgroup mask.
        "ORT_DISABLE_THREAD_AFFINITY": "1",
        # 3 = ERROR
        "ORT_LOG_SEVERITY_LEVEL": "3",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output_medisimplifier",
    "best_model",
)

ONNX_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model_medisimplifier")
QUANTIZED_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_quantized_medisimplifier")


def patch_generation_config(model_dir: str) -> None:
    """
    Overwrite generation_config.json with repetition-suppression params.
    Prevents the decoder repetition loop seen during ONNX inference under 8-bit quantization noise.
    """
    config_path = Path(model_dir) / "generation_config.json"
    existing = {}
    if config_path.exists():
        with open(config_path) as f:
            existing = json.load(f)

    existing.update(
        {
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.3,
            "max_new_tokens": 512,
        }
    )
    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Patched generation_config.json at {config_path}")


def export_to_onnx():
    print(f"Exporting T5 model from {MODEL_PATH} to ONNX...")

    if os.path.exists(ONNX_OUTPUT):
        shutil.rmtree(ONNX_OUTPUT)
    os.makedirs(ONNX_OUTPUT, exist_ok=True)

    main_export(
        model_name_or_path=MODEL_PATH,
        output=Path(ONNX_OUTPUT),
        task="text2text-generation-with-past",
        opset=14,
        # Optimum's default atol (1e-5) can be slightly too strict for large seq2seq
        # exports; we still validate, but allow tiny numerical drift.
        atol=1e-4,
        # Dummy export length for tracing; keeps encoder sequence axis compatible with long clinical inputs
        # (avoids static tiny shapes that break ORT when prompts tokenize longer than the export dummy).
        sequence_length=512,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(ONNX_OUTPUT)
    patch_generation_config(ONNX_OUTPUT)

    print(f"ONNX model exported to {ONNX_OUTPUT}")

    onnx_files = list(Path(ONNX_OUTPUT).rglob("*.onnx"))
    print(f"Generated {len(onnx_files)} ONNX files:")
    for f in onnx_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    return ONNX_OUTPUT


def quantize_model(onnx_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("\nApplying 8-bit dynamic quantization...")

    if os.path.exists(QUANTIZED_OUTPUT):
        shutil.rmtree(QUANTIZED_OUTPUT)
    os.makedirs(QUANTIZED_OUTPUT, exist_ok=True)

    # Optimum export writes ONNX files under ONNX_OUTPUT/ (top-level or subfolders).
    for src_file in Path(onnx_path).rglob("*"):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(onnx_path)
        dest_file = Path(QUANTIZED_OUTPUT) / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        if src_file.suffix == ".onnx":
            quantized_name = src_file.stem + "_quantized.onnx"
            dest_quantized = dest_file.with_name(quantized_name)
            print(f"  Quantizing {rel.as_posix()} -> {dest_quantized.relative_to(QUANTIZED_OUTPUT).as_posix()}")
            try:
                quantize_dynamic(
                    model_input=str(src_file),
                    model_output=str(dest_quantized),
                    weight_type=QuantType.QUInt8,
                )
            except Exception as e:
                print(f"    Warning: quantization failed for {rel.as_posix()}: {e}")
                print("    Copying unquantized file instead.")
                shutil.copy2(src_file, dest_quantized)
        else:
            shutil.copy2(src_file, dest_file)

    patch_generation_config(QUANTIZED_OUTPUT)

    original_size = sum(f.stat().st_size for f in Path(onnx_path).rglob("*.onnx")) / (1024 * 1024)
    quantized_size = sum(f.stat().st_size for f in Path(QUANTIZED_OUTPUT).rglob("*.onnx")) / (1024 * 1024)
    print(f"\nOriginal total size:  {original_size:.1f} MB")
    print(f"Quantized total size: {quantized_size:.1f} MB")
    print(f"Compression ratio:    {original_size / max(quantized_size, 0.01):.1f}x")
    print(f"Quantized model saved to {QUANTIZED_OUTPUT}")


def setup_for_transformers_js():
    """
    Arrange files for Transformers.js compatibility.
    Transformers.js expects ONNX files inside an onnx/ subdirectory.
    Only the encoder and merged decoder are needed — the separate decoder_model /
    decoder_with_past_model files are redundant when the merged variant is present
    and would waste ~110MB of browser cache.
    """
    onnx_subdir = Path(QUANTIZED_OUTPUT) / "onnx"
    os.makedirs(onnx_subdir, exist_ok=True)

    needed_prefixes = ("encoder_model", "decoder_model_merged")
    for f in Path(QUANTIZED_OUTPUT).glob("*.onnx"):
        if not any(f.name.startswith(p) for p in needed_prefixes):
            continue
        dest = onnx_subdir / f.name
        shutil.copy2(f, dest)
        print(f"  Copied {f.name} -> onnx/{f.name}")

    print(f"\nModel ready for Transformers.js at {QUANTIZED_OUTPUT}")
    print("Files in onnx/ subdirectory:")
    for f in sorted(onnx_subdir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


def main():
    _configure_runtime_thread_env()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run finetune_t5_medisimplifier.py first to train the model.")
        return

    onnx_path = export_to_onnx()
    quantize_model(onnx_path)
    setup_for_transformers_js()


if __name__ == "__main__":
    main()

