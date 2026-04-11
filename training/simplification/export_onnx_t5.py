"""
Export the fine-tuned T5 model to ONNX format with 8-bit quantization.

Converts the T5 simplification model to ONNX with merged decoders
(combining with-past and without-past into one file) and applies
dynamic quantization for browser deployment via Transformers.js.

Usage:
    python training/simplification/finetune_t5.py       # train first
    python training/simplification/export_onnx_t5.py

Output:
    training/simplification/onnx_quantized/  — quantized ONNX model
"""

import os
import shutil
from pathlib import Path
from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "best_model")
ONNX_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")
QUANTIZED_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_quantized")


def export_to_onnx():
    """Export T5 to ONNX with merged decoder (required by Transformers.js)."""
    print(f"Exporting T5 model from {MODEL_PATH} to ONNX...")

    if os.path.exists(ONNX_OUTPUT):
        shutil.rmtree(ONNX_OUTPUT)
    os.makedirs(ONNX_OUTPUT, exist_ok=True)

    main_export(
        model_name_or_path=MODEL_PATH,
        output=Path(ONNX_OUTPUT),
        task="text2text-generation-with-past",
        opset=14,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(ONNX_OUTPUT)

    print(f"ONNX model exported to {ONNX_OUTPUT}")

    onnx_files = list(Path(ONNX_OUTPUT).rglob("*.onnx"))
    print(f"Generated {len(onnx_files)} ONNX files:")
    for f in onnx_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    return ONNX_OUTPUT


def quantize_model(onnx_path):
    """Apply 8-bit dynamic quantization to each ONNX file individually."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("\nApplying 8-bit dynamic quantization...")

    if os.path.exists(QUANTIZED_OUTPUT):
        shutil.rmtree(QUANTIZED_OUTPUT)
    os.makedirs(QUANTIZED_OUTPUT, exist_ok=True)

    for src_file in Path(onnx_path).iterdir():
        dest_file = Path(QUANTIZED_OUTPUT) / src_file.name
        if src_file.suffix == ".onnx":
            quantized_name = src_file.stem + "_quantized.onnx"
            dest_quantized = Path(QUANTIZED_OUTPUT) / quantized_name
            print(f"  Quantizing {src_file.name} -> {quantized_name}")
            try:
                quantize_dynamic(
                    model_input=str(src_file),
                    model_output=str(dest_quantized),
                    weight_type=QuantType.QUInt8,
                )
            except Exception as e:
                print(f"    Warning: quantization failed for {src_file.name}: {e}")
                print(f"    Copying unquantized file instead.")
                shutil.copy2(src_file, dest_quantized)
        else:
            shutil.copy2(src_file, dest_file)

    original_size = sum(
        f.stat().st_size for f in Path(onnx_path).rglob("*.onnx")
    ) / (1024 * 1024)
    quantized_size = sum(
        f.stat().st_size for f in Path(QUANTIZED_OUTPUT).rglob("*.onnx")
    ) / (1024 * 1024)

    print(f"\nOriginal total size:  {original_size:.1f} MB")
    print(f"Quantized total size: {quantized_size:.1f} MB")
    print(f"Compression ratio:    {original_size / max(quantized_size, 0.01):.1f}x")
    print(f"Quantized model saved to {QUANTIZED_OUTPUT}")


def setup_for_transformers_js():
    """
    Arrange files for Transformers.js compatibility.
    Transformers.js expects ONNX files inside an onnx/ subdirectory.
    """
    onnx_subdir = Path(QUANTIZED_OUTPUT) / "onnx"
    os.makedirs(onnx_subdir, exist_ok=True)

    for f in Path(QUANTIZED_OUTPUT).glob("*.onnx"):
        dest = onnx_subdir / f.name
        shutil.copy2(f, dest)
        print(f"  Copied {f.name} -> onnx/{f.name}")

    print(f"\nModel ready for Transformers.js at {QUANTIZED_OUTPUT}")
    print("Files in onnx/ subdirectory:")
    for f in sorted(onnx_subdir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run finetune_t5.py first to train the model.")
        return

    onnx_path = export_to_onnx()
    quantize_model(onnx_path)
    setup_for_transformers_js()


if __name__ == "__main__":
    main()
