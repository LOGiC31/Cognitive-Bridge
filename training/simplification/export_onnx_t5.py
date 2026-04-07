"""
Export the fine-tuned T5 model to ONNX format with 8-bit quantization.

Converts the T5 simplification model (encoder + decoder) to ONNX and applies
dynamic quantization for browser deployment via Transformers.js.

Usage:
    python training/simplification/finetune_t5.py       # train first
    python training/simplification/export_onnx_t5.py

Output:
    training/simplification/onnx_model/  — quantized ONNX model
"""

import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "best_model")
ONNX_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")
QUANTIZED_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_quantized")


def export_to_onnx():
    """Export T5 to ONNX (encoder + decoder + decoder-with-past)."""
    print(f"Exporting T5 model from {MODEL_PATH} to ONNX...")

    os.makedirs(ONNX_OUTPUT, exist_ok=True)

    main_export(
        model_name_or_path=MODEL_PATH,
        output=Path(ONNX_OUTPUT),
        task="text2text-generation",
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
    """Apply 8-bit dynamic quantization to encoder and decoder."""
    print("\nApplying 8-bit dynamic quantization...")

    os.makedirs(QUANTIZED_OUTPUT, exist_ok=True)

    model = ORTModelForSeq2SeqLM.from_pretrained(onnx_path)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    from optimum.onnxruntime import ORTQuantizer

    for component_name in ["encoder_model", "decoder_model", "decoder_with_past_model"]:
        component = getattr(model, component_name, None)
        if component is not None:
            try:
                quantizer = ORTQuantizer.from_pretrained(model, file_name=f"{component_name}.onnx")
                quantizer.quantize(save_dir=QUANTIZED_OUTPUT, quantization_config=qconfig)
                print(f"  Quantized {component_name}")
            except Exception as e:
                print(f"  Warning: Could not quantize {component_name}: {e}")

    tokenizer = AutoTokenizer.from_pretrained(onnx_path)
    tokenizer.save_pretrained(QUANTIZED_OUTPUT)

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


def verify_model():
    """Quick verification that the quantized model generates output."""
    print("\nVerifying quantized model...")

    model = ORTModelForSeq2SeqLM.from_pretrained(QUANTIZED_OUTPUT)
    tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_OUTPUT)

    test_input = "simplify: The patient presents with acute myocardial infarction with ST-segment elevation."
    inputs = tokenizer(test_input, return_tensors="np", max_length=256, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=128)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Input:  {test_input}")
    print(f"Output: {decoded}")
    print("Verification complete.")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run finetune_t5.py first to train the model.")
        return

    onnx_path = export_to_onnx()
    quantize_model(onnx_path)
    verify_model()


if __name__ == "__main__":
    main()
