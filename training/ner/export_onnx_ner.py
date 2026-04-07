"""
Export the fine-tuned NER model to ONNX format with 8-bit quantization.

Converts the DistilBERT NER model to ONNX and applies dynamic quantization
to reduce model size for browser deployment via Transformers.js.

Usage:
    python training/ner/finetune_ner.py       # train first
    python training/ner/export_onnx_ner.py

Output:
    training/ner/onnx_model/  — quantized ONNX model
"""

import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "best_model")
ONNX_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")
QUANTIZED_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_quantized")


def export_to_onnx():
    """Export the PyTorch model to ONNX format."""
    print(f"Exporting model from {MODEL_PATH} to ONNX...")

    os.makedirs(ONNX_OUTPUT, exist_ok=True)

    main_export(
        model_name_or_path=MODEL_PATH,
        output=Path(ONNX_OUTPUT),
        task="token-classification",
        opset=14,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(ONNX_OUTPUT)

    print(f"ONNX model exported to {ONNX_OUTPUT}")
    return ONNX_OUTPUT


def quantize_model(onnx_path):
    """Apply 8-bit dynamic quantization to reduce model size."""
    print("Applying 8-bit dynamic quantization...")

    os.makedirs(QUANTIZED_OUTPUT, exist_ok=True)

    model = ORTModelForTokenClassification.from_pretrained(onnx_path)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    from optimum.onnxruntime import ORTQuantizer

    quantizer = ORTQuantizer.from_pretrained(model)
    quantizer.quantize(save_dir=QUANTIZED_OUTPUT, quantization_config=qconfig)

    tokenizer = AutoTokenizer.from_pretrained(onnx_path)
    tokenizer.save_pretrained(QUANTIZED_OUTPUT)

    original_size = sum(
        f.stat().st_size for f in Path(onnx_path).rglob("*.onnx")
    ) / (1024 * 1024)
    quantized_size = sum(
        f.stat().st_size for f in Path(QUANTIZED_OUTPUT).rglob("*.onnx")
    ) / (1024 * 1024)

    print(f"Original ONNX size:  {original_size:.1f} MB")
    print(f"Quantized ONNX size: {quantized_size:.1f} MB")
    print(f"Compression ratio:   {original_size / max(quantized_size, 0.01):.1f}x")
    print(f"Quantized model saved to {QUANTIZED_OUTPUT}")


def verify_model():
    """Quick verification that the quantized model works."""
    print("\nVerifying quantized model...")

    model = ORTModelForTokenClassification.from_pretrained(QUANTIZED_OUTPUT)
    tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_OUTPUT)

    test_text = "The patient was diagnosed with hypertension and prescribed lisinopril."
    inputs = tokenizer(test_text, return_tensors="np")
    outputs = model(**inputs)

    import numpy as np
    predictions = np.argmax(outputs.logits, axis=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print(f"Input: {test_text}")
    print("Predictions:")
    for token, pred_id in zip(tokens, predictions[0]):
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            print(f"  {token:20s} -> {pred_id}")

    print("Verification complete.")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run finetune_ner.py first to train the model.")
        return

    onnx_path = export_to_onnx()
    quantize_model(onnx_path)
    verify_model()


if __name__ == "__main__":
    main()
