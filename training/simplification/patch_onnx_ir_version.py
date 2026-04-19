"""
Patch exported ONNX files from IR version 9 -> 8.

@xenova/transformers v2.x bundles ORT-WASM built against ONNX <=1.13,
which only supports IR version <=8. The HPRC export environment has onnx
>=1.14 which writes IR version 9 by default, causing a hard session-load
failure in the browser.

Run this on the login node after export, before uploading to HuggingFace:
    python training/simplification/patch_onnx_ir_version.py
"""

import os
from pathlib import Path

import onnx

TARGET_IR_VERSION = 8

DIRS = [
    Path(__file__).parent / "onnx_quantized_medisimplifier",
]


def patch_file(path: Path) -> None:
    model = onnx.load(str(path))
    if model.ir_version <= TARGET_IR_VERSION:
        print(f"  SKIP {path.name} (already IR version {model.ir_version})")
        return
    old = model.ir_version
    model.ir_version = TARGET_IR_VERSION
    onnx.save(model, str(path))
    print(f"  PATCHED {path.name}: IR {old} -> {TARGET_IR_VERSION}")


def main():
    for d in DIRS:
        if not d.exists():
            print(f"Directory not found: {d}")
            continue
        print(f"\nScanning {d} ...")
        for f in sorted(d.rglob("*.onnx")):
            patch_file(f)
    print("\nDone.")


if __name__ == "__main__":
    main()
