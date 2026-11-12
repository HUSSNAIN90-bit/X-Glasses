#!/usr/bin/env python3
"""Export the repository's YOLO26n detector to a mobile LiteRT model."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "yolo26n.pt"
OUTPUT = ROOT / "models" / "yolo26n.tflite"
MANIFEST = ROOT / "models" / "yolo26n_manifest.json"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Expected YOLO26 weights at {SOURCE}")
    model = YOLO(str(SOURCE))
    exported = Path(model.export(format="litert", quantize="w8a32", imgsz=640, batch=1))
    if not exported.is_file():
        raise FileNotFoundError(f"Ultralytics did not create the LiteRT model: {exported}")
    if exported.resolve() != OUTPUT.resolve():
        shutil.copy2(exported, OUTPUT)
    # Smoke-test the exported artifact with a local dummy frame before publishing it.
    mobile_model = YOLO(str(OUTPUT))
    mobile_model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    names = model.names
    if isinstance(names, list):
        labels = names
    else:
        labels = [names[index] for index in sorted(names)]
    manifest = {
        "model": "yolo26n",
        "format": "litert",
        "input_size": 640,
        "quantization": "w8a32",
        "output_layout": "end2end_xyxy_score_class",
        "labels": labels,
        "sha256": digest,
        "bytes": OUTPUT.stat().st_size,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {OUTPUT} ({manifest['bytes']} bytes)")
    print(f"SHA-256: {digest}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
