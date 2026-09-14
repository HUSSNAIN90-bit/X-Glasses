from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import zxingcpp


@dataclass(frozen=True)
class BarcodeResult:
    data: str
    format: str
    position: list[list[float]] | None = None


def _position(result) -> list[list[float]] | None:
    position = getattr(result, "position", None)
    if position is None:
        return None

    points = []
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        point = getattr(position, name, None)
        if point is not None:
            points.append([float(point.x), float(point.y)])
    return points or None


def scan_barcodes(image: np.ndarray) -> list[BarcodeResult]:
    """Decode QR/1D barcodes locally with ZXing-C++."""
    if image is None or image.size == 0:
        return []

    try:
        results = zxingcpp.read_barcodes(image, try_rotate=True, try_downscale=True)
    except (RuntimeError, ValueError, TypeError, cv2.error):
        return []

    decoded: list[BarcodeResult] = []
    seen: set[tuple[str, str]] = set()

    for result in results:
        data = str(getattr(result, "text", "") or "").strip()
        if not data:
            continue

        fmt = str(getattr(result, "format", "UNKNOWN"))
        key = (fmt, data)
        if key in seen:
            continue
        seen.add(key)
        decoded.append(
            BarcodeResult(
                data=data,
                format=fmt,
                position=_position(result),
            )
        )

    return decoded


def scan_barcode(image: np.ndarray) -> BarcodeResult | None:
    results = scan_barcodes(image)
    return results[0] if results else None
