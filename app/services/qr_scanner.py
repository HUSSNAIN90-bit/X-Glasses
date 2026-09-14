from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class QRCodeResult:
    data: str
    points: list[list[float]] | None = None


def scan_qr_codes(image: np.ndarray) -> list[QRCodeResult]:
    """Decode QR codes locally from an OpenCV BGR frame."""
    if image is None or image.size == 0:
        return []

    detector = cv2.QRCodeDetector()
    results: list[QRCodeResult] = []

    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
        if ok and decoded_info:
            for index, value in enumerate(decoded_info):
                value = (value or "").strip()
                if not value:
                    continue
                polygon = None
                if points is not None and index < len(points):
                    polygon = [[float(x), float(y)] for x, y in points[index]]
                results.append(QRCodeResult(data=value, points=polygon))
            if results:
                return results
    except (cv2.error, ValueError, TypeError):
        pass

    try:
        data, points, _ = detector.detectAndDecode(image)
    except (cv2.error, ValueError, TypeError):
        return []

    data = (data or "").strip()
    if not data:
        return []

    polygon = None
    if points is not None:
        polygon = [[float(x), float(y)] for x, y in np.asarray(points).reshape(-1, 2)]

    return [QRCodeResult(data=data, points=polygon)]


def scan_qr_code(image: np.ndarray) -> QRCodeResult | None:
    results = scan_qr_codes(image)
    return results[0] if results else None
