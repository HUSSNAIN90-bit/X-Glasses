from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class DetectedCode:
    format: str
    value: str


def detect_codes(image_path: str) -> list[DetectedCode]:
    """Decode QR codes and supported linear barcodes without an LLM."""
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise ValueError("Unable to read image for code detection.")

    detected: list[DetectedCode] = []
    seen_values: set[str] = set()

    barcode_detector = cv2.barcode_BarcodeDetector()
    found, values, code_types, _ = barcode_detector.detectAndDecodeWithType(image)
    if found:
        for value, code_type in zip(values, code_types):
            value = value.strip()
            if value and value not in seen_values:
                detected.append(
                    DetectedCode(format=code_type or "BARCODE", value=value)
                )
                seen_values.add(value)

    qr_detector = cv2.QRCodeDetector()
    found, values, _, _ = qr_detector.detectAndDecodeMulti(image)
    if found:
        for value in values:
            value = value.strip()
            if value and value not in seen_values:
                detected.append(DetectedCode(format="QR_CODE", value=value))
                seen_values.add(value)

    return detected
