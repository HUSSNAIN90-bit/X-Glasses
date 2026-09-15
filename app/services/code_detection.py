from dataclasses import dataclass

import cv2

from app.services.product_lookup import lookup_product


@dataclass(frozen=True)
class DetectedProduct:
    name: str | None
    brand: str | None
    category: str | None
    source: str | None
    prices: list[dict]
    price_note: str | None


@dataclass(frozen=True)
class DetectedCode:
    format: str
    value: str
    product: DetectedProduct | None = None


def _product_for_barcode(value: str) -> DetectedProduct | None:
    normalized = "".join(ch for ch in value if ch.isdigit())
    if len(normalized) not in {8, 10, 12, 13, 14}:
        return None

    product = lookup_product(normalized)
    if not product.success:
        return None

    prices = [
        {"value": p.value, "currency": p.currency, "source": p.source, "kind": p.kind}
        for p in (product.prices or [])
    ]
    price_note = None
    lowest = [p["value"] for p in prices if p["kind"] == "lowest_recorded_offer"]
    highest = [p["value"] for p in prices if p["kind"] == "highest_recorded_offer"]
    if lowest and highest:
        currency = next((p["currency"] for p in prices if p["currency"]), None)
        suffix = f" {currency}" if currency else ""
        price_note = f"Recorded offers range from {lowest[0]:.2f}{suffix} to {highest[0]:.2f}{suffix}."

    return DetectedProduct(
        name=product.name,
        brand=product.brand,
        category=product.category,
        source=product.source,
        prices=prices,
        price_note=price_note,
    )


def _variants(image):
    """Create local barcode/QR-friendly variants for small or difficult codes."""
    variants = [image]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    height, width = gray.shape[:2]
    scale = min(2.5, 1600 / max(width, 1)) if width < 1600 else 1.5
    variants.append(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(enhanced)
    variants.append(cv2.convertScaleAbs(enhanced, alpha=1.35, beta=0))
    variants.append(
        cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
        )
    )
    return variants


def _decode_variant(image, seen_values: set[str]) -> list[DetectedCode]:
    found_codes: list[DetectedCode] = []

    try:
        detector = cv2.barcode_BarcodeDetector()
        found, values, code_types, _ = detector.detectAndDecodeWithType(image)
        if found:
            for value, code_type in zip(values, code_types):
                value = (value or "").strip()
                if value and value not in seen_values:
                    found_codes.append(
                        DetectedCode(
                            format=code_type or "BARCODE",
                            value=value,
                            product=_product_for_barcode(value),
                        )
                    )
                    seen_values.add(value)
    except Exception:
        pass

    try:
        qr = cv2.QRCodeDetector()
        found, values, _, _ = qr.detectAndDecodeMulti(image)
        if found and values is not None:
            for value in values:
                value = (value or "").strip()
                if value and value not in seen_values:
                    found_codes.append(DetectedCode(format="QR_CODE", value=value))
                    seen_values.add(value)
        else:
            value, _, _ = qr.detectAndDecode(image)
            value = (value or "").strip()
            if value and value not in seen_values:
                found_codes.append(DetectedCode(format="QR_CODE", value=value))
                seen_values.add(value)
    except Exception:
        pass

    return found_codes


def detect_codes(image_path: str) -> list[DetectedCode]:
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise ValueError("Unable to read image for code detection.")

    detected: list[DetectedCode] = []
    seen_values: set[str] = set()
    for variant in _variants(image):
        detected.extend(_decode_variant(variant, seen_values))
        if detected:
            break
    return detected


def detect_codes_from_frames(image_paths: list[str]) -> list[DetectedCode]:
    """Search the whole short camera burst and merge decoded values."""
    detected: list[DetectedCode] = []
    seen_values: set[str] = set()
    for path in image_paths:
        try:
            image = cv2.imread(path)
            if image is None or image.size == 0:
                continue
            for variant in _variants(image):
                new_codes = _decode_variant(variant, seen_values)
                detected.extend(new_codes)
                if new_codes:
                    break
        except Exception:
            continue
    return detected
