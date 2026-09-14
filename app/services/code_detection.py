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
    # QR payloads can be arbitrary text/URLs; only numeric product codes are
    # sent to product databases to avoid unnecessary external requests.
    normalized = "".join(ch for ch in value if ch.isdigit())
    if len(normalized) not in {8, 10, 12, 13, 14}:
        return None

    product = lookup_product(normalized)
    if not product.success:
        return None

    prices = [
        {
            "value": price.value,
            "currency": price.currency,
            "source": price.source,
            "kind": price.kind,
        }
        for price in (product.prices or [])
    ]

    price_note = None
    if prices:
        lowest = [item["value"] for item in prices if item["kind"] == "lowest_recorded_offer"]
        highest = [item["value"] for item in prices if item["kind"] == "highest_recorded_offer"]
        if lowest and highest:
            currency = next((item["currency"] for item in prices if item["currency"]), None)
            symbol = f" {currency}" if currency else ""
            price_note = f"Recorded offers range from {lowest[0]:.2f}{symbol} to {highest[0]:.2f}{symbol}."

    return DetectedProduct(
        name=product.name,
        brand=product.brand,
        category=product.category,
        source=product.source,
        prices=prices,
        price_note=price_note,
    )


def detect_codes(image_path: str) -> list[DetectedCode]:
    """Decode QR/linear codes and enrich product barcodes when possible."""
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
                    DetectedCode(
                        format=code_type or "BARCODE",
                        value=value,
                        product=_product_for_barcode(value),
                    )
                )
                seen_values.add(value)

    qr_detector = cv2.QRCodeDetector()
    found, values, _, _ = qr_detector.detectAndDecodeMulti(image)
    if found:
        for value in values:
            value = value.strip()
            if value and value not in seen_values:
                detected.append(
                    DetectedCode(format="QR_CODE", value=value)
                )
                seen_values.add(value)

    return detected
