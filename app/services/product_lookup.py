from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import requests


UPCITEMDB_URL = "https://api.upcitemdb.com/prod/trial/lookup"
OPENFOODFACTS_URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
REQUEST_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ProductPrice:
    value: float
    currency: str | None
    source: str
    kind: str


@dataclass(frozen=True)
class ProductLookupResult:
    success: bool
    code: str
    name: str | None = None
    brand: str | None = None
    category: str | None = None
    source: str | None = None
    prices: list[ProductPrice] | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["prices"] = [asdict(item) for item in (self.prices or [])]
        return result


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


def _upc_lookup(code: str) -> ProductLookupResult | None:
    try:
        response = requests.get(
            UPCITEMDB_URL,
            params={"upc": code},
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "X-Glasses/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    items = payload.get("items") or []
    if not items:
        return None

    item = items[0]
    prices: list[ProductPrice] = []
    for offer in item.get("offers") or []:
        value = _number(offer.get("price"))
        if value is None:
            continue
        prices.append(
            ProductPrice(
                value=value,
                currency=offer.get("currency"),
                source=str(offer.get("merchant") or "UPCitemdb"),
                kind="offer",
            )
        )

    if prices:
        values = [price.value for price in prices]
        currency = next((p.currency for p in prices if p.currency), None)
        prices.append(ProductPrice(min(values), currency, "UPCitemdb", "lowest_recorded_offer"))
        prices.append(ProductPrice(max(values), currency, "UPCitemdb", "highest_recorded_offer"))

    return ProductLookupResult(
        success=True,
        code=code,
        name=item.get("title"),
        brand=item.get("brand"),
        category=item.get("category"),
        source="UPCitemdb",
        prices=prices,
    )


def _off_lookup(code: str) -> ProductLookupResult | None:
    try:
        response = requests.get(
            OPENFOODFACTS_URL.format(code=code),
            params={"fields": "code,product_name,brands,categories,product_type"},
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "X-Glasses/1.0 (barcode lookup)"},
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return None

    if payload.get("status") != 1:
        return None

    product = payload.get("product") or {}
    return ProductLookupResult(
        success=True,
        code=code,
        name=product.get("product_name"),
        brand=product.get("brands"),
        category=product.get("categories"),
        source="Open Food Facts",
        prices=[],
    )


@lru_cache(maxsize=256)
def lookup_product(code: str) -> ProductLookupResult:
    """Look up a decoded UPC/EAN/GTIN/ISBN without inventing product data."""
    normalized = "".join(ch for ch in str(code).strip() if ch.isdigit() or ch in "Xx")
    if not normalized:
        return ProductLookupResult(False, str(code), message="Invalid barcode")

    result = _upc_lookup(normalized)
    if result is not None:
        return result

    result = _off_lookup(normalized)
    if result is not None:
        return result

    return ProductLookupResult(
        success=False,
        code=normalized,
        message="Product not found",
    )
