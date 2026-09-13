from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQualityMetrics:
    good: bool
    blur_score: float
    brightness: float
    width: int
    height: int
    exposure_score: float
    quality_score: float
    reason: str | None = None


def evaluate_frame_quality(
    image_path: str,
    blur_threshold: float = 80.0,
    min_brightness: float = 35.0,
    max_brightness: float = 220.0,
    min_width: int = 320,
    min_height: int = 240,
) -> FrameQualityMetrics:
    image = cv2.imread(image_path)

    if image is None or image.size == 0:
        return FrameQualityMetrics(
            good=False,
            blur_score=0.0,
            brightness=0.0,
            width=0,
            height=0,
            exposure_score=0.0,
            quality_score=0.0,
            reason="invalid_image",
        )

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    clipped_ratio = float(
        np.mean(gray <= 5) + np.mean(gray >= 250)
    )
    exposure_score = max(0.0, 1.0 - min(1.0, clipped_ratio / 0.65))
    brightness_score = max(
        0.0,
        1.0 - abs(brightness - 127.5) / 127.5,
    )
    sharpness_score = min(1.0, blur_score / max(blur_threshold * 3.0, 1.0))
    dimension_score = min(
        1.0,
        min(width / max(min_width, 1), height / max(min_height, 1)),
    )
    quality_score = float(
        0.50 * sharpness_score
        + 0.25 * brightness_score
        + 0.15 * exposure_score
        + 0.10 * dimension_score
    )

    reason: str | None = None
    if width < min_width or height < min_height:
        reason = "dimensions_too_small"
    elif blur_score < blur_threshold:
        reason = "too_blurry"
    elif brightness < min_brightness:
        reason = "too_dark"
    elif brightness > max_brightness:
        reason = "too_bright"
    elif clipped_ratio > 0.80:
        reason = "poor_exposure"

    return FrameQualityMetrics(
        good=reason is None,
        blur_score=blur_score,
        brightness=brightness,
        width=int(width),
        height=int(height),
        exposure_score=exposure_score,
        quality_score=max(0.0, min(1.0, quality_score)),
        reason=reason,
    )


def calculate_blur_score(
    image_path: str,
) -> float:

    image = cv2.imread(
        image_path
    )

    if image is None:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    return float(variance)


def calculate_brightness(
    image_path: str,
) -> float:

    image = cv2.imread(
        image_path
    )

    if image is None:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return float(gray.mean())


def is_frame_good(
    image_path: str,
    blur_threshold: float = 80.0,
    min_brightness: float = 35.0,
    max_brightness: float = 220.0,
) -> bool:
    return evaluate_frame_quality(
        image_path=image_path,
        blur_threshold=blur_threshold,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
    ).good


def optimize_image_for_vision(
    image_path: str,
    max_dimension: int = 1800,
    jpeg_quality: int = 88,
) -> bytes:
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise ValueError("Unable to read selected image.")

    height, width = image.shape[:2]
    largest_dimension = max(width, height)
    if largest_dimension > max_dimension:
        scale = max_dimension / float(largest_dimension)
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    encoded, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not encoded:
        raise ValueError("Unable to optimize selected image.")
    return buffer.tobytes()
