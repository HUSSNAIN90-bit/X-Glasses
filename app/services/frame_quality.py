import cv2


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

    blur_score = calculate_blur_score(
        image_path
    )

    brightness = calculate_brightness(
        image_path
    )

    if blur_score < blur_threshold:
        return False

    if brightness < min_brightness:
        return False

    if brightness > max_brightness:
        return False

    return True