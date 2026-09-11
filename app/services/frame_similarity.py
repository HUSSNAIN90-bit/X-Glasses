import cv2


def calculate_frame_difference(
    first_path: str,
    second_path: str,
) -> float:

    first = cv2.imread(
        first_path,
        cv2.IMREAD_GRAYSCALE,
    )

    second = cv2.imread(
        second_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if first is None or second is None:
        return 1.0

    first = cv2.resize(
        first,
        (320, 240),
    )

    second = cv2.resize(
        second,
        (320, 240),
    )

    difference = cv2.absdiff(
        first,
        second,
    )

    return float(
        difference.mean()
        / 255.0
    )


def are_frames_duplicates(
    first_path: str,
    second_path: str,
    threshold: float = 0.01,
) -> bool:

    difference = calculate_frame_difference(
        first_path,
        second_path,
    )

    return difference <= threshold