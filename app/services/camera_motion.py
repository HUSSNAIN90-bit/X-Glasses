import cv2
import numpy as np


def estimate_camera_motion(
    previous_path: str,
    current_path: str,
    excluded_boxes: list[
        tuple[float, float, float, float]
    ] | None = None,
) -> tuple[float, float, bool]:

    previous = cv2.imread(
        previous_path,
        cv2.IMREAD_GRAYSCALE,
    )

    current = cv2.imread(
        current_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if previous is None or current is None:
        return 0.0, 0.0, False

    # -----------------------------------------------------
    # MASK DYNAMIC REGIONS
    # -----------------------------------------------------

    mask = np.full(
        previous.shape,
        255,
        dtype=np.uint8,
    )

    if excluded_boxes:

        for x1, y1, x2, y2 in excluded_boxes:

            cv2.rectangle(
                mask,
                (
                    max(0, int(x1)),
                    max(0, int(y1)),
                ),
                (
                    min(
                        previous.shape[1],
                        int(x2),
                    ),
                    min(
                        previous.shape[0],
                        int(y2),
                    ),
                ),
                0,
                -1,
            )

    # -----------------------------------------------------
    # ORB FEATURES
    # -----------------------------------------------------

    orb = cv2.ORB_create(
        nfeatures=2000,
    )

    previous_keypoints, previous_descriptors = (
        orb.detectAndCompute(
            previous,
            mask,
        )
    )

    current_keypoints, current_descriptors = (
        orb.detectAndCompute(
            current,
            None,
        )
    )

    if (
        previous_descriptors is None
        or current_descriptors is None
    ):
        return 0.0, 0.0, False

    if len(previous_keypoints) < 10:
        return 0.0, 0.0, False

    # -----------------------------------------------------
    # MATCH FEATURES
    # -----------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True,
    )

    matches = matcher.match(
        previous_descriptors,
        current_descriptors,
    )

    if len(matches) < 10:
        return 0.0, 0.0, False

    matches = sorted(
        matches,
        key=lambda match: match.distance,
    )

    good_matches = matches[:200]

    previous_points = np.float32(
        [
            previous_keypoints[
                match.queryIdx
            ].pt
            for match in good_matches
        ]
    )

    current_points = np.float32(
        [
            current_keypoints[
                match.trainIdx
            ].pt
            for match in good_matches
        ]
    )

    if len(previous_points) < 8:
        return 0.0, 0.0, False

    # -----------------------------------------------------
    # ROBUST GLOBAL MOTION
    # -----------------------------------------------------

    matrix, inliers = (
        cv2.estimateAffinePartial2D(
            previous_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
    )

    if matrix is None or inliers is None:
        return 0.0, 0.0, False

    inlier_count = int(
        inliers.sum()
    )

    total_matches = len(
        good_matches
    )

    inlier_ratio = (
        inlier_count
        / total_matches
    )

    # -----------------------------------------------------
    # REQUIRE RELIABLE BACKGROUND MOTION
    # -----------------------------------------------------

    if inlier_count < 8:
        return 0.0, 0.0, False

    if inlier_ratio < 0.35:
        return 0.0, 0.0, False

    dx = float(
        matrix[0, 2]
    )

    dy = float(
        matrix[1, 2]
    )

    return dx, dy, True