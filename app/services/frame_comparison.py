from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.vision import (
    Detection,
    FrameAnalysis,
    PersonDetection,
)


# =========================================================
# MOVEMENT RESULT
# =========================================================

@dataclass(frozen=True)
class Movement:
    label: str
    from_position: str
    to_position: str
    distance: float
    direction: str


# =========================================================
# CENTER
# =========================================================

def calculate_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float]:

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    return center_x, center_y


# =========================================================
# EUCLIDEAN DISTANCE
# =========================================================

def calculate_distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:

    x1, y1 = point_a
    x2, y2 = point_b

    dx = x2 - x1
    dy = y2 - y1

    return float(
        (dx * dx + dy * dy) ** 0.5
    )


# =========================================================
# CAMERA MOTION
# =========================================================

def estimate_camera_motion(
    previous_path: str,
    current_path: str,
) -> tuple[float, float]:

    previous = cv2.imread(
        previous_path,
        cv2.IMREAD_GRAYSCALE,
    )

    current = cv2.imread(
        current_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if previous is None or current is None:
        return 0.0, 0.0

    orb = cv2.ORB_create(
        nfeatures=1000,
    )

    previous_keypoints, previous_descriptors = (
        orb.detectAndCompute(
            previous,
            None,
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
        return 0.0, 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True,
    )

    matches = matcher.match(
        previous_descriptors,
        current_descriptors,
    )

    if len(matches) < 8:
        return 0.0, 0.0

    matches = sorted(
        matches,
        key=lambda match: match.distance,
    )

    good_matches = matches[:100]

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

    matrix, _ = cv2.estimateAffinePartial2D(
        previous_points,
        current_points,
        method=cv2.RANSAC,
    )

    if matrix is None:
        return 0.0, 0.0

    dx = float(
        matrix[0, 2]
    )

    dy = float(
        matrix[1, 2]
    )

    return dx, dy


# =========================================================
# PERSON MATCHING
# =========================================================

def calculate_iou(
    first: PersonDetection,
    second: PersonDetection,
) -> float:

    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection_area = (
        intersection_width * intersection_height
    )

    first_area = (
        max(0.0, first.x2 - first.x1)
        * max(0.0, first.y2 - first.y1)
    )
    second_area = (
        max(0.0, second.x2 - second.x1)
        * max(0.0, second.y2 - second.y1)
    )
    union_area = (
        first_area + second_area - intersection_area
    )

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def classify_movement(
    dx: float,
    dy: float,
    threshold: float = 25.0,
) -> str:

    if abs(dx) < threshold and abs(dy) < threshold:
        return "stationary"

    if abs(dx) > abs(dy):
        if dx > 0:
            return "moving right"

        return "moving left"

    if dy > 0:
        return "moving down"

    return "moving up"

def find_matching_person(
    person: PersonDetection,
    candidates: list[PersonDetection],
) -> PersonDetection | None:

    # -----------------------------------------------------
    # RECOGNIZED PERSON: NAME IS THE BEST IDENTITY
    # -----------------------------------------------------

    if person.name is not None:

        same_name = [
            candidate
            for candidate in candidates
            if candidate.name is not None
            and candidate.name.lower() == person.name.lower()
        ]

        if same_name:
            return max(
                same_name,
                key=lambda candidate: calculate_iou(
                    person,
                    candidate,
                ),
            )

    # -----------------------------------------------------
    # UNKNOWN PERSON: USE BOX OVERLAP
    # -----------------------------------------------------

    unknown_candidates = [
        candidate
        for candidate in candidates
        if candidate.name is None
    ]

    if unknown_candidates:
        best_candidate: PersonDetection | None = None
        best_iou = 0.0

        for candidate in unknown_candidates:
            iou = calculate_iou(person, candidate)

            if iou > best_iou:
                best_iou = iou
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate

    # -----------------------------------------------------
    # FALLBACK: YOLO PERSON INDEX
    # -----------------------------------------------------

    if person.person_index is not None:

        for candidate in candidates:

            if (
                candidate.person_index
                == person.person_index
            ):
                return candidate

    return None


# =========================================================
# PERSON COMPARISON
# =========================================================

def compare_person(
    first: PersonDetection,
    second: PersonDetection,
    camera_dx: float = 0.0,
    camera_dy: float = 0.0,
) -> Movement:

    first_center = calculate_center(
        first.x1,
        first.y1,
        first.x2,
        first.y2,
    )

    second_center = calculate_center(
        second.x1,
        second.y1,
        second.x2,
        second.y2,
    )

    # -----------------------------------------------------
    # REMOVE GLOBAL CAMERA MOVEMENT
    # -----------------------------------------------------

    corrected_second_center = (
        second_center[0] - camera_dx,
        second_center[1] - camera_dy,
    )

    movement_distance = calculate_distance(
        first_center,
        corrected_second_center,
    )

    dx = corrected_second_center[0] - first_center[0]
    dy = corrected_second_center[1] - first_center[1]

    name = (
        second.name
        or first.name
        or "Unknown person"
    )

    return Movement(
        label=name,
        from_position=(
            first.relative_position
            or "unknown"
        ),
        to_position=(
            second.relative_position
            or "unknown"
        ),
        distance=movement_distance,
        direction=classify_movement(dx=dx, dy=dy),
    )


# =========================================================
# PEOPLE COMPARISON
# =========================================================

def compare_people(
    first: FrameAnalysis,
    second: FrameAnalysis,
    camera_dx: float = 0.0,
    camera_dy: float = 0.0,
) -> list[Movement]:

    movements: list[Movement] = []

    for person in first.people:

        match = find_matching_person(
            person=person,
            candidates=second.people,
        )

        if match is None:
            continue

        movement = compare_person(
            first=person,
            second=match,
            camera_dx=camera_dx,
            camera_dy=camera_dy,
        )

        movements.append(
            movement
        )

    return movements

# =========================================================
# OBJECT MATCHING
# =========================================================

def find_matching_object(
    detection: Detection,
    candidates: list[Detection],
) -> Detection | None:

    # People are not objects here.
    same_class = [
        candidate
        for candidate in candidates
        if candidate.class_name
        == detection.class_name
        and candidate.class_name != "person"
    ]

    if not same_class:
        return None

    first_center = calculate_center(
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
    )

    best_candidate: Detection | None = None
    best_distance: float | None = None

    for candidate in same_class:

        candidate_center = calculate_center(
            candidate.x1,
            candidate.y1,
            candidate.x2,
            candidate.y2,
        )

        distance = calculate_distance(
            first_center,
            candidate_center,
        )

        if (
            best_distance is None
            or distance < best_distance
        ):
            best_distance = distance
            best_candidate = candidate

    return best_candidate


# =========================================================
# OBJECT COMPARISON
# =========================================================

def compare_objects(
    first: FrameAnalysis,
    second: FrameAnalysis,
    camera_dx: float = 0.0,
    camera_dy: float = 0.0,
) -> list[Movement]:

    movements: list[Movement] = []

    for detection in first.objects:

        # -------------------------------------------------
        # PERSONS ARE HANDLED BY compare_people()
        # -------------------------------------------------

        if detection.class_name == "person":
            continue

        match = find_matching_object(
            detection=detection,
            candidates=second.objects,
        )

        if match is None:
            continue

        first_center = calculate_center(
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        )

        second_center = calculate_center(
            match.x1,
            match.y1,
            match.x2,
            match.y2,
        )

        # -------------------------------------------------
        # CAMERA MOTION COMPENSATION
        # -------------------------------------------------

        corrected_second_center = (
            second_center[0] - camera_dx,
            second_center[1] - camera_dy,
        )

        movement_distance = calculate_distance(
            first_center,
            corrected_second_center,
        )

        dx = corrected_second_center[0] - first_center[0]
        dy = corrected_second_center[1] - first_center[1]

        movements.append(
            Movement(
                label=detection.class_name,
                from_position="visible",
                to_position="visible",
                distance=movement_distance,
                direction=classify_movement(dx=dx, dy=dy),
            )
        )

    return movements