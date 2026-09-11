from dataclasses import dataclass

from app.schemas.vision import (
    Detection,
    FrameAnalysis,
    PersonDetection,
)


@dataclass(frozen=True)
class Movement:
    label: str
    from_position: str
    to_position: str
    distance: float


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


def calculate_distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:

    x1, y1 = point_a
    x2, y2 = point_b

    dx = x2 - x1
    dy = y2 - y1

    return (
        (dx * dx) +
        (dy * dy)
    ) ** 0.5


def compare_person(
    first: PersonDetection,
    second: PersonDetection,
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

    movement_distance = calculate_distance(
        first_center,
        second_center,
    )

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
    )


def find_matching_person(
    person: PersonDetection,
    candidates: list[PersonDetection],
) -> PersonDetection | None:

    # ---------------------------------------------
    # BEST MATCH: SAME RECOGNIZED NAME
    # ---------------------------------------------

    if person.name is not None:

        for candidate in candidates:

            if candidate.name is None:
                continue

            if (
                candidate.name.lower()
                == person.name.lower()
            ):
                return candidate

    # ---------------------------------------------
    # FALLBACK: SAME PERSON INDEX
    # ---------------------------------------------

    if person.person_index is not None:

        for candidate in candidates:

            if (
                candidate.person_index
                == person.person_index
            ):
                return candidate

    return None


def compare_people(
    first: FrameAnalysis,
    second: FrameAnalysis,
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
        )

        movements.append(
            movement
        )

    return movements


def find_matching_object(
    detection: Detection,
    candidates: list[Detection],
) -> Detection | None:

    same_class = [
        candidate
        for candidate in candidates
        if candidate.class_name
        == detection.class_name
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


def compare_objects(
    first: FrameAnalysis,
    second: FrameAnalysis,
) -> list[Movement]:

    movements: list[Movement] = []

    for detection in first.objects:

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

        movement_distance = calculate_distance(
            first_center,
            second_center,
        )

        movements.append(
            Movement(
                label=detection.class_name,
                from_position="visible",
                to_position="visible",
                distance=movement_distance,
            )
        )

    return movements