from __future__ import annotations

from dataclasses import dataclass

from app.schemas.vision import Detection, PersonDetection


@dataclass
class ObjectRelationship:
    subject: str
    relation: str
    object: str
    confidence: float


HANDHELD_OBJECTS = {
    "cell phone",
    "phone",
    "remote",
    "bottle",
    "cup",
    "book",
    "camera",
}


def calculate_center(
    bbox: list[float],
) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def is_inside(
    inner_bbox: list[float],
    outer_bbox: list[float],
) -> bool:
    ox1, oy1, ox2, oy2 = outer_bbox

    center_x, center_y = calculate_center(inner_bbox)

    return (
        ox1 <= center_x <= ox2
        and
        oy1 <= center_y <= oy2
    )
    
def object_near_person(
    person: PersonDetection,
    obj: Detection,
    distance_ratio: float = 0.35,
) -> bool:
    person_x1, person_y1, person_x2, person_y2 = person.bbox
    object_x, object_y = calculate_center(obj.bbox)

    person_width = person_x2 - person_x1
    person_height = person_y2 - person_y1

    if person_width <= 0 or person_height <= 0:
        return False

    expanded_x1 = person_x1 - person_width * distance_ratio
    expanded_x2 = person_x2 + person_width * distance_ratio

    expanded_y1 = person_y1 - person_height * distance_ratio
    expanded_y2 = person_y2 + person_height * distance_ratio

    return (
        expanded_x1 <= object_x <= expanded_x2
        and
        expanded_y1 <= object_y <= expanded_y2
    )


def detect_relationships(
    people: list[PersonDetection],
    objects: list[Detection],
) -> list[ObjectRelationship]:

    relationships: list[ObjectRelationship] = []

    for person in people:
        for obj in objects:

            if obj.class_name not in HANDHELD_OBJECTS:
                continue

            if not object_near_person(person, obj):
                continue

            relationships.append(
                ObjectRelationship(
                    subject=f"person_{person.person_index}",
                    relation="near",
                    object=obj.class_name,
                    confidence=min(
                        person.confidence,
                        obj.confidence,
                    ),
                )
            )

    return relationships

def get_hand_points(
    keypoints: list[list[float]],
) -> list[tuple[float, float]]:
    hands: list[tuple[float, float]] = []

    # COCO: 9 = left wrist, 10 = right wrist
    for index in (9, 10):
        if index >= len(keypoints):
            continue

        x, y = keypoints[index]

        if x <= 0 and y <= 0:
            continue

        hands.append((x, y))

    return hands

def point_near_bbox(
    point: tuple[float, float],
    bbox: list[float],
    margin: float = 40.0,
) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox

    return (
        x1 - margin <= x <= x2 + margin
        and
        y1 - margin <= y <= y2 + margin
    )
    
def is_hand_near_object(
    keypoints: list[list[float]],
    object_bbox: list[float],
) -> bool:
    hands = get_hand_points(keypoints)

    return any(
        point_near_bbox(hand, object_bbox)
        for hand in hands
    )