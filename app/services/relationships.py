from __future__ import annotations

from dataclasses import dataclass

from app.schemas.vision import Detection, PersonDetection

from dataclasses import dataclass
from typing import Optional

@dataclass
class ObjectRelationship:
    subject: str
    relation: str
    object: str
    confidence: float
@dataclass
class HoldingState:
    holding: bool = False
    missed_frames: int = 0
    last_confidence: float = 0.0

HANDHELD_OBJECTS = {
    "cell phone",
    "phone",
    "remote",
    "bottle",
    "cup",
    "book",
    "camera",
}

def adaptive_hand_near_object(
    keypoints: list[list[float]],
    object_bbox: list[float],
    person_bbox: list[float],
    distance_ratio: float = 0.25,
) -> bool:
    """
    Check whether either wrist is close to an object.

    Distance threshold scales with the person's height,
    making the check more robust across different image sizes.
    """

    if len(person_bbox) != 4:
        return False

    person_height = person_bbox[3] - person_bbox[1]

    if person_height <= 0:
        return False

    threshold = person_height * distance_ratio

    hands = get_hand_points(keypoints)

    object_x, object_y = calculate_center(object_bbox)

    for hand_x, hand_y in hands:

        distance = (
            (hand_x - object_x) ** 2
            + (hand_y - object_y) ** 2
        ) ** 0.5

        if distance <= threshold:
            return True

    return False

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
    
def update_holding_state(
    state: HoldingState,
    detected_near_hand: bool,
    max_missed_frames: int = 3,
    confidence: float = 0.0,
) -> HoldingState:

    if detected_near_hand:
        state.holding = True
        state.missed_frames = 0

        if confidence > 0:
            state.last_confidence = confidence

        return state

    if not state.holding:
        return state

    state.missed_frames += 1

    if state.missed_frames > max_missed_frames:
        state.holding = False
        state.missed_frames = 0
        state.last_confidence = 0.0

    return state

def select_best_objects(
    objects: list[Detection],
    min_confidence: float = 0.10,
) -> list[Detection]:
    best_by_class: dict[str, Detection] = {}

    for obj in objects:
        if obj.class_name not in HANDHELD_OBJECTS:
            continue

        if obj.confidence < min_confidence:
            continue

        current = best_by_class.get(obj.class_name)

        if current is None or obj.confidence > current.confidence:
            best_by_class[obj.class_name] = obj

    return list(best_by_class.values())

def detect_holding_relationships(
    person: PersonDetection,
    keypoints: list[list[float]],
    objects: list[Detection],
    states: dict[tuple[int, str], HoldingState],
    min_confidence: float = 0.10,
    max_missed_frames: int = 3,
) -> list[ObjectRelationship]:

    relationships: list[ObjectRelationship] = []

    selected_objects = select_best_objects(
        objects=objects,
        min_confidence=min_confidence,
    )

    for obj in selected_objects:

        if obj.class_name not in HANDHELD_OBJECTS:
            continue

        near_hand = adaptive_hand_near_object(
            keypoints=keypoints,
            object_bbox=obj.bbox,
            person_bbox=[
                person.x1,
                person.y1,
                person.x2,
                person.y2,
            ],
        )

        state_key = (
            person.person_index,
            obj.class_name,
        )

        state = states.setdefault(
            state_key,
            HoldingState(),
        )

        update_holding_state(
            state=state,
            detected_near_hand=near_hand,
            max_missed_frames=max_missed_frames,
            confidence=obj.confidence,
        )

        if state.holding:
            relationships.append(
                ObjectRelationship(
                    subject=f"person_{person.person_index}",
                    relation="holding",
                    object=obj.class_name,
                    confidence=state.last_confidence,
                )
            )

    return relationships

def get_memory_holding_relationships(
    person: PersonDetection,
    states: dict[tuple[int, str], HoldingState],
) -> list[ObjectRelationship]:

    relationships: list[ObjectRelationship] = []

    for (person_index, class_name), state in states.items():

        if person_index != person.person_index:
            continue

        if not state.holding:
            continue

        relationships.append(
            ObjectRelationship(
                subject=f"person_{person.person_index}",
                relation="holding",
                object=class_name,
                confidence=state.last_confidence,
            )
        )

    return relationships