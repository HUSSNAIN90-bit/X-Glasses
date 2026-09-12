from pathlib import Path
from typing import Any

from ultralytics import YOLO

from app.schemas.vision import Detection


MODEL_PATH = Path("models/yolo26n.pt")

model = YOLO(str(MODEL_PATH))


def detect_objects(
    image_path: str,
    confidence_threshold: float = 0.25,
) -> list[Detection]:


    results: list[Any] = model.predict(
        source=image_path,
        conf=confidence_threshold,
        verbose=False,
    )

    detections: list[Detection] = []

    for result in results:

        if result.boxes is None:
            continue

        # ---------------------------------------------
        # IMAGE SIZE
        # ---------------------------------------------

        image_height: float = float(result.orig_shape[0])
        image_width: float = float(result.orig_shape[1])

        for box in result.boxes:

            class_id: int = int(
                box.cls.item()
            )

            confidence: float = float(
                box.conf.item()
            )

            coordinates: list[float] = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]

            x1: float = float(
                coordinates[0]
            )

            y1: float = float(
                coordinates[1]
            )

            x2: float = float(
                coordinates[2]
            )

            y2: float = float(
                coordinates[3]
            )

            class_name: str = str(
                result.names[class_id]
            )

            # -----------------------------------------
            # OBJECT POSITION
            # -----------------------------------------

            relative_position = (
                get_relative_position(
                    box=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    image_width=image_width,
                )
            )

            vertical_position = (
                get_relative_vertical_position(
                    face_box=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    image_height=float(
                        image_height
                    ),
                )
            )

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    relative_position=(
                        relative_position
                    ),
                    vertical_position=(
                        vertical_position
                    ),
                )
            )

    return detections

def calculate_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area == 0.0:
        return 0.0

    return intersection_area / union_area

def is_face_inside_person(
    face_box: tuple[float, float, float, float],
    person_box: tuple[float, float, float, float],
) -> bool:

    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box

    face_center_x = (
        fx1 + fx2
    ) / 2

    face_center_y = (
        fy1 + fy2
    ) / 2

    return (
        px1 <= face_center_x <= px2
        and
        py1 <= face_center_y <= py2
    )

def get_relative_position(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
    image_width: float,
) -> str:

    x1, _, x2, _ = box

    center_x = (
        x1 + x2
    ) / 2

    normalized_x = (
        center_x / image_width
    )

    if normalized_x < 0.33:
        return "left"

    if normalized_x > 0.66:
        return "right"

    return "center"

def get_relative_vertical_position(
    face_box: tuple[
        float,
        float,
        float,
        float,
    ],
    image_height: float,
) -> str:

    _, y1, _, y2 = face_box

    center_y = (
        y1 + y2
    ) / 2

    normalized_y = (
        center_y / image_height
    )

    if normalized_y < 0.35:
        return "above"

    if normalized_y > 0.65:
        return "below"

    return "center"

def describe_person_position(
    x1: float,
    x2: float,
    image_width: float,
) -> str:

    center_x = (
        x1 + x2
    ) / 2

    normalized_x = (
        center_x / image_width
    )

    if normalized_x < 0.20:
        return "far to your left"

    if normalized_x < 0.40:
        return "slightly to your left"

    if normalized_x <= 0.60:
        return "in front of you"

    if normalized_x <= 0.80:
        return "slightly to your right"

    return "far to your right"