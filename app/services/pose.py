from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ultralytics import YOLO

from app.schemas.vision import PoseDetection


MODEL_PATH = Path("models/yolo11n-pose.pt")

model = YOLO(str(MODEL_PATH))


def calculate_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

def detect_pose(
    image_path: str,
    confidence_threshold: float = 0.40,
) -> list[PoseDetection]:

    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        verbose=False,
    )

    poses: list[PoseDetection] = []

    for result in results:

        if result.keypoints is None:
            continue

        if result.boxes is None:
            continue

        boxes = result.boxes
        keypoints = result.keypoints

        for person_index in range(len(boxes)):

            confidence = float(
                boxes.conf[person_index].item()
            )

            x1, y1, x2, y2 = (
                boxes.xyxy[person_index]
                .tolist()
            )

            posture = classify_posture(
                result.keypoints.xy[person_index]
            )

            keypoints = (
                result.keypoints.xy[person_index]
                .cpu()
                .tolist()
            )

            poses.append(
                PoseDetection(
                    person_index=person_index,
                    posture=posture,
                    confidence=confidence,
                    bbox=[
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    keypoints=keypoints,
                )
            )

    return poses


def classify_posture(
    keypoints,
) -> str:

    # YOLO pose keypoints:
    # 5 = left shoulder
    # 6 = right shoulder
    # 11 = left hip
    # 12 = right hip
    # 15 = left ankle
    # 16 = right ankle

    shoulder_y = (
        float(keypoints[5][1])
        + float(keypoints[6][1])
    ) / 2

    hip_y = (
        float(keypoints[11][1])
        + float(keypoints[12][1])
    ) / 2

    ankle_y = (
        float(keypoints[15][1])
        + float(keypoints[16][1])
    ) / 2

    torso_height = hip_y - shoulder_y
    leg_height = ankle_y - hip_y

    if torso_height <= 0:
        return "unknown"

    ratio = leg_height / torso_height

    if ratio > 0.8:
        return "standing"

    if ratio < 0.45:
        return "sitting"

    return "unknown"

def match_pose_to_person(
    person_bbox: Sequence[float],
    poses: list[PoseDetection],
    iou_threshold: float = 0.5,
) -> PoseDetection | None:
    best_pose: PoseDetection | None = None
    best_iou = 0.0

    for pose in poses:
        iou = calculate_iou(person_bbox, pose.bbox)

        if iou > best_iou:
            best_iou = iou
            best_pose = pose

    if best_iou >= iou_threshold:
        return best_pose

    return None