from dataclasses import dataclass

from ultralytics import YOLO
from dataclasses import dataclass

from app.services.pose import PoseDetection, detect_pose



MODEL_NAME: str = "yolo26n.pt"


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float




@dataclass
class TrackedPoseDetection:
    track_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    posture: str | None
    pose_confidence: float | None
    keypoints: list[list[float]] | None


def calculate_bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
        return 0.0

    intersection_area = (
        (intersection_x2 - intersection_x1)
        * (intersection_y2 - intersection_y1)
    )

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

tracker_model = YOLO(
    MODEL_NAME
)


def track_pose_frames(
    image_paths: list[str],
    confidence_threshold: float = 0.40,
) -> list[list[TrackedPoseDetection]]:

    tracked_frames = track_frames(
        image_paths=image_paths,
        confidence_threshold=confidence_threshold,
    )

    all_results: list[list[TrackedPoseDetection]] = []

    for frame_index, detections in enumerate(tracked_frames):

        image_path = image_paths[frame_index]

        poses: list[PoseDetection] = detect_pose(image_path)

        frame_results: list[TrackedPoseDetection] = []

        for detection in detections:

            if detection.class_name != "person":
                frame_results.append(
                    TrackedPoseDetection(
                        track_id=detection.track_id,
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                        x1=detection.x1,
                        y1=detection.y1,
                        x2=detection.x2,
                        y2=detection.y2,
                        posture=None,
                        pose_confidence=None,
                        keypoints=None,
                    )
                )
                continue

            detection_box = (
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
            )

            best_pose: PoseDetection | None = None
            best_iou = 0.0

            for pose in poses:

                pose_box = (
                    pose.bbox[0],
                    pose.bbox[1],
                    pose.bbox[2],
                    pose.bbox[3],
                )

                iou = calculate_bbox_iou(
                    detection_box,
                    pose_box,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_pose = pose

            frame_results.append(
                TrackedPoseDetection(
                    track_id=detection.track_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                    posture=(
                        best_pose.posture
                        if best_pose is not None
                        else None
                    ),
                    pose_confidence=(
                        best_pose.confidence
                        if best_pose is not None
                        else None
                    ),
                    keypoints=(
                        best_pose.keypoints
                        if best_pose is not None
                        else None
                    ),
                )
            )

        all_results.append(frame_results)

    return all_results

def track_frames(
    image_paths: list[str],
    confidence_threshold: float = 0.40,
) -> list[list[TrackedDetection]]:

    all_frames: list[
        list[TrackedDetection]
    ] = []

    for image_path in image_paths:

        results = tracker_model.track(
            source=image_path,
            conf=confidence_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        frame_detections: list[
            TrackedDetection
        ] = []

        for result in results:

            if result.boxes is None:
                continue

            if result.boxes.id is None:
                continue

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            classes = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )

            boxes = (
                result.boxes.xyxy
                .cpu()
                .tolist()
            )

            for (
                track_id,
                class_id,
                confidence,
                coordinates,
            ) in zip(
                track_ids,
                classes,
                confidences,
                boxes,
            ):

                class_name = (
                    result.names[class_id]
                )

                x1, y1, x2, y2 = (
                    coordinates
                )

                frame_detections.append(
                    TrackedDetection(
                        track_id=int(
                            track_id
                        ),
                        class_name=class_name,
                        confidence=float(
                            confidence
                        ),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    )
                )

        all_frames.append(
            frame_detections
        )

    return all_frames