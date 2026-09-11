from dataclasses import dataclass

from ultralytics import YOLO


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


tracker_model = YOLO(
    MODEL_NAME
)


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