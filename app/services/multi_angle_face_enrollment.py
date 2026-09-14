"""Automatic passive multi-angle face enrollment for X-Glasses.

This module is intentionally separate from the normal one-frame enrollment flow.
It captures a small set of diverse, quality-gated face embeddings from a camera
without asking the person to follow voice/visual angle instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time

import cv2
import numpy as np

from app.services.face_database import (
    add_embedding_to_person,
    create_person,
    get_person_by_name,
    update_person_relationship,
)
from app.services.face_recognition import face_app, normalize_embedding


TARGET_SAMPLES = 7
MAX_SAMPLES = 10
MAX_DURATION_SECONDS = 30.0
COOLDOWN_SECONDS = 0.75
DUPLICATE_SIMILARITY = 0.985
MIN_DETECTION_CONFIDENCE = 0.55
MIN_FACE_SIZE = 90
MIN_BLUR_SCORE = 45.0
MIN_BRIGHTNESS = 45.0
MAX_BRIGHTNESS = 220.0

# Keep the gallery diverse instead of filling it with repeated frames from one
# easy angle. Two samples per bucket is enough to retain a little redundancy.
MAX_SAMPLES_PER_BUCKET = 2

# Approximate InsightFace pose ranges. The model exposes pose as three angles;
# this module uses the first two as horizontal/vertical orientation signals.
ANGLE_BUCKETS = {
    "front": ("yaw", -12.0, 12.0, "pitch", -12.0, 12.0),
    "slight_left": ("yaw", -35.0, -12.0, "pitch", -20.0, 20.0),
    "left": ("yaw", -90.0, -35.0, "pitch", -25.0, 25.0),
    "slight_right": ("yaw", 12.0, 35.0, "pitch", -20.0, 20.0),
    "right": ("yaw", 35.0, 90.0, "pitch", -25.0, 25.0),
    "up": ("yaw", -35.0, 35.0, "pitch", -90.0, -18.0),
    "down": ("yaw", -35.0, 35.0, "pitch", 18.0, 90.0),
}


@dataclass
class CapturedSample:
    embedding: np.ndarray
    bucket: str
    blur_score: float
    brightness: float
    detection_confidence: float
    captured_at: str


def _blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _pose_values(face) -> tuple[float, float] | None:
    pose = getattr(face, "pose", None)
    if pose is None:
        return None
    values = np.asarray(pose, dtype=np.float32).reshape(-1)
    if values.size < 2 or not np.isfinite(values[:2]).all():
        return None
    return float(values[0]), float(values[1])


def classify_angle(face) -> str | None:
    pose = _pose_values(face)
    if pose is None:
        return None

    yaw, pitch = pose

    if pitch <= -18.0 and abs(yaw) <= 35.0:
        return "up"
    if pitch >= 18.0 and abs(yaw) <= 35.0:
        return "down"
    if -12.0 <= yaw <= 12.0 and -12.0 <= pitch <= 12.0:
        return "front"
    if -35.0 <= yaw < -12.0 and -20.0 <= pitch <= 20.0:
        return "slight_left"
    if -90.0 <= yaw < -35.0 and -25.0 <= pitch <= 25.0:
        return "left"
    if 12.0 < yaw <= 35.0 and -20.0 <= pitch <= 20.0:
        return "slight_right"
    if 35.0 < yaw <= 90.0 and -25.0 <= pitch <= 25.0:
        return "right"
    return None


def _is_duplicate(embedding: np.ndarray, samples: list[CapturedSample]) -> bool:
    for sample in samples:
        if float(np.dot(embedding, sample.embedding)) >= DUPLICATE_SIMILARITY:
            return True
    return False


def _bucket_count(samples: list[CapturedSample], bucket: str) -> int:
    return sum(sample.bucket == bucket for sample in samples)


def _has_required_coverage(samples: list[CapturedSample]) -> bool:
    buckets = {sample.bucket for sample in samples}
    horizontal = buckets.intersection(
        {"left", "right", "slight_left", "slight_right"}
    )
    vertical = buckets.intersection({"up", "down"})
    return "front" in buckets and len(horizontal) >= 2 and bool(vertical)


def _get_single_face(frame: np.ndarray):
    faces = face_app.get(frame)
    if len(faces) != 1:
        return None
    return faces[0]


def capture_multi_angle_enrollment(
    name: str,
    relationship: str | None = None,
    camera_index: int = 0,
    target_samples: int = TARGET_SAMPLES,
    max_samples: int = MAX_SAMPLES,
    max_duration_seconds: float = MAX_DURATION_SECONDS,
) -> dict:
    """Passively capture diverse face embeddings from the default camera.

    The person is never instructed to look in a particular direction. Frames
    are accepted opportunistically when a naturally observed angle is useful,
    high quality, and not already over-represented. The session ends when the
    target count plus required angle diversity is reached, or when the timeout
    is reached. Press Q/Esc to cancel.

    Accepted embeddings are persisted to the existing face database.
    """
    target_samples = max(3, min(int(target_samples), MAX_SAMPLES))
    max_samples = max(target_samples, min(int(max_samples), MAX_SAMPLES))
    max_duration_seconds = max(5.0, float(max_duration_seconds))

    existing = get_person_by_name(name)
    if existing is None:
        person_id = create_person(name=name, relationship=relationship)
    else:
        person_id = existing[0]
        if relationship:
            update_person_relationship(person_id, relationship)

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera.")

    samples: list[CapturedSample] = []
    rejected = 0
    duplicates = 0
    last_capture = 0.0
    started = time.monotonic()
    stop_reason = "cancelled"

    try:
        while camera.isOpened():
            elapsed = time.monotonic() - started
            if elapsed >= max_duration_seconds:
                stop_reason = "timeout"
                break

            ok, frame = camera.read()
            if not ok:
                rejected += 1
                stop_reason = "camera_read_failed"
                break

            now = time.monotonic()
            face = _get_single_face(frame)
            status = "Observing naturally..."
            bucket = None

            if face is not None:
                confidence = float(face.det_score)
                bbox = np.asarray(face.bbox, dtype=np.float32)
                width = max(0.0, float(bbox[2] - bbox[0]))
                height = max(0.0, float(bbox[3] - bbox[1]))
                blur = _blur_score(frame)
                bright = _brightness(frame)
                bucket = classify_angle(face)

                if confidence < MIN_DETECTION_CONFIDENCE:
                    rejected += 1
                    status = "Detection confidence too low"
                elif min(width, height) < MIN_FACE_SIZE:
                    rejected += 1
                    status = "Move closer"
                elif blur < MIN_BLUR_SCORE:
                    rejected += 1
                    status = "Hold still"
                elif not MIN_BRIGHTNESS <= bright <= MAX_BRIGHTNESS:
                    rejected += 1
                    status = "Adjust lighting"
                elif bucket is None:
                    rejected += 1
                    status = "Observing angle..."
                elif _bucket_count(samples, bucket) >= MAX_SAMPLES_PER_BUCKET:
                    # Do not waste the gallery on repeated frames from one
                    # already-covered angle. Keep observing for new angles.
                    status = f"{bucket} covered"
                elif now - last_capture < COOLDOWN_SECONDS:
                    status = f"Good: {bucket}"
                else:
                    embedding = normalize_embedding(face.embedding)
                    if _is_duplicate(embedding, samples):
                        duplicates += 1
                        status = "Already captured"
                    else:
                        sample = CapturedSample(
                            embedding=embedding,
                            bucket=bucket,
                            blur_score=blur,
                            brightness=bright,
                            detection_confidence=confidence,
                            captured_at=datetime.now(timezone.utc).isoformat(),
                        )
                        samples.append(sample)
                        add_embedding_to_person(person_id, embedding)
                        last_capture = now
                        status = f"Captured {bucket}"

            cv2.putText(
                frame,
                f"Enrollment: {name}  {len(samples)}/{target_samples}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            cv2.putText(
                frame,
                status,
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )

            captured_buckets = sorted({sample.bucket for sample in samples})
            cv2.putText(
                frame,
                "Angles: " + (", ".join(captured_buckets) or "none"),
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
            cv2.putText(
                frame,
                "Q / Esc: cancel",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )

            cv2.imshow("X-Glasses Multi-Angle Enrollment", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                stop_reason = "cancelled"
                break

            if len(samples) >= target_samples and _has_required_coverage(samples):
                stop_reason = "coverage_complete"
                break

            # The max sample count is only a safety cap. It no longer causes
            # early completion before the required angle coverage is present.
            if len(samples) >= max_samples:
                stop_reason = "max_samples"
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    duration = time.monotonic() - started
    bucket_counts = {
        bucket: sum(sample.bucket == bucket for sample in samples)
        for bucket in ANGLE_BUCKETS
    }
    coverage_complete = _has_required_coverage(samples)

    return {
        "success": bool(samples),
        "person_id": person_id,
        "name": name,
        "relationship": relationship,
        "accepted_samples": len(samples),
        "rejected_frames": rejected,
        "duplicate_frames": duplicates,
        "duration_seconds": round(duration, 2),
        "angle_buckets": bucket_counts,
        "angles_captured": sorted({sample.bucket for sample in samples}),
        "coverage_complete": coverage_complete,
        "completed_automatically": coverage_complete and len(samples) >= target_samples,
        "stop_reason": stop_reason,
    }


if __name__ == "__main__":
    person_name = input("Person name: ").strip()
    if not person_name:
        raise SystemExit("A person name is required.")

    result = capture_multi_angle_enrollment(name=person_name)
    print("\nEnrollment result:")
    for key, value in result.items():
        print(f"{key}: {value}")
