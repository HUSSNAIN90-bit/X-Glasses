from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.services.face_database import (
    get_all_people_with_embeddings,
)


MODEL_NAME = "buffalo_l"

# Keep the existing threshold conservative. The outdoor fix below improves
# illumination before recognition instead of lowering the threshold and
# increasing false-positive matches.
FACE_MATCH_THRESHOLD = 0.45

# Only run the extra illumination variants when the frame is likely to have
# difficult lighting. This keeps normal frames fast enough for live use.
LOW_LIGHT_BRIGHTNESS = 95.0
HIGH_LIGHT_BRIGHTNESS = 185.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


@dataclass(frozen=True)
class FaceRecognitionData:
    name: str | None
    similarity: float | None
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


face_app = FaceAnalysis(
    name=MODEL_NAME,
    providers=[
        "CPUExecutionProvider",
    ],
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640),
)


def normalize_embedding(
    embedding: np.ndarray,
) -> np.ndarray:
    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    norm = float(np.linalg.norm(embedding))

    if norm == 0.0:
        return embedding

    return embedding / norm


def _gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction without changing the camera image dimensions."""
    gamma = max(0.1, float(gamma))
    table = np.array(
        [((i / 255.0) ** gamma) * 255.0 for i in np.arange(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def _clahe_image(image: np.ndarray) -> np.ndarray:
    """Improve local contrast while preserving the original colour balance."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    l_channel = clahe.apply(l_channel)

    return cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )


def _lighting_variants(image: np.ndarray) -> list[np.ndarray]:
    """Return the original image plus only the useful lighting fallbacks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    # Normal lighting: don't spend extra inference time on preprocessing.
    if LOW_LIGHT_BRIGHTNESS <= brightness <= HIGH_LIGHT_BRIGHTNESS:
        return [image]

    clahe = _clahe_image(image)

    if brightness < LOW_LIGHT_BRIGHTNESS:
        # Gamma < 1 lifts shadow detail.
        gamma = _gamma_correct(image, 0.65)
    else:
        # Gamma > 1 reduces harsh highlights from strong outdoor light.
        gamma = _gamma_correct(image, 1.35)

    return [image, clahe, gamma]


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def _recognize_candidate(
    face,
    people,
) -> tuple[str | None, float | None]:
    embedding = normalize_embedding(face.embedding)

    if embedding.size == 0:
        return None, None

    best_name: str | None = None
    best_similarity: float | None = None

    for (
        _person_id,
        name,
        saved_embedding,
    ) in people:
        saved_embedding = normalize_embedding(saved_embedding)

        if saved_embedding.size == 0:
            continue

        similarity = float(np.dot(embedding, saved_embedding))

        if best_similarity is None or similarity > best_similarity:
            best_similarity = similarity
            best_name = name

    if (
        best_similarity is None
        or best_similarity < FACE_MATCH_THRESHOLD
    ):
        best_name = None

    return best_name, best_similarity


def extract_face_embeddings(
    image_path: str,
) -> list[np.ndarray]:
    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_file))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # Enrollment also benefits from the same lighting normalization. We still
    # prefer the original frame and use enhanced variants only when needed.
    embeddings: list[np.ndarray] = []

    for variant in _lighting_variants(image):
        faces = face_app.get(variant)

        if not faces:
            continue

        # Enrollment requires exactly one face. Use the strongest detection
        # from the best available lighting variant and stop here.
        face = max(
            faces,
            key=lambda item: float(item.det_score),
        )
        embedding = normalize_embedding(face.embedding)

        if embedding.size:
            embeddings.append(embedding)

        # Avoid returning duplicate embeddings from several lighting variants.
        break

    return embeddings


def recognize_faces(
    image_path: str,
) -> list[FaceRecognitionData]:
    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_file))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    people = get_all_people_with_embeddings()
    candidates: list[dict] = []

    # Original frame is always tried first. On difficult lighting, CLAHE and
    # gamma-corrected versions give the detector/ArcFace a second chance.
    for variant in _lighting_variants(image):
        faces = face_app.get(variant)

        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)
            name, similarity = _recognize_candidate(face, people)

            # Prefer identity similarity when available, while retaining the
            # detector confidence as a tie-breaker for unknown faces.
            rank = (
                float(similarity) if similarity is not None else -1.0,
                float(face.det_score),
            )

            candidates.append(
                {
                    "face": face,
                    "bbox": bbox,
                    "name": name,
                    "similarity": similarity,
                    "rank": rank,
                }
            )

    # The same physical face can be detected in the original + enhanced
    # variants. Merge those overlapping detections so the API never reports
    # duplicates just because lighting correction found the face too.
    selected: list[dict] = []
    for candidate in candidates:
        duplicate_index = None

        for index, existing in enumerate(selected):
            if _bbox_iou(candidate["bbox"], existing["bbox"]) >= 0.45:
                duplicate_index = index
                break

        if duplicate_index is None:
            selected.append(candidate)
        elif candidate["rank"] > selected[duplicate_index]["rank"]:
            selected[duplicate_index] = candidate

    results: list[FaceRecognitionData] = []

    for candidate in selected:
        face = candidate["face"]
        bbox = candidate["bbox"].tolist()

        results.append(
            FaceRecognitionData(
                name=candidate["name"],
                similarity=candidate["similarity"],
                confidence=float(face.det_score),
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
            )
        )

    return results
