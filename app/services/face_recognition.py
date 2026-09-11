from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.services.face_database import (
    get_all_people_with_embeddings,
)


MODEL_NAME = "buffalo_l"

# 0.45 se low matches reject honge.
# Real-world testing ke baad is value ko tune karenge.
FACE_MATCH_THRESHOLD = 0.45


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

    norm = float(
        np.linalg.norm(embedding)
    )

    if norm == 0.0:
        return embedding

    return embedding / norm


def extract_face_embeddings(
    image_path: str,
) -> list[np.ndarray]:

    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(
        str(image_file)
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    faces = face_app.get(image)

    embeddings: list[np.ndarray] = []

    for face in faces:

        embedding = normalize_embedding(
            face.embedding
        )

        if embedding.size == 0:
            continue

        embeddings.append(
            embedding
        )

    return embeddings


def recognize_faces(
    image_path: str,
) -> list[FaceRecognitionData]:

    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(
        str(image_file)
    ) 

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    faces = face_app.get(image)

    people = get_all_people_with_embeddings()

    results: list[
        FaceRecognitionData
    ] = []

    for face in faces:

        embedding = normalize_embedding(
            face.embedding
        )

        if embedding.size == 0:
            continue

        best_name: str | None = None
        best_similarity: float | None = None

        for (
            person_id,
            name,
            saved_embedding,
        ) in people:

            saved_embedding = normalize_embedding(
                saved_embedding
            )

            if saved_embedding.size == 0:
                continue

            similarity = float(
                np.dot(
                    embedding,
                    saved_embedding,
                )
            )

            if (
                best_similarity is None
                or similarity > best_similarity
            ):
                best_similarity = similarity
                best_name = name

        if (
            best_similarity is None
            or best_similarity < FACE_MATCH_THRESHOLD
        ):
            best_name = None

        bbox = face.bbox.tolist()

        results.append(
            FaceRecognitionData(
                name=best_name,
                similarity=best_similarity,
                confidence=float(
                    face.det_score
                ),
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
            )
        )

    return results