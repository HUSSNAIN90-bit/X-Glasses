from dataclasses import dataclass
import re

import numpy as np

from app.services.face_database import (
    add_embedding_to_person,
    create_person,
    get_person_by_name,
    update_person_relationship,
)
from app.services.face_recognition import extract_single_face_embedding


RELATIONSHIP_ALIASES = {
    "friend": "friend",
    "frnd": "friend",
    "brother": "brother",
    "sister": "sister",
    "mother": "mother",
    "father": "father",
    "wife": "wife",
    "husband": "husband",
    "colleague": "colleague",
    "coworker": "coworker",
    "teacher": "teacher",
}
RELATIONSHIP_PATTERN = "|".join(RELATIONSHIP_ALIASES)
NAME_PATTERN = r"[A-Za-z][A-Za-z' -]{0,48}"

# Keep enrollment conservative: store genuinely different views rather than
# filling the gallery with near-identical camera frames.
ENROLLMENT_TARGET_FRAMES = 5
ENROLLMENT_DUPLICATE_SIMILARITY = 0.97


@dataclass(frozen=True)
class FaceIntroduction:
    name: str
    relationship: str | None


@dataclass(frozen=True)
class FaceEnrollmentResult:
    success: bool
    reply: str
    person_id: str | None = None
    accepted_frames: int = 0
    attempted_frames: int = 0


def _normalize_name(value: str) -> str | None:
    name = re.sub(r"\s+", " ", value).strip(" ,.!?")
    name = re.sub(
        r"\s+(?:here|today|please|right now)$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    if not name or name.lower() in {"a", "an", "friend", "person", "someone"}:
        return None
    return " ".join(part.capitalize() for part in name.split(" "))


def parse_face_introduction(message: str) -> FaceIntroduction | None:
    """Recognize explicit spoken introductions without spending an LLM request."""
    normalized = re.sub(r"\s+", " ", message).strip()
    if not normalized:
        return None

    relation_match = re.search(
        rf"\b(?:this(?:\s+person)?\s+is|he\s+is|she\s+is|they\s+are|meet)\s+"
        rf"(?:my\s+)?(?P<relationship>{RELATIONSHIP_PATTERN})\s*[,:]?\s*"
        rf"(?:named\s+)?(?P<name>{NAME_PATTERN})(?:[.!?]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if relation_match:
        name = _normalize_name(relation_match.group("name"))
        if name:
            relationship = RELATIONSHIP_ALIASES[
                relation_match.group("relationship").lower()
            ]
            return FaceIntroduction(name=name, relationship=relationship)

    name_match = re.search(
        rf"\b(?:this(?:\s+person)?\s+is|he\s+is|she\s+is|they\s+are|meet)\s+"
        rf"(?:named\s+)?(?P<name>{NAME_PATTERN})(?:[.!?]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if name_match:
        name = _normalize_name(name_match.group("name"))
        if name:
            return FaceIntroduction(name=name, relationship=None)

    save_match = re.search(
        rf"\b(?:remember|save|introduce)\s+(?:this\s+(?:person|face)|him|her|them)\s+"
        rf"(?:as|is)\s+(?:(?:my\s+)?(?P<relationship>{RELATIONSHIP_PATTERN})\s+)?"
        rf"(?P<name>{NAME_PATTERN})(?:[.!?]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if save_match:
        name = _normalize_name(save_match.group("name"))
        if name:
            relationship_value = save_match.group("relationship")
            relationship = (
                RELATIONSHIP_ALIASES[relationship_value.lower()]
                if relationship_value
                else None
            )
            return FaceIntroduction(name=name, relationship=relationship)

    return None


def _is_duplicate(
    embedding: np.ndarray,
    selected_embeddings: list[np.ndarray],
) -> bool:
    for existing in selected_embeddings:
        similarity = float(np.dot(embedding, existing))
        if similarity >= ENROLLMENT_DUPLICATE_SIMILARITY:
            return True
    return False


def enroll_introduced_person(
    image_paths: str | list[str],
    introduction: FaceIntroduction,
) -> FaceEnrollmentResult:
    """Enroll several good camera frames from one spoken introduction.

    The caller is responsible for selecting good-quality frames. This service
    performs the biometric-specific gates: exactly one face and embedding
    diversity. Existing single-frame callers remain supported.
    """
    if isinstance(image_paths, str):
        paths = [image_paths]
    else:
        paths = list(image_paths)

    if not paths:
        return FaceEnrollmentResult(
            success=False,
            reply="I couldn't read any camera frames.",
        )

    selected_embeddings: list[np.ndarray] = []

    for path in paths:
        try:
            embedding = extract_single_face_embedding(path)
        except (FileNotFoundError, ValueError):
            continue
        except Exception:
            continue

        if embedding is None:
            continue

        if _is_duplicate(embedding, selected_embeddings):
            continue

        selected_embeddings.append(embedding)
        if len(selected_embeddings) >= ENROLLMENT_TARGET_FRAMES:
            break

    if not selected_embeddings:
        return FaceEnrollmentResult(
            success=False,
            reply="I can't save them because I don't see one clear face in the captured frames.",
            attempted_frames=len(paths),
        )

    if len(selected_embeddings) < ENROLLMENT_TARGET_FRAMES:
        return FaceEnrollmentResult(
            success=False,
            reply=(
                f"I only got {len(selected_embeddings)} useful face views. "
                "Please keep the person in view and try the introduction again."
            ),
            accepted_frames=len(selected_embeddings),
            attempted_frames=len(paths),
        )

    try:
        existing_person = get_person_by_name(introduction.name)
        if existing_person is None:
            person_id = create_person(
                name=introduction.name,
                relationship=introduction.relationship,
            )
        else:
            person_id = existing_person[0]
            if introduction.relationship:
                update_person_relationship(person_id, introduction.relationship)

        for embedding in selected_embeddings:
            add_embedding_to_person(
                person_id=person_id,
                embedding=embedding,
            )
    except Exception:
        return FaceEnrollmentResult(
            success=False,
            reply="I couldn't save that person right now. Please try again.",
            accepted_frames=len(selected_embeddings),
            attempted_frames=len(paths),
        )

    relationship_phrase = (
        f" as your {introduction.relationship}"
        if introduction.relationship
        else ""
    )
    return FaceEnrollmentResult(
        success=True,
        person_id=person_id,
        accepted_frames=len(selected_embeddings),
        attempted_frames=len(paths),
        reply=(
            f"I've saved {introduction.name}{relationship_phrase} "
            f"using {len(selected_embeddings)} different face views."
        ),
    )
