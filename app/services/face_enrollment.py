from dataclasses import dataclass
import re

import numpy as np

from app.services.face_database import (
    add_embedding_to_person,
    create_person,
    get_all_people_with_embeddings,
    get_person_by_name,
    update_person_relationship,
)
from app.services.face_recognition import extract_single_face_embedding

RELATIONSHIP_ALIASES = {"friend": "friend", "frnd": "friend", "brother": "brother", "sister": "sister", "mother": "mother", "father": "father", "wife": "wife", "husband": "husband", "colleague": "colleague", "coworker": "coworker", "teacher": "teacher"}
RELATIONSHIP_PATTERN = "|".join(RELATIONSHIP_ALIASES)
NAME_PATTERN = r"[A-Za-z][A-Za-z' -]{0,48}"
ENROLLMENT_TARGET_FRAMES = 5
ENROLLMENT_DUPLICATE_SIMILARITY = 0.97
# Frames that do not resemble the first reliable face are treated as another
# person / accidental frame and are never enrolled.
ENROLLMENT_TARGET_SIMILARITY = 0.45


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
    name = re.sub(r"\s+(?:here|today|please|right now)$", "", name, flags=re.IGNORECASE)
    if not name or name.lower() in {"a", "an", "friend", "person", "someone"}:
        return None
    return " ".join(part.capitalize() for part in name.split(" "))


def parse_face_introduction(message: str) -> FaceIntroduction | None:
    normalized = re.sub(r"\s+", " ", message).strip()
    if not normalized:
        return None
    relation_match = re.search(rf"\b(?:this(?:\s+person)?\s+is|he\s+is|she\s+is|they\s+are|meet)\s+(?:my\s+)?(?P<relationship>{RELATIONSHIP_PATTERN})\s*[,:]?\s*(?:named\s+)?(?P<name>{NAME_PATTERN})(?:[.!?]|$)", normalized, flags=re.IGNORECASE)
    if relation_match:
        name = _normalize_name(relation_match.group("name"))
        if name:
            return FaceIntroduction(name=name, relationship=RELATIONSHIP_ALIASES[relation_match.group("relationship").lower()])
    name_match = re.search(rf"\b(?:this(?:\s+person)?\s+is|he\s+is|she\s+is|they\s+are|meet)\s+(?:named\s+)?(?P<name>{NAME_PATTERN})(?:[.!?]|$)", normalized, flags=re.IGNORECASE)
    if name_match:
        name = _normalize_name(name_match.group("name"))
        if name:
            return FaceIntroduction(name=name, relationship=None)
    save_match = re.search(rf"\b(?:remember|save|introduce)\s+(?:this\s+(?:person|face)|him|her|them)\s+(?:as|is)\s+(?:(?:my\s+)?(?P<relationship>{RELATIONSHIP_PATTERN})\s+)?(?P<name>{NAME_PATTERN})(?:[.!?]|$)", normalized, flags=re.IGNORECASE)
    if save_match:
        name = _normalize_name(save_match.group("name"))
        if name:
            value = save_match.group("relationship")
            return FaceIntroduction(name=name, relationship=RELATIONSHIP_ALIASES[value.lower()] if value else None)
    return None


def _is_duplicate(embedding: np.ndarray, selected_embeddings: list[np.ndarray]) -> bool:
    return any(float(np.dot(embedding, existing)) >= ENROLLMENT_DUPLICATE_SIMILARITY for existing in selected_embeddings)


def _matches_anchor(embedding: np.ndarray, anchor: np.ndarray) -> bool:
    return float(np.dot(embedding, anchor)) >= ENROLLMENT_TARGET_SIMILARITY


def _matches_existing_person(embedding: np.ndarray, name: str) -> bool:
    """Prevent a different person from being saved under an existing name."""
    existing_embeddings = [e for _, person_name, e in get_all_people_with_embeddings() if person_name.lower() == name.lower()]
    if not existing_embeddings:
        return True
    return max(float(np.dot(embedding, existing)) for existing in existing_embeddings) >= ENROLLMENT_TARGET_SIMILARITY


def enroll_introduced_person(image_paths: str | list[str], introduction: FaceIntroduction) -> FaceEnrollmentResult:
    paths = [image_paths] if isinstance(image_paths, str) else list(image_paths)
    if not paths:
        return FaceEnrollmentResult(False, "I couldn't read any camera frames.")

    selected_embeddings: list[np.ndarray] = []
    anchor: np.ndarray | None = None

    for path in paths:
        try:
            embedding = extract_single_face_embedding(path)
        except Exception:
            continue
        if embedding is None:
            continue

        # First usable face locks the enrollment target. A moving camera or a
        # second person entering the frame cannot contaminate the gallery.
        if anchor is None:
            if not _matches_existing_person(embedding, introduction.name):
                return FaceEnrollmentResult(False, f"This doesn't look like the existing {introduction.name}. Please make sure only {introduction.name} is in view.", attempted_frames=len(paths))
            anchor = embedding
            selected_embeddings.append(embedding)
            continue

        if not _matches_anchor(embedding, anchor):
            continue
        if _is_duplicate(embedding, selected_embeddings):
            continue
        selected_embeddings.append(embedding)
        if len(selected_embeddings) >= ENROLLMENT_TARGET_FRAMES:
            break

    if not selected_embeddings:
        return FaceEnrollmentResult(False, "I can't save them because I don't see one clear face in the captured frames.", attempted_frames=len(paths))
    if len(selected_embeddings) < ENROLLMENT_TARGET_FRAMES:
        return FaceEnrollmentResult(False, f"I only got {len(selected_embeddings)} useful views of {introduction.name}. Please keep them in view and try again.", len(selected_embeddings), len(paths))

    try:
        existing_person = get_person_by_name(introduction.name)
        if existing_person is None:
            person_id = create_person(introduction.name, introduction.relationship)
        else:
            person_id = existing_person[0]
            if introduction.relationship:
                update_person_relationship(person_id, introduction.relationship)
        for embedding in selected_embeddings:
            add_embedding_to_person(person_id, embedding)
    except Exception:
        return FaceEnrollmentResult(False, "I couldn't save that person right now. Please try again.", len(selected_embeddings), len(paths))

    relationship_phrase = f" as your {introduction.relationship}" if introduction.relationship else ""
    return FaceEnrollmentResult(True, f"I've saved {introduction.name}{relationship_phrase} using {len(selected_embeddings)} different face views.", person_id, len(selected_embeddings), len(paths))
