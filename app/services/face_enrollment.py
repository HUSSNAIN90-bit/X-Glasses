from dataclasses import dataclass
import re

from app.services.face_database import (
    add_embedding_to_person,
    create_person,
    get_person_by_name,
    update_person_relationship,
)
from app.services.face_recognition import extract_face_embeddings


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


@dataclass(frozen=True)
class FaceIntroduction:
    name: str
    relationship: str | None


@dataclass(frozen=True)
class FaceEnrollmentResult:
    success: bool
    reply: str
    person_id: str | None = None


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


def enroll_introduced_person(
    image_path: str,
    introduction: FaceIntroduction,
) -> FaceEnrollmentResult:
    try:
        embeddings = extract_face_embeddings(image_path)
    except (FileNotFoundError, ValueError):
        return FaceEnrollmentResult(
            success=False,
            reply="I couldn't read the current camera frame.",
        )
    except Exception:
        return FaceEnrollmentResult(
            success=False,
            reply="I couldn't detect the face right now. Please try again.",
        )

    if not embeddings:
        return FaceEnrollmentResult(
            success=False,
            reply="I can't save them because I don't see a clear face.",
        )
    if len(embeddings) != 1:
        return FaceEnrollmentResult(
            success=False,
            reply="Please show only one face while introducing someone.",
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

        add_embedding_to_person(
            person_id=person_id,
            embedding=embeddings[0],
        )
    except Exception:
        return FaceEnrollmentResult(
            success=False,
            reply="I couldn't save that person right now. Please try again.",
        )

    relationship_phrase = (
        f" as your {introduction.relationship}"
        if introduction.relationship
        else ""
    )
    return FaceEnrollmentResult(
        success=True,
        person_id=person_id,
        reply=f"I've saved {introduction.name}{relationship_phrase}.",
    )
