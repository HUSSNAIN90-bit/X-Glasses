import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


DATABASE_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "xglasses.db"
)


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                relationship TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (person_id)
                    REFERENCES people(id)
                    ON DELETE CASCADE
            )
            """
        )

        # Existing database migration
        columns = connection.execute(
            """
            PRAGMA table_info(people)
            """
        ).fetchall()

        column_names = {
            str(column[1])
            for column in columns
        }

        if "relationship" not in column_names:
            connection.execute(
                """
                ALTER TABLE people
                ADD COLUMN relationship TEXT
                """
            )

        connection.commit()

def create_person(
    name: str,
    relationship: str | None = None,
) -> str:

    person_id = str(uuid4())

    created_at = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.execute(
            """
            INSERT INTO people
            (id, name, relationship, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                person_id,
                name,
                relationship,
                created_at,
            ),
        )

        connection.commit()

    return person_id


def save_embedding(
    person_id: str,
    embedding: np.ndarray,
) -> str:

    embedding_id = str(uuid4())

    normalized_embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    created_at = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    with sqlite3.connect(DATABASE_PATH) as connection:

        connection.execute(
            """
            INSERT INTO face_embeddings
            (
                id,
                person_id,
                embedding,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                embedding_id,
                person_id,
                normalized_embedding.tobytes(),
                created_at,
            ),
        )

        connection.commit()

    return embedding_id


def get_all_people_with_embeddings() -> list[
    tuple[str, str, np.ndarray]
]:

    with sqlite3.connect(DATABASE_PATH) as connection:

        rows = connection.execute(
            """
            SELECT
                people.id,
                people.name,
                face_embeddings.embedding

            FROM people

            INNER JOIN face_embeddings
                ON people.id = face_embeddings.person_id
            """
        ).fetchall()

    people: list[
        tuple[str, str, np.ndarray]
    ] = []

    for (
        person_id,
        name,
        embedding_blob,
    ) in rows:

        embedding = np.frombuffer(
            embedding_blob,
            dtype=np.float32,
        ).copy()

        people.append(
            (
                str(person_id),
                str(name),
                embedding,
            )
        )

    return people


def get_person_by_name(
    name: str,
) -> tuple[str, str] | None:

    with sqlite3.connect(DATABASE_PATH) as connection:

        row = connection.execute(
            """
            SELECT
                id,
                name

            FROM people

            WHERE LOWER(name) = LOWER(?)

            LIMIT 1
            """,
            (
                name.strip(),
            ),
        ).fetchone()

    if row is None:
        return None

    return (
        str(row[0]),
        str(row[1]),
    )

def get_all_people() -> list[
    tuple[str, str, str, int]
]:

    with sqlite3.connect(DATABASE_PATH) as connection:

        rows = connection.execute(
            """
            SELECT
                people.id,
                people.name,
                people.created_at,
                COUNT(face_embeddings.id) AS embedding_count

            FROM people

            LEFT JOIN face_embeddings
                ON people.id = face_embeddings.person_id

            GROUP BY
                people.id,
                people.name,
                people.created_at

            ORDER BY people.created_at ASC
            """
        ).fetchall()

    people: list[
        tuple[str, str, str, int]
    ] = []

    for (
        person_id,
        name,
        created_at,
        embedding_count,
    ) in rows:

        people.append(
            (
                str(person_id),
                str(name),
                str(created_at),
                int(embedding_count),
            )
        )

    return people

def delete_person(
    person_id: str,
) -> bool:

    with sqlite3.connect(DATABASE_PATH) as connection:

        # Delete embeddings first
        connection.execute(
            """
            DELETE FROM face_embeddings
            WHERE person_id = ?
            """,
            (person_id,),
        )

        cursor = connection.execute(
            """
            DELETE FROM people
            WHERE id = ?
            """,
            (person_id,),
        )

        connection.commit()

        return cursor.rowcount > 0

def add_embedding_to_person(
    person_id: str,
    embedding: np.ndarray,
) -> str:

    return save_embedding(
        person_id=person_id,
        embedding=embedding,
    )
def resolve_person_reference(
    reference: str,
) -> tuple[str, str] | None:

    clean_reference = (
        reference.strip().lower()
    )

    # Direct name lookup
    person = get_person_by_name(
        clean_reference
    )

    if person is not None:
        return person

    # Relationship lookup
    relationship = clean_reference

    if relationship.startswith("my "):
        relationship = relationship[3:].strip()

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        rows = connection.execute(
            """
            SELECT id, name
            FROM people
            WHERE LOWER(relationship) = LOWER(?)
            """,
            (relationship,),
        ).fetchall()

    # No match
    if not rows:
        return None

    # More than one person has same relationship
    if len(rows) > 1:
        return None

    return (
        str(rows[0][0]),
        str(rows[0][1]),
    )

def update_person_relationship(
    person_id: str,
    relationship: str,
) -> bool:

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        cursor = connection.execute(
            """
            UPDATE people
            SET relationship = ?
            WHERE id = ?
            """,
            (
                relationship.strip().lower(),
                person_id,
            ),
        )

        connection.commit()

        return cursor.rowcount > 0