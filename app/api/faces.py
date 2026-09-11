from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.schemas.face import (
    DeletePersonResponse,
    FaceRecognitionResponse,
    FaceRecognitionResult,
    PeopleListResponse,
    PersonListItem,
    PersonResponse,
)

from app.services.face_database import (
    add_embedding_to_person,
    create_person,
    delete_person,
    get_all_people,
    get_person_by_name,
)
from app.services.face_recognition import (
    extract_face_embeddings,
    recognize_faces,
)


router = APIRouter(
    prefix="/api/faces",
    tags=["Faces"],
)

@router.get(
    "/people",
    response_model=PeopleListResponse,
)
async def list_people() -> PeopleListResponse:

    people = get_all_people()

    results: list[PersonListItem] = []

    for (
        person_id,
        name,
        created_at,
        embedding_count,
    ) in people:

        results.append(
            PersonListItem(
                person_id=person_id,
                name=name,
                created_at=created_at,
                embedding_count=embedding_count,
            )
        )

    return PeopleListResponse(
        success=True,
        people=results,
    )


@router.post(
    "/enroll",
    response_model=PersonResponse,
)
async def enroll_face(
    name: str = Form(...),
    relationship: str | None = Form(None),
    image: UploadFile = File(...),
) -> PersonResponse:
    clean_relationship = (
    relationship.strip().lower()
    if relationship is not None
    and relationship.strip()
    else None
    )
    
    
    clean_name = name.strip()

    if not clean_name:
        raise HTTPException(
            status_code=400,
            detail="Name cannot be empty.",
        )

    suffix = Path(
        image.filename or "face.jpg"
    ).suffix

    with NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:

        image_bytes = await image.read()
        temp_file.write(image_bytes)
        temp_path = temp_file.name

    try:
        embeddings = extract_face_embeddings(
            temp_path
        )

        if len(embeddings) == 0:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image.",
            )

        if len(embeddings) > 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Multiple faces detected. "
                    "Enrollment requires exactly one face."
                ),
            )

        existing_person = get_person_by_name(
            clean_name
        )

        if existing_person is None:
            person_id = create_person(
                name=clean_name,
                relationship=clean_relationship,
            )
        else:
            person_id = existing_person[0]

        add_embedding_to_person(
            person_id=person_id,
            embedding=embeddings[0],
        )

        return PersonResponse(
            success=True,
            person_id=person_id,
            name=clean_name,
        )

    finally:
        Path(temp_path).unlink(
            missing_ok=True
        )


@router.post(
    "/recognize",
    response_model=FaceRecognitionResponse,
)
async def recognize(
    image: UploadFile = File(...),
) -> FaceRecognitionResponse:

    suffix = Path(
        image.filename or "faces.jpg"
    ).suffix

    with NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:

        image_bytes = await image.read()
        temp_file.write(image_bytes)
        temp_path = temp_file.name

    try:
        recognition_results = recognize_faces(
            temp_path
        )

        results: list[
            FaceRecognitionResult
        ] = []

        for index, recognition in enumerate(
            recognition_results
        ):
            results.append(
                FaceRecognitionResult(
                    face_id=f"face-{index + 1}",
                    name=recognition.name,
                    similarity=recognition.similarity,
                    recognized=(
                        recognition.name is not None
                    ),
                    confidence=recognition.confidence,
                    x1=recognition.x1,
                    y1=recognition.y1,
                    x2=recognition.x2,
                    y2=recognition.y2,
                )
            )

        return FaceRecognitionResponse(
            success=True,
            results=results,
        )

    finally:
        Path(temp_path).unlink(
            missing_ok=True
        )
        
@router.delete(
    "/people/{person_id}",
    response_model=DeletePersonResponse,
)
async def remove_person(
    person_id: str,
) -> DeletePersonResponse:

    deleted = delete_person(
        person_id=person_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Person not found.",
        )

    return DeletePersonResponse(
        success=True,
        person_id=person_id,
        message="Person and all face embeddings deleted successfully.",
    )
    
@router.post(
    "/people/{person_id}/embeddings",
    response_model=PersonResponse,
)
async def add_face_embedding(
    person_id: str,
    image: UploadFile = File(...),
) -> PersonResponse:

    suffix = Path(
        image.filename or "face.jpg"
    ).suffix

    with NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:

        image_bytes = await image.read()

        temp_file.write(image_bytes)

        temp_path = temp_file.name

    try:

        embeddings = extract_face_embeddings(
            temp_path
        )

        if len(embeddings) == 0:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image.",
            )

        if len(embeddings) > 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Multiple faces detected. "
                    "Please upload an image containing "
                    "only one face."
                ),
            )

        people = get_all_people()

        person = next(
            (
                person
                for person in people
                if person[0] == person_id
            ),
            None,
        )

        if person is None:
            raise HTTPException(
                status_code=404,
                detail="Person not found.",
            )

        add_embedding_to_person(
            person_id=person_id,
            embedding=embeddings[0],
        )

        return PersonResponse(
            success=True,
            person_id=person_id,
            name=person[1],
        )

    finally:

        Path(temp_path).unlink(
            missing_ok=True
        )