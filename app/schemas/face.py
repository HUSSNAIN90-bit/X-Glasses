from pydantic import BaseModel, Field


class FaceDetection(BaseModel):
    face_id: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    x1: float
    y1: float
    x2: float
    y2: float


class PersonCreate(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=100,
    )


class PersonResponse(BaseModel):
    success: bool
    person_id: str
    name: str


class FaceRecognitionResult(BaseModel):
    face_id: str

    name: str | None

    similarity: float | None

    recognized: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    x1: float
    y1: float
    x2: float
    y2: float


class FaceRecognitionResponse(BaseModel):
    success: bool
    results: list[FaceRecognitionResult]

class PersonListItem(BaseModel):
    person_id: str
    name: str
    created_at: str
    embedding_count: int


class PeopleListResponse(BaseModel):
    success: bool
    people: list[PersonListItem]
    

class DeletePersonResponse(BaseModel):
    success: bool
    person_id: str
    message: str