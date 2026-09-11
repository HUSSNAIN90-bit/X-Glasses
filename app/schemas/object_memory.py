from pydantic import BaseModel, Field


class ObjectMemoryCreate(BaseModel):
    object_name: str = Field(
        min_length=1,
        max_length=100,
    )

    room: str = Field(
        min_length=1,
        max_length=100,
    )

    description: str | None = Field(
        default=None,
        max_length=500,
    )


class ObjectMemoryResponse(BaseModel):
    success: bool
    object_id: str
    object_name: str
    room: str
    description: str | None