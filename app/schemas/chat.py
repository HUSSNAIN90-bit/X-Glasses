from typing import Literal

from pydantic import BaseModel, Field


Intent = Literal[
    "general",
    "vision",
    "face_enrollment",
    "person_location",
    "object_search",
    "memory",
]


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    session_id: str = Field(
        min_length=1,
        max_length=100,
    )


class ChatResponse(BaseModel):
    success: bool
    session_id: str
    user_message: str
    reply: str
    intent: Intent


class IntentResult(BaseModel):
    intent: Intent
    person: str | None = None
    object_name: str | None = None
