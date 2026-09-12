from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.schemas.chat import ChatResponse
from app.services.memory import save_latest_frame
from app.services.orchestrator import process_request


router = APIRouter(
    prefix="/api/session",
    tags=["session"],
)


@router.post("/input", response_model=ChatResponse)
async def session_input(
    session_id: str = Form(...),
    command: str = Form(...),
    image: UploadFile | None = File(default=None),
) -> ChatResponse:
    if not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id is required",
        )

    if not command.strip():
        raise HTTPException(
            status_code=400,
            detail="command is required",
        )

    if image is not None:
        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty",
            )

        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be an image",
            )

        save_latest_frame(
            session_id=session_id,
            image_bytes=image_bytes,
        )

    reply, intent = process_request(
        session_id=session_id,
        message=command,
    )

    return ChatResponse(
        success=True,
        session_id=session_id,
        user_message=command,
        reply=reply,
        intent=intent,
    )