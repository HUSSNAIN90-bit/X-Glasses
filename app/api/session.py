from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.chat import ChatResponse
from app.services.memory import save_latest_frame
from app.services.orchestrator import process_request


router = APIRouter(
    prefix="/api/session",
    tags=["session"],
)

FRAME_DIR = Path("data/frames")
FRAME_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/input", response_model=ChatResponse)
async def session_input(
    session_id: str = Form(...),
    command: str = Form(...),
    image: UploadFile | None = File(default=None),
) -> ChatResponse:
    session_id = session_id.strip()
    command = command.strip()

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required",
        )

    if not command:
        raise HTTPException(
            status_code=400,
            detail="command is required",
        )

    if image is not None:
        if not image.content_type or not image.content_type.startswith(
            "image/"
        ):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be an image",
            )

        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty",
            )

        frame_path = FRAME_DIR / f"{session_id}.jpg"
        frame_path.write_bytes(image_bytes)

        save_latest_frame(
            session_id=session_id,
            image_path=str(frame_path),
        )

    reply, intent = await process_request(
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