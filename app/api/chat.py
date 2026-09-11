from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import (
    APIRouter,
    File,
    Form,
    UploadFile,
)

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from app.services.memory import save_latest_frame
from app.services.orchestrator import process_request


router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
)


# =========================================================
# TEXT-ONLY CHAT
# =========================================================

@router.post(
    "/",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:

    reply, intent = await process_request(
        session_id=request.session_id,
        message=request.message,
    )

    return ChatResponse(
        success=True,
        session_id=request.session_id,
        user_message=request.message,
        reply=reply,
        intent=intent,
    )


# =========================================================
# CHAT + CURRENT CAMERA FRAME
# =========================================================

@router.post(
    "/with-frame",
    response_model=ChatResponse,
)
async def chat_with_frame(
    session_id: str = Form(...),
    message: str = Form(...),
    image: UploadFile = File(...),
) -> ChatResponse:

    suffix = Path(
        image.filename or "frame.jpg"
    ).suffix

    with NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:

        image_bytes = await image.read()

        temp_file.write(
            image_bytes
        )

        temp_path = temp_file.name

    try:

        # -------------------------------------------------
        # SAVE CURRENT FRAME
        # -------------------------------------------------

        frames_dir = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "frames"
        )

        frames_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        latest_frame_path = (
            frames_dir
            / f"{session_id}.jpg"
        )

        latest_frame_path.write_bytes(
            Path(temp_path).read_bytes()
        )

        save_latest_frame(
            session_id=session_id,
            image_path=str(
                latest_frame_path
            ),
        )

        # -------------------------------------------------
        # PROCESS MESSAGE
        # -------------------------------------------------

        reply, intent = await process_request(
            session_id=session_id,
            message=message,
        )

        return ChatResponse(
            success=True,
            session_id=session_id,
            user_message=message,
            reply=reply,
            intent=intent,
        )

    finally:

        Path(
            temp_path
        ).unlink(
            missing_ok=True
        )