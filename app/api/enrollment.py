from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.face_enrollment import (
    FaceIntroduction,
    enroll_introduced_person,
)
from app.services.frame_quality import evaluate_frame_quality


router = APIRouter(
    prefix="/api/faces",
    tags=["Faces"],
)

MAX_ENROLLMENT_FRAMES = 8
MIN_GOOD_FRAMES = 5


@router.post("/enroll-multi")
async def enroll_multi_face(
    name: str = Form(...),
    relationship: str | None = Form(None),
    images: list[UploadFile] = File(...),
):
    clean_name = name.strip()
    clean_relationship = relationship.strip().lower() if relationship and relationship.strip() else None

    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    if not images:
        raise HTTPException(status_code=400, detail="At least one enrollment frame is required.")
    if len(images) > MAX_ENROLLMENT_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Send no more than {MAX_ENROLLMENT_FRAMES} enrollment frames.",
        )

    temp_paths: list[str] = []
    try:
        for image in images:
            if image.content_type and not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Every enrollment frame must be an image.")

            suffix = Path(image.filename or "face.jpg").suffix or ".jpg"
            with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(await image.read())
                temp_paths.append(temp_file.name)

        good_paths: list[str] = []
        for path in temp_paths:
            metrics = evaluate_frame_quality(
                path,
                80.0,
                35.0,
                220.0,
            )
            if metrics.good:
                good_paths.append(path)

        if len(good_paths) < MIN_GOOD_FRAMES:
            return {
                "success": False,
                "reply": (
                    f"I only got {len(good_paths)} good-quality frames. "
                    "Please keep the face clearly visible and try again."
                ),
                "accepted_frames": 0,
                "good_quality_frames": len(good_paths),
                "attempted_frames": len(temp_paths),
            }

        result = enroll_introduced_person(
            image_paths=good_paths,
            introduction=FaceIntroduction(
                name=clean_name,
                relationship=clean_relationship,
            ),
        )

        return {
            "success": result.success,
            "reply": result.reply,
            "person_id": result.person_id,
            "accepted_frames": result.accepted_frames,
            "attempted_frames": result.attempted_frames,
            "good_quality_frames": len(good_paths),
        }
    finally:
        for path in temp_paths:
            Path(path).unlink(missing_ok=True)
