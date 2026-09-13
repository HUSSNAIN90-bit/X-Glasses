import asyncio
import cv2
import hashlib
import logging
import shutil
import time
import uuid

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse

from app.schemas.vision import (
    CodeDetection,
    CombinedVisionResponse,
    CommandVisionResponse,
    Detection,
    FaceMatch,
    FrameAnalysis,
    FrameQuality,
    FrameQualityBatchResponse,
    MultiFrameAnalysisResponse,
    MultiFrameComparisonResponse,
    MovementResult,
    OCRResult,
    PersonDetection,
    PoseDetection,
    Relationship,
    TrackingResponse,
    TrackedFrameResponse,
    TrackedDetectionResponse,
    TrackedPersonResponse,
    VisionProcessing,
)

from app.services.vision import (
    detect_objects,
    get_relative_position,
    get_relative_vertical_position,
    is_face_inside_person,
)

from app.services.face_recognition import (
    recognize_faces,
)

from app.services.llm import (
    describe_scene,
    describe_image_command,
    VisionLLMAuthenticationError,
    VisionLLMConfigurationError,
    VisionLLMError,
    VisionLLMRequestError,
    VisionLLMTimeoutError,
)

from app.services.memory import (
    save_latest_frame,
)

from app.services.frame_quality import (
    calculate_blur_score,
    calculate_brightness,
    evaluate_frame_quality,
    is_frame_good,
    optimize_image_for_vision,
)

from app.services.code_detection import detect_codes
from app.core.config import settings
from app.services.face_enrollment import (
    enroll_introduced_person,
    parse_face_introduction,
)

from app.services.frame_comparison import (
    Movement,
    compare_people,
    compare_objects,
    estimate_camera_motion,
)

from app.services.frame_similarity import (
    are_frames_duplicates,
)

from app.services.tracking import (
    track_pose_frames,
)

from app.services.pose import (
    detect_pose,
    match_pose_to_person,
)

from app.services.track_identity import (
    associate_faces_with_tracks,
)

from app.services.relationships import (
    HoldingState,
    detect_holding_relationships,
    get_memory_holding_relationships,
)

from app.services.vision import detect_objects

router = APIRouter(
    prefix="/api/vision",
    tags=["Vision"],
)
command_router = APIRouter(
    prefix="/vision",
    tags=["Vision"],
)

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {
    "image/bmp": ".bmp",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}
ALLOWED_IMAGE_SUFFIXES = set(IMAGE_SUFFIXES.values()) | {".jpeg"}


# =========================================================
# COMMAND-TRIGGERED MULTI-FRAME VISION
# =========================================================

@router.post("/command", response_model=CommandVisionResponse)
@command_router.post("/command", response_model=CommandVisionResponse)
@router.post(
    "/analyze-command-multi",
    response_model=CommandVisionResponse,
    include_in_schema=False,
)
async def analyze_command_multi(
    session_id: str = Form(...),
    command: str = Form(...),
    language: str = Form("en"),
    frames: list[UploadFile] | None = File(None),
    images: list[UploadFile] | None = File(None),
) -> Any:
    request_id = str(uuid.uuid4())
    request_started = time.perf_counter()
    session_id = session_id.strip()
    command = command.strip()
    language = language.strip() or "en"

    if not session_id or len(session_id) > 128:
        raise HTTPException(status_code=400, detail="A valid session_id is required.")
    if not command:
        raise HTTPException(status_code=400, detail="Command is required.")
    if len(command) > 2000:
        raise HTTPException(status_code=400, detail="Command is too long.")
    if len(language) > 16:
        raise HTTPException(status_code=400, detail="Language value is too long.")
    if frames and images:
        raise HTTPException(
            status_code=400,
            detail="Send frames using either 'frames' or legacy 'images', not both.",
        )

    uploads = frames or images or []
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one frame is required.")
    if len(uploads) > 3:
        raise HTTPException(status_code=400, detail="Send no more than 3 frames.")

    safe_session_id = session_id.replace("\r", "").replace("\n", "")
    logger.info(
        "vision_command received request_id=%s session_id=%s frame_count=%d",
        request_id,
        safe_session_id,
        len(uploads),
    )

    frame_quality: list[FrameQuality] = []
    saved: list[tuple[int, str]] = []
    detector_status = {
        "yolo": "not_run",
        "face_recognition": "not_run",
        "ocr": "unavailable",
        "barcode_qr": "not_run",
    }
    durations_ms: dict[str, float] = {}

    def processing(
        *,
        mode: str = "command_triggered",
        selected_frame: int | None = None,
        quality_score: float | None = None,
        llm_attempted: bool = False,
        llm_used: bool = False,
    ) -> VisionProcessing:
        return VisionProcessing(
            mode=mode,
            request_id=request_id,
            frame_count=len(saved),
            selected_frame=selected_frame,
            quality_score=quality_score,
            llm_attempted=llm_attempted,
            llm_used=llm_used,
            ai_called=llm_attempted,
            yolo_verified=detector_status["yolo"] == "ok",
            detectors=detector_status,
            durations_ms=durations_ms,
            frame_quality=frame_quality,
        )

    def error_response(
        *,
        status_code: int,
        reply: str,
        selected_frame: int | None = None,
        quality_score: float | None = None,
        objects: list[Detection] | None = None,
        face_matches: list[FaceMatch] | None = None,
        barcodes: list[CodeDetection] | None = None,
        llm_attempted: bool = False,
    ) -> JSONResponse:
        durations_ms.setdefault(
            "total",
            round((time.perf_counter() - request_started) * 1000, 1),
        )
        response = CommandVisionResponse(
            success=False,
            session_id=session_id,
            command=command,
            reply=reply,
            language=language,
            objects=objects or [],
            face_matches=face_matches or [],
            ocr=[],
            barcodes=barcodes or [],
            processing=processing(
                selected_frame=selected_frame,
                quality_score=quality_score,
                llm_attempted=llm_attempted,
            ),
        )
        return JSONResponse(
            status_code=status_code,
            content=response.model_dump(mode="json"),
        )

    async def save_selected_frame(selected_path: str) -> None:
        frames_dir = Path(__file__).resolve().parents[2] / "data" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_name = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
        latest_frame_path = frames_dir / f"{frame_name}.jpg"
        await asyncio.to_thread(shutil.copyfile, selected_path, latest_frame_path)
        save_latest_frame(session_id=session_id, image_path=str(latest_frame_path))

    try:
        for index, image in enumerate(uploads):
            if image.content_type and not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail="Every uploaded frame must be an image.",
                )

            image_bytes = await image.read(settings.max_vision_frame_bytes + 1)
            await image.close()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Uploaded frames cannot be empty.")
            if len(image_bytes) > settings.max_vision_frame_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="An uploaded frame is too large.",
                )

            suffix = IMAGE_SUFFIXES.get(image.content_type or "")
            if suffix is None:
                candidate_suffix = Path(image.filename or "").suffix.lower()
                suffix = (
                    candidate_suffix
                    if candidate_suffix in ALLOWED_IMAGE_SUFFIXES
                    else ".jpg"
                )

            with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(image_bytes)
            saved.append((index, temp_path))

        quality_started = time.perf_counter()
        metrics = await asyncio.gather(
            *[
                asyncio.to_thread(
                    evaluate_frame_quality,
                    path,
                    80.0,
                    35.0,
                    220.0,
                )
                for _, path in saved
            ]
        )
        durations_ms["quality"] = round(
            (time.perf_counter() - quality_started) * 1000,
            1,
        )

        for (index, _), item in zip(saved, metrics):
            if item.reason == "invalid_image":
                raise HTTPException(
                    status_code=400,
                    detail="One or more uploaded frames is not a valid image.",
                )
            if item.width * item.height > settings.max_vision_frame_pixels:
                raise HTTPException(
                    status_code=413,
                    detail="An uploaded frame has oversized dimensions.",
                )
            frame_quality.append(
                FrameQuality(
                    index=index,
                    good=item.good,
                    blur_score=item.blur_score,
                    brightness=item.brightness,
                    width=item.width,
                    height=item.height,
                    exposure_score=item.exposure_score,
                    quality_score=item.quality_score,
                    reason=item.reason,
                )
            )

        candidates = [
            (idx, path, q) for (idx, path), q in zip(saved, frame_quality) if q.good
        ]
        if not candidates:
            logger.info(
                "vision_command rejected request_id=%s reason=no_usable_frame",
                request_id,
            )
            return error_response(
                status_code=200,
                reply="Please hold the camera steady and try again.",
            )

        selected_idx, selected_path, selected_quality = max(
            candidates,
            key=lambda item: item[2].quality_score,
        )
        logger.info(
            "vision_command selected request_id=%s frame=%d quality_score=%.3f",
            request_id,
            selected_idx,
            selected_quality.quality_score,
        )

        introduction = parse_face_introduction(command)
        if introduction is not None:
            enrollment_started = time.perf_counter()
            enrollment = await asyncio.to_thread(
                enroll_introduced_person,
                selected_path,
                introduction,
            )
            durations_ms["face_enrollment"] = round(
                (time.perf_counter() - enrollment_started) * 1000,
                1,
            )
            detector_status["face_recognition"] = (
                "enrolled" if enrollment.success else "rejected"
            )
            await save_selected_frame(selected_path)
            durations_ms["total"] = round(
                (time.perf_counter() - request_started) * 1000,
                1,
            )
            logger.info(
                "vision_command enrollment_completed request_id=%s success=%s total_ms=%.1f",
                request_id,
                enrollment.success,
                durations_ms["total"],
            )
            return CommandVisionResponse(
                success=enrollment.success,
                session_id=session_id,
                command=command,
                reply=enrollment.reply,
                language=language,
                objects=[],
                face_matches=[],
                ocr=[],
                barcodes=[],
                processing=processing(
                    mode="face_enrollment",
                    selected_frame=selected_idx,
                    quality_score=selected_quality.quality_score,
                ),
            )

        async def run_detector(
            name: str,
            detector: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(detector, *args, **kwargs)
                detector_status[name] = "ok"
                return result
            except Exception as exc:
                detector_status[name] = "failed"
                logger.warning(
                    "vision_command detector_failed request_id=%s detector=%s error_type=%s",
                    request_id,
                    name,
                    type(exc).__name__,
                )
                return []
            finally:
                durations_ms[name] = round(
                    (time.perf_counter() - started) * 1000,
                    1,
                )
                logger.info(
                    "vision_command detector_completed request_id=%s detector=%s "
                    "status=%s duration_ms=%.1f",
                    request_id,
                    name,
                    detector_status[name],
                    durations_ms[name],
                )

        objects, recognized_faces, raw_codes = await asyncio.gather(
            run_detector(
                "yolo",
                detect_objects,
                image_path=selected_path,
                confidence_threshold=0.40,
            ),
            run_detector("face_recognition", recognize_faces, selected_path),
            run_detector("barcode_qr", detect_codes, selected_path),
        )
        face_matches = [
            FaceMatch(
                name=face.name,
                recognized=face.name is not None,
                confidence=face.similarity if face.name is not None else None,
                detection_confidence=face.confidence,
            )
            for face in recognized_faces
        ]
        barcodes = [
            CodeDetection(format=code.format, value=code.value)
            for code in raw_codes
        ]
        ocr_results: list[OCRResult] = []

        try:
            optimized_image = await asyncio.to_thread(
                optimize_image_for_vision,
                selected_path,
                settings.vision_max_dimension,
                settings.vision_jpeg_quality,
            )
        except ValueError:
            return error_response(
                status_code=400,
                reply="I couldn't read that image. Please try another frame.",
                selected_frame=selected_idx,
                quality_score=selected_quality.quality_score,
                objects=objects,
                face_matches=face_matches,
                barcodes=barcodes,
            )

        llm_started = time.perf_counter()

        def llm_error_response(
            *,
            status_code: int,
            reply: str,
            error_type: str,
            attempted: bool,
            provider_status_code: int | None = None,
            provider_error_type: str | None = None,
        ) -> JSONResponse:
            durations_ms["openai"] = round(
                (time.perf_counter() - llm_started) * 1000,
                1,
            )
            durations_ms["total"] = round(
                (time.perf_counter() - request_started) * 1000,
                1,
            )
            logger.warning(
                "vision_command llm_failed request_id=%s error_type=%s "
                "provider_status=%s provider_error=%s openai_ms=%.1f",
                request_id,
                error_type,
                provider_status_code,
                provider_error_type,
                durations_ms["openai"],
            )
            return error_response(
                status_code=status_code,
                reply=reply,
                selected_frame=selected_idx,
                quality_score=selected_quality.quality_score,
                objects=objects,
                face_matches=face_matches,
                barcodes=barcodes,
                llm_attempted=attempted,
            )

        try:
            reply = await describe_image_command(
                image_bytes=optimized_image,
                command=command,
                detected_objects=objects,
                face_matches=face_matches,
                ocr_results=ocr_results,
                detected_codes=barcodes,
            )
        except VisionLLMConfigurationError:
            return llm_error_response(
                status_code=503,
                reply="Vision AI is not configured on the server.",
                error_type="configuration",
                attempted=False,
            )
        except VisionLLMAuthenticationError:
            return llm_error_response(
                status_code=502,
                reply="Vision AI authentication failed. Please contact support.",
                error_type="authentication",
                attempted=True,
            )
        except VisionLLMTimeoutError:
            return llm_error_response(
                status_code=504,
                reply="Vision analysis took too long. Please try again.",
                error_type="timeout",
                attempted=True,
            )
        except (VisionLLMRequestError, VisionLLMError) as exc:
            return llm_error_response(
                status_code=502,
                reply="I couldn't analyze the image right now. Please try again.",
                error_type="provider",
                attempted=True,
                provider_status_code=exc.provider_status_code,
                provider_error_type=exc.provider_error_type,
            )
        finally:
            durations_ms["openai"] = round(
                (time.perf_counter() - llm_started) * 1000,
                1,
            )

        await save_selected_frame(selected_path)

        durations_ms["total"] = round(
            (time.perf_counter() - request_started) * 1000,
            1,
        )
        logger.info(
            "vision_command completed request_id=%s openai_ms=%.1f total_ms=%.1f",
            request_id,
            durations_ms["openai"],
            durations_ms["total"],
        )

        return CommandVisionResponse(
            success=True,
            session_id=session_id,
            command=command,
            reply=reply,
            language=language,
            objects=objects,
            face_matches=face_matches,
            ocr=ocr_results,
            barcodes=barcodes,
            processing=processing(
                selected_frame=selected_idx,
                quality_score=selected_quality.quality_score,
                llm_attempted=True,
                llm_used=True,
            ),
        )
    finally:
        for _, path in saved:
            Path(path).unlink(missing_ok=True)

# =========================================================
# ANALYZE IMAGE
# =========================================================

@router.post(
    "/analyze",
    response_model=CombinedVisionResponse,
)
async def analyze_frame(
    session_id: str = Form(...),
    image: UploadFile = File(...),
) -> CombinedVisionResponse:

    suffix = Path(
        image.filename or "frame.jpg"
    ).suffix

    # -----------------------------------------------------
    # CREATE TEMPORARY FILE
    # -----------------------------------------------------

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

        # =================================================
        # SAVE PERSISTENT LATEST FRAME
        # =================================================

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

        shutil.copyfile(
            temp_path,
            latest_frame_path,
        )

        save_latest_frame(
            session_id=session_id,
            image_path=str(
                latest_frame_path
            ),
        )

        # =================================================
        # READ IMAGE
        # =================================================

        frame = cv2.imread(
            temp_path
        )

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unable to read uploaded image."
                ),
            )

        image_height, image_width = (
            frame.shape[:2]
        )

        # =================================================
        # YOLO26 OBJECT DETECTION
        # =================================================

        objects: list[Detection] = (
            detect_objects(
                image_path=temp_path,
                confidence_threshold=0.40,
            )
        )

        poses: list[PoseDetection] = detect_pose(
            temp_path
        )

        # =================================================
        # INSIGHTFACE RECOGNITION
        # =================================================

        face_results = recognize_faces(
            temp_path
        )

        # =================================================
        # FIND YOLO PERSON OBJECTS
        # =================================================

        person_objects = [
            detection
            for detection in objects
            if detection.class_name == "person"
        ]

        # =================================================
        # MATCH FACES TO PERSONS
        # =================================================

        people: list[
            PersonDetection
        ] = []

        for index, face in enumerate(
            face_results
        ):

            # ---------------------------------------------
            # FACE BOX
            # ---------------------------------------------

            face_box = (
                face.x1,
                face.y1,
                face.x2,
                face.y2,
            )

            # ---------------------------------------------
            # MATCH FACE TO YOLO PERSON
            # ---------------------------------------------

            matched_person_index: int | None = None

            for person_index, person in enumerate(
                person_objects
            ):

                person_box = (
                    person.x1,
                    person.y1,
                    person.x2,
                    person.y2,
                )

                if is_face_inside_person(
                    face_box,
                    person_box,
                ):
                    matched_person_index = (
                        person_index
                    )
                    break

            # ---------------------------------------------
            # HORIZONTAL POSITION
            # ---------------------------------------------

            horizontal_position = (
                get_relative_position(
                    box=face_box,
                    image_width=float(
                        image_width
                    ),
                )
            )

            # ---------------------------------------------
            # VERTICAL POSITION
            # ---------------------------------------------

            vertical_position = (
                get_relative_vertical_position(
                    face_box=face_box,
                    image_height=float(
                        image_height
                    ),
                )
            )

            matched_pose = None

            if matched_person_index is not None:
                matched_person = person_objects[
                    matched_person_index
                ]

                matched_pose = match_pose_to_person(
                    person_bbox=(
                        matched_person.x1,
                        matched_person.y1,
                        matched_person.x2,
                        matched_person.y2,
                    ),
                    poses=poses,
                )

            # ---------------------------------------------
            # SAVE PERSON RESULT
            # ---------------------------------------------

            people.append(
                PersonDetection(
                    face_id=(
                        f"face-{index + 1}"
                    ),
                    person_index=(
                        matched_person_index
                    ),
                    name=face.name,
                    similarity=(
                        face.similarity
                    ),
                    recognized=(
                        face.name is not None
                    ),
                    confidence=(
                        face.confidence
                    ),
                    x1=face.x1,
                    y1=face.y1,
                    x2=face.x2,
                    y2=face.y2,
                    person_bbox=(
                        matched_person.bbox
                        if matched_person
                        else None
                    ),
                    relative_position=(
                        horizontal_position
                    ),
                    vertical_position=(
                        vertical_position
                    ),
                    posture=(
                        matched_pose.posture
                        if matched_pose
                        else None
                    ),
                    posture_confidence=(
                        matched_pose.confidence
                        if matched_pose
                        else None
                    ),
                )
            )

        # =================================================
        # LLM SCENE DESCRIPTION
        # =================================================

        scene_description = (
            await describe_scene(
                objects=objects,
                people=people,
            )
        )
        
        relationships: list[Relationship] = []
        holding_states: dict[tuple[int, str], HoldingState] = {}

        for person in people:
            if person.person_index is None:
                continue

            matched_pose = next(
                (
                    pose
                    for pose in poses
                    if pose.person_index == person.person_index
                ),
                None,
            )

            if matched_pose is None:
                continue

            detected_relationships = detect_holding_relationships(
                person=person,
                keypoints=matched_pose.keypoints,
                objects=objects,
                states=holding_states,
            )

            relationships.extend(
                Relationship(
                    subject=relationship.subject,
                    relation=relationship.relation,
                    object=relationship.object,
                    confidence=relationship.confidence,
                )
                for relationship in detected_relationships
            )
        
        # =================================================
        # FINAL RESPONSE
        # =================================================

        return CombinedVisionResponse(
            success=True,
            scene=scene_description,
            objects=objects,
            people=people,
            poses=poses,
            relationships=relationships,
        )

    finally:

        # -------------------------------------------------
        # DELETE ONLY TEMPORARY FILE
        # -------------------------------------------------

        Path(
            temp_path
        ).unlink(
            missing_ok=True
        )


# =========================================================
# SAVE LATEST CAMERA FRAME
# =========================================================

@router.post(
    "/frame",
)
async def upload_frame(
    session_id: str = Form(...),
    image: UploadFile = File(...),
) -> dict[str, str | bool]:

    frames_dir = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "frames"
    )

    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_path = (
        frames_dir
        / f"{session_id}.jpg"
    )

    try:

        # -------------------------------------------------
        # SAVE FRAME
        # -------------------------------------------------

        with frame_path.open(
            "wb"
        ) as output_file:

            shutil.copyfileobj(
                image.file,
                output_file,
            )

        # -------------------------------------------------
        # UPDATE MEMORY
        # -------------------------------------------------

        save_latest_frame(
            session_id=session_id,
            image_path=str(
                frame_path
            ),
        )

        return {
            "success": True,
            "message": (
                "Latest camera frame saved."
            ),
        }

    finally:

        await image.close()
        
@router.post(
    "/frame-quality",
)
async def check_frame_quality(
    image: UploadFile = File(...),
) -> dict[str, float | bool]:

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

        blur_score = (
            calculate_blur_score(
                temp_path
            )
        )

        brightness = (
            calculate_brightness(
                temp_path
            )
        )

        good = is_frame_good(
            temp_path
        )

        return {
            "good": good,
            "blur_score": blur_score,
            "brightness": brightness,
        }

    finally:

        Path(
            temp_path
        ).unlink(
            missing_ok=True
        )
        
@router.post(
    "/frame-quality-batch",
    response_model=FrameQualityBatchResponse,
)
async def check_frame_quality_batch(
    image1: UploadFile = File(...),
    image2: UploadFile | None = File(None),
    image3: UploadFile | None = File(None),
    image4: UploadFile | None = File(None),
    image5: UploadFile | None = File(None),
) -> FrameQualityBatchResponse:

    images: list[UploadFile] = [
        image1,
    ]

    if image2 is not None:
        images.append(image2)

    if image3 is not None:
        images.append(image3)

    if image4 is not None:
        images.append(image4)

    if image5 is not None:
        images.append(image5)

    frame_quality: list[FrameQuality] = []

    for index, image in enumerate(images):

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

            blur_score = (
                calculate_blur_score(
                    temp_path
                )
            )

            brightness = (
                calculate_brightness(
                    temp_path
                )
            )

            good = is_frame_good(
                temp_path,
                blur_threshold=100.0,
            )

            frame_quality.append(
                FrameQuality(
                    index=index,
                    good=good,
                    blur_score=blur_score,
                    brightness=brightness,
                )
            )

        finally:

            Path(
                temp_path
            ).unlink(
                missing_ok=True
            )

        await image.close()

    accepted_frames = sum(
        1
        for frame in frame_quality
        if frame.good
    )

    return FrameQualityBatchResponse(
        success=True,
        accepted_frames=accepted_frames,
        total_frames=len(images),
        frame_quality=frame_quality,
        message=(
            "Frames evaluated successfully."
        ),
    )
    
@router.post(
    "/analyze-frames",
    response_model=MultiFrameAnalysisResponse,
)
async def analyze_frames(
    image1: UploadFile = File(...),
    image2: UploadFile | None = File(None),
    image3: UploadFile | None = File(None),
    image4: UploadFile | None = File(None),
    image5: UploadFile | None = File(None),
) -> MultiFrameAnalysisResponse:

    images: list[UploadFile] = [
        image1,
    ]

    if image2 is not None:
        images.append(image2)

    if image3 is not None:
        images.append(image3)

    if image4 is not None:
        images.append(image4)

    if image5 is not None:
        images.append(image5)

    analyzed_frames: list[
        FrameAnalysis
    ] = []

    for index, image in enumerate(images):

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

            # ---------------------------------------------
            # FRAME QUALITY
            # ---------------------------------------------

            blur_score = (
                calculate_blur_score(
                    temp_path
                )
            )

            brightness = (
                calculate_brightness(
                    temp_path
                )
            )

            good = is_frame_good(
                temp_path,
                blur_threshold=100.0,
                min_brightness=35.0,
                max_brightness=220.0,
            )

            # ---------------------------------------------
            # REJECT BAD FRAME
            # ---------------------------------------------

            if not good:
                continue

            # ---------------------------------------------
            # YOLO26
            # ---------------------------------------------

            objects: list[Detection] = (
                detect_objects(
                    image_path=temp_path,
                    confidence_threshold=0.40,
                )
            )

            # ---------------------------------------------
            # INSIGHTFACE
            # ---------------------------------------------

            face_results = recognize_faces(
                temp_path
            )

            # ---------------------------------------------
            # READ IMAGE SIZE
            # ---------------------------------------------

            frame = cv2.imread(
                temp_path
            )

            if frame is None:
                continue

            image_height, image_width = (
                frame.shape[:2]
            )

            # ---------------------------------------------
            # YOLO PERSON OBJECTS
            # ---------------------------------------------

            person_objects = [
                detection
                for detection in objects
                if detection.class_name == "person"
            ]

            # ---------------------------------------------
            # BUILD PEOPLE
            # ---------------------------------------------

            people: list[
                PersonDetection
            ] = []

            for face_index, face in enumerate(
                face_results
            ):

                face_box = (
                    face.x1,
                    face.y1,
                    face.x2,
                    face.y2,
                )

                matched_person_index: int | None = None

                for person_index, person in enumerate(
                    person_objects
                ):

                    person_box = (
                        person.x1,
                        person.y1,
                        person.x2,
                        person.y2,
                    )

                    if is_face_inside_person(
                        face_box,
                        person_box,
                    ):
                        matched_person_index = (
                            person_index
                        )
                        break

                horizontal_position = (
                    get_relative_position(
                        box=face_box,
                        image_width=float(
                            image_width
                        ),
                    )
                )

                vertical_position = (
                    get_relative_vertical_position(
                        face_box=face_box,
                        image_height=float(
                            image_height
                        ),
                    )
                )

                people.append(
                    PersonDetection(
                        face_id=(
                            f"face-{face_index + 1}"
                        ),
                        person_index=(
                            matched_person_index
                        ),
                        name=face.name,
                        similarity=(
                            face.similarity
                        ),
                        recognized=(
                            face.name is not None
                        ),
                        confidence=(
                            face.confidence
                        ),
                        x1=face.x1,
                        y1=face.y1,
                        x2=face.x2,
                        y2=face.y2,
                        relative_position=(
                            horizontal_position
                        ),
                        vertical_position=(
                            vertical_position
                        ),
                    )
                )

            # ---------------------------------------------
            # SAVE FRAME RESULT
            # ---------------------------------------------

            analyzed_frames.append(
                FrameAnalysis(
                    index=index,
                    blur_score=blur_score,
                    brightness=brightness,
                    objects=objects,
                    people=people,
                )
            )

        finally:

            Path(
                temp_path
            ).unlink(
                missing_ok=True
            )

            await image.close()

    return MultiFrameAnalysisResponse(
        success=True,
        total_frames=len(images),
        analyzed_frames=len(
            analyzed_frames
        ),
        frames=analyzed_frames,
    )
    
@router.post(
    "/compare-frames",
    response_model=MultiFrameComparisonResponse,
)
async def compare_frames_endpoint(
    analyses: list[FrameAnalysis],
) -> MultiFrameComparisonResponse:

    if len(analyses) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least 2 analyzed frames "
                "are required."
            ),
        )

    person_movements: list[MovementResult] = []
    object_movements: list[MovementResult] = []

    for index in range(
        len(analyses) - 1
    ):

        first = analyses[index]
        second = analyses[index + 1]

        for movement in compare_people(
            first=first,
            second=second,
        ):
            person_movements.append(
                MovementResult(
                    label=movement.label,
                    from_position=(
                        movement.from_position
                    ),
                    to_position=(
                        movement.to_position
                    ),
                    movement_distance=(
                        movement.distance
                    ),
                    direction=movement.direction,
                )
            )

        for movement in compare_objects(
            first=first,
            second=second,
        ):
            object_movements.append(
                MovementResult(
                    label=movement.label,
                    from_position=(
                        movement.from_position
                    ),
                    to_position=(
                        movement.to_position
                    ),
                    movement_distance=(
                        movement.distance
                    ),
                    direction=movement.direction,
                )
            )

    return MultiFrameComparisonResponse(
        success=True,
        frames_compared=len(
            analyses
        ),
        person_movements=person_movements,
        object_movements=object_movements,
    )

@router.post(
    "/analyze-and-compare",
    response_model=MultiFrameComparisonResponse,
)
async def analyze_and_compare(
    image1: UploadFile = File(...),
    image2: UploadFile | None = File(None),
    image3: UploadFile | None = File(None),
    image4: UploadFile | None = File(None),
    image5: UploadFile | None = File(None),
) -> MultiFrameComparisonResponse:

    images: list[UploadFile] = [
        image1,
    ]

    if image2 is not None:
        images.append(image2)

    if image3 is not None:
        images.append(image3)

    if image4 is not None:
        images.append(image4)

    if image5 is not None:
        images.append(image5)

    analyses: list[FrameAnalysis] = []

    frame_paths: list[str] = []

    # =====================================================
    # SAVE UPLOADED FRAMES
    # =====================================================

    for image in images:

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

        frame_paths.append(
            temp_path
        )

        await image.close()

    # =====================================================
    # REMOVE CONSECUTIVE DUPLICATE FRAMES
    # =====================================================

    unique_frame_paths: list[str] = []

    for frame_path in frame_paths:

        if not unique_frame_paths:
            unique_frame_paths.append(
                frame_path
            )
            continue

        previous_path = (
            unique_frame_paths[-1]
        )

        if are_frames_duplicates(
            first_path=previous_path,
            second_path=frame_path,
        ):
            Path(frame_path).unlink(
                missing_ok=True
            )
            continue

        unique_frame_paths.append(
            frame_path
        )

    # =====================================================
    # ANALYZE UNIQUE FRAMES
    # =====================================================

    for index, temp_path in enumerate(
        unique_frame_paths
    ):

        try:

            # ---------------------------------------------
            # FRAME QUALITY
            # ---------------------------------------------

            blur_score = calculate_blur_score(
                temp_path
            )

            brightness = calculate_brightness(
                temp_path
            )

            good = is_frame_good(
                temp_path,
                blur_threshold=100.0,
                min_brightness=35.0,
                max_brightness=220.0,
            )

            if not good:
                continue

            # ---------------------------------------------
            # YOLO26
            # ---------------------------------------------

            objects: list[Detection] = (
                detect_objects(
                    image_path=temp_path,
                    confidence_threshold=0.40,
                )
            )

            # ---------------------------------------------
            # INSIGHTFACE
            # ---------------------------------------------

            face_results = recognize_faces(
                temp_path
            )

            # ---------------------------------------------
            # IMAGE SIZE
            # ---------------------------------------------

            frame = cv2.imread(
                temp_path
            )

            if frame is None:
                continue

            image_height, image_width = (
                frame.shape[:2]
            )

            # ---------------------------------------------
            # PERSON OBJECTS
            # ---------------------------------------------

            person_objects = [
                detection
                for detection in objects
                if detection.class_name == "person"
            ]

            # ---------------------------------------------
            # PEOPLE
            # ---------------------------------------------

            people: list[
                PersonDetection
            ] = []

            for face_index, face in enumerate(
                face_results
            ):

                face_box = (
                    face.x1,
                    face.y1,
                    face.x2,
                    face.y2,
                )

                matched_person_index: int | None = None

                for person_index, person in enumerate(
                    person_objects
                ):

                    person_box = (
                        person.x1,
                        person.y1,
                        person.x2,
                        person.y2,
                    )

                    if is_face_inside_person(
                        face_box,
                        person_box,
                    ):
                        matched_person_index = (
                            person_index
                        )
                        break

                horizontal_position = (
                    get_relative_position(
                        box=face_box,
                        image_width=float(
                            image_width
                        ),
                    )
                )

                vertical_position = (
                    get_relative_vertical_position(
                        face_box=face_box,
                        image_height=float(
                            image_height
                        ),
                    )
                )

                people.append(
                    PersonDetection(
                        face_id=(
                            f"face-{face_index + 1}"
                        ),
                        person_index=(
                            matched_person_index
                        ),
                        name=face.name,
                        similarity=(
                            face.similarity
                        ),
                        recognized=(
                            face.name is not None
                        ),
                        confidence=face.confidence,
                        x1=face.x1,
                        y1=face.y1,
                        x2=face.x2,
                        y2=face.y2,
                        relative_position=(
                            horizontal_position
                        ),
                        vertical_position=(
                            vertical_position
                        ),
                    )
                )

            # ---------------------------------------------
            # SAVE ANALYSIS
            # ---------------------------------------------

            analyses.append(
                FrameAnalysis(
                    index=index,
                    blur_score=blur_score,
                    brightness=brightness,
                    objects=objects,
                    people=people,
                )
            )

        except Exception:
            # Bad frame should not break
            # the complete batch.
            continue

    # =====================================================
    # NEED AT LEAST TWO GOOD FRAMES
    # =====================================================

    if len(analyses) < 2:

        for path in frame_paths:
            Path(path).unlink(
                missing_ok=True
            )

        raise HTTPException(
            status_code=400,
            detail=(
                "At least 2 good frames "
                "are required for comparison."
            ),
        )

    # =====================================================
    # COMPARE FRAMES
    # =====================================================

    person_movements: list[Movement] = []
    object_movements: list[Movement] = []

    analysis_paths: list[tuple[FrameAnalysis, str]] = []

    for analysis in analyses:
        if 0 <= analysis.index < len(unique_frame_paths):
            analysis_paths.append(
                (
                    analysis,
                    unique_frame_paths[analysis.index],
                )
            )

    try:
        for index in range(len(analysis_paths) - 1):
            first_analysis, first_path = analysis_paths[index]
            second_analysis, second_path = analysis_paths[index + 1]

            camera_dx, camera_dy = estimate_camera_motion(
                previous_path=first_path,
                current_path=second_path,
            )

            person_movements.extend(
                compare_people(
                    first=first_analysis,
                    second=second_analysis,
                    camera_dx=camera_dx,
                    camera_dy=camera_dy,
                )
            )

            object_movements.extend(
                compare_objects(
                    first=first_analysis,
                    second=second_analysis,
                    camera_dx=camera_dx,
                    camera_dy=camera_dy,
                )
            )

        return MultiFrameComparisonResponse(
            success=True,
            frames_compared=len(analysis_paths),
            person_movements=[
                MovementResult(
                    label=movement.label,
                    from_position=movement.from_position,
                    to_position=movement.to_position,
                    movement_distance=movement.distance,
                    direction=movement.direction,
                )
                for movement in person_movements
            ],
            object_movements=[
                MovementResult(
                    label=movement.label,
                    from_position=movement.from_position,
                    to_position=movement.to_position,
                    movement_distance=movement.distance,
                    direction=movement.direction,
                )
                for movement in object_movements
            ],
        )

    finally:
        for path in frame_paths:
            Path(path).unlink(missing_ok=True)
            
@router.post(
    "/track-frames",
    response_model=TrackingResponse,
)
async def track_uploaded_frames(
    image1: UploadFile = File(...),
    image2: UploadFile | None = File(None),
    image3: UploadFile | None = File(None),
    image4: UploadFile | None = File(None),
    image5: UploadFile | None = File(None),
) -> TrackingResponse:

    images: list[UploadFile] = [
        image1,
    ]

    if image2 is not None:
        images.append(image2)

    if image3 is not None:
        images.append(image3)

    if image4 is not None:
        images.append(image4)

    if image5 is not None:
        images.append(image5)

    temp_paths: list[str] = []

    # --------------------------------------------------
    # TEMPORAL HOLDING MEMORY
    # --------------------------------------------------

    holding_states: dict[
        tuple[int, str],
        HoldingState,
    ] = {}

    max_missed_frames = 3
    min_confidence = 0.10

    try:

        # --------------------------------------------------
        # SAVE UPLOADED FRAMES
        # --------------------------------------------------

        for image in images:

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

                temp_paths.append(
                    temp_file.name
                )

            await image.close()

        # --------------------------------------------------
        # YOLO26 + BYTE TRACK + POSE
        # --------------------------------------------------

        tracked_frames = track_pose_frames(
            image_paths=temp_paths,
            confidence_threshold=0.40,
        )

        frames: list[
            TrackedFrameResponse
        ] = []

        # --------------------------------------------------
        # PROCESS EACH FRAME
        # --------------------------------------------------

        for index, detections in enumerate(
            tracked_frames
        ):

            # ----------------------------------------------
            # FACE RECOGNITION
            # ----------------------------------------------

            associations = (
                associate_faces_with_tracks(
                    image_path=temp_paths[index],
                    tracked_detections=detections,
                )
            )

            objects: list[Detection] = [
                Detection(
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                )
                for detection in detections
                if detection.class_name != "person"
            ]

            # ----------------------------------------------
            # DETECTED OBJECT CLASSES IN CURRENT FRAME
            # ----------------------------------------------

            detected_classes: set[str] = {
                obj.class_name
                for obj in objects
                if obj.confidence >= min_confidence
            }

            people: list[
                TrackedPersonResponse
            ] = []

            frame_relationships: list[
                Relationship
            ] = []

            # ----------------------------------------------
            # PROCESS TRACKED DETECTIONS
            # ----------------------------------------------

            for detection in detections:

                if detection.class_name != "person":
                    continue

                # ------------------------------------------
                # FACE RECOGNITION
                # ------------------------------------------

                name, similarity = (
                    associations.get(
                        detection.track_id,
                        (None, None),
                    )
                )

                people.append(
                    TrackedPersonResponse(
                    track_id=detection.track_id,
                    name=name,
                    recognized=(
                        name is not None
                    ),
                    recognition_similarity=similarity,
                    confidence=detection.confidence,
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                    posture=detection.posture,
                    posture_confidence=detection.pose_confidence,
                    keypoints=detection.keypoints,
                    )
)

                # ------------------------------------------
                # POSE KEYPOINTS
                # ------------------------------------------

                if detection.keypoints is None:
                    continue

                # ------------------------------------------
                # CREATE PERSON OBJECT FOR RELATIONSHIP
                # ------------------------------------------

                person = PersonDetection(
                    face_id=(
                        f"track-{detection.track_id}"
                    ),
                    person_index=detection.track_id,
                    name=name,
                    similarity=similarity,
                    recognized=(
                        name is not None
                    ),
                    confidence=detection.confidence,
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                    relative_position="center",
                    vertical_position="center",
                )

                # ------------------------------------------
                # UPDATE STATES FOR OBJECTS THAT DISAPPEARED
                # ------------------------------------------

                for (
                    state_key,
                    state,
                ) in holding_states.items():

                    track_id, class_name = state_key

                    if track_id != detection.track_id:
                        continue

                    if class_name in detected_classes:
                        continue

                    if not state.holding:
                        continue

                    state.missed_frames += 1

                    if (
                        state.missed_frames
                        > max_missed_frames
                    ):
                        state.holding = False
                        state.missed_frames = 0
                        state.last_confidence = 0.0

                # ------------------------------------------
                # DETECT CURRENT HOLDING
                # ------------------------------------------

                detect_holding_relationships(
                    person=person,
                    keypoints=detection.keypoints,
                    objects=objects,
                    states=holding_states,
                    min_confidence=min_confidence,
                    max_missed_frames=max_missed_frames,
                )

                # ------------------------------------------
                # GET MEMORY RELATIONSHIPS
                # ------------------------------------------

                memory_relationships = (
                    get_memory_holding_relationships(
                        person=person,
                        states=holding_states,
                    )
                )

                # ------------------------------------------
                # CONVERT TO API RELATIONSHIPS
                # ------------------------------------------

                frame_relationships.extend(
                    Relationship(
                        subject=relationship.subject,
                        relation=relationship.relation,
                        object=relationship.object,
                        confidence=relationship.confidence,
                    )
                    for relationship in memory_relationships
                )

            # ----------------------------------------------
            # BUILD FRAME RESPONSE
            # ----------------------------------------------

            frames.append(
                TrackedFrameResponse(
                    index=index,

                    detections=[
                        TrackedDetectionResponse(
                            track_id=detection.track_id,
                            class_name=detection.class_name,
                            confidence=detection.confidence,
                            x1=detection.x1,
                            y1=detection.y1,
                            x2=detection.x2,
                            y2=detection.y2,
                        )
                        for detection in detections
                    ],

                    people=people,

                    relationships=frame_relationships,
                )
            )

        # --------------------------------------------------
        # FINAL RESPONSE
        # --------------------------------------------------

        return TrackingResponse(
            success=True,
            frames=frames,
        )

    finally:

        for path in temp_paths:
            Path(path).unlink(
                missing_ok=True
            )
