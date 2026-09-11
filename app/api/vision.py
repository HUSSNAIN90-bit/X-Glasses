import cv2
import shutil

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.schemas.vision import (
    CombinedVisionResponse,
    Detection,
    FrameAnalysis,
    FrameQuality,
    FrameQualityBatchResponse,
    MultiFrameAnalysisResponse,
    MultiFrameComparisonResponse,
    MovementResult,
    PersonDetection,
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
)

from app.services.memory import (
    save_latest_frame,
)

from app.services.frame_quality import (
    calculate_blur_score,
    calculate_brightness,
    is_frame_good,
)

from app.services.frame_comparison import (
    compare_people,
    compare_objects,
)

router = APIRouter(
    prefix="/api/vision",
    tags=["Vision"],
)


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
                    face_box=face_box,
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
                    relative_position=(
                        horizontal_position
                    ),
                    vertical_position=(
                        vertical_position
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

        # =================================================
        # FINAL RESPONSE
        # =================================================

        return CombinedVisionResponse(
            success=True,
            scene=scene_description,
            objects=objects,
            people=people,
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
                        face_box=face_box,
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

    person_movements: list[Any] = []
    object_movements: list[Any] = []

    for index in range(
        len(analyses) - 1
    ):

        first = analyses[index]
        second = analyses[index + 1]

        person_movements.extend(
            compare_people(
                first=first,
                second=second,
            )
        )

        object_movements.extend(
            compare_objects(
                first=first,
                second=second,
            )
        )

    return MultiFrameComparisonResponse(
        success=True,
        frames_compared=len(
            analyses
        ),
        person_movements=[
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
            )
            for movement in person_movements
        ],
        object_movements=[
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
            )
            for movement in object_movements
        ],
    )