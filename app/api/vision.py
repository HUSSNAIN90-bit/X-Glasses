import cv2
import shutil

from pathlib import Path
from tempfile import NamedTemporaryFile
# from typing import Any

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
    PoseDetection,
    Relationship,
    TrackingResponse,
    TrackedFrameResponse,
    TrackedDetectionResponse,
    TrackedPersonResponse,
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

            objects = detect_objects(
                temp_paths[index],
                confidence_threshold=min_confidence,
            )

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