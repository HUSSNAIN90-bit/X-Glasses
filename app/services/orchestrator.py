import asyncio

import cv2

from app.schemas.chat import (
    Intent,
    IntentResult,
)
from app.schemas.vision import (
    PersonDetection,
)
from app.services.face_recognition import (
    recognize_faces,
)
from app.services.llm import (
    describe_scene,
    generate_response,
    parse_intent,
)
from app.services.memory import (
    add_message,
    get_history,
    get_latest_frame,
)
from app.services.vision import (
    describe_person_position,
    detect_objects,
)

from app.services.face_database import (
    resolve_person_reference,
)
from app.services.face_enrollment import (
    enroll_introduced_person,
    parse_face_introduction,
)


# =========================================================
# FIND PERSON LOCATION
# =========================================================

async def find_person_location(
    session_id: str,
    person_reference: str,
) -> str:

    latest_frame = get_latest_frame(
        session_id
    )

    if latest_frame is None:
        return (
            "I don't have a camera frame yet."
        )

    # -----------------------------------------------------
    # RESOLVE PERSON REFERENCE
    # -----------------------------------------------------

    resolved_person = resolve_person_reference(
        person_reference
    )

    if resolved_person is None:
        return (
            f"I don't know which person "
            f"you mean by {person_reference}."
        )

    _, person_name = resolved_person

    # -----------------------------------------------------
    # READ CURRENT FRAME
    # -----------------------------------------------------

    frame = cv2.imread(
        latest_frame
    )

    if frame is None:
        return (
            "I couldn't read the latest "
            "camera frame."
        )

    _, image_width = frame.shape[:2]

    # -----------------------------------------------------
    # RECOGNIZE ALL FACES
    # -----------------------------------------------------

    face_results = recognize_faces(
        latest_frame
    )

    if not face_results:
        return (
            "I don't see anyone clearly "
            "in the current frame."
        )

    # -----------------------------------------------------
    # FIND REQUESTED PERSON
    # -----------------------------------------------------

    target_person = None

    for face in face_results:

        if face.name is None:
            continue

        if (
            face.name.lower()
            == person_name.lower()
        ):
            target_person = face
            break

    # -----------------------------------------------------
    # PERSON NOT FOUND
    # -----------------------------------------------------

    if target_person is None:
        return (
            f"I don't see {person_name} "
            "in the current frame."
        )

    # -----------------------------------------------------
    # CALCULATE POSITION
    # -----------------------------------------------------

    position = describe_person_position(
        x1=target_person.x1,
        x2=target_person.x2,
        image_width=float(
            image_width
        ),
    )

    return (
        f"{target_person.name} is "
        f"{position}."
    )
# =========================================================
# MAIN REQUEST PROCESSOR
# =========================================================

async def process_request(
    session_id: str,
    message: str,
) -> tuple[str, Intent]:

    introduction = parse_face_introduction(message)
    if introduction is not None:
        latest_frame = get_latest_frame(session_id)
        if latest_frame is None:
            response = "I need a clear camera frame before I can save them."
        else:
            enrollment = await asyncio.to_thread(
                enroll_introduced_person,
                latest_frame,
                introduction,
            )
            response = enrollment.reply

        add_message(session_id=session_id, role="user", content=message)
        add_message(session_id=session_id, role="assistant", content=response)
        return response, "face_enrollment"

    # -----------------------------------------------------
    # AI INTENT PARSER
    # -----------------------------------------------------

    parsed: IntentResult = await parse_intent(
        message=message
    )

    intent: Intent = parsed.intent

    history = get_history(
        session_id
    )

    # =====================================================
    # GENERAL
    # =====================================================

    if intent == "general":

        response = await generate_response(
            message=message,
            history=history,
        )

    # =====================================================
    # VISION
    # =====================================================

    elif intent == "vision":

        latest_frame = get_latest_frame(
            session_id
        )

        if latest_frame is None:

            response = (
                "I don't have a camera "
                "frame yet."
            )

        else:

            # ---------------------------------------------
            # YOLO26
            # ---------------------------------------------

            objects = detect_objects(
                image_path=latest_frame,
                confidence_threshold=0.40,
            )

            # ---------------------------------------------
            # INSIGHTFACE
            # ---------------------------------------------

            face_results = recognize_faces(
                latest_frame
            )

            # ---------------------------------------------
            # BUILD PEOPLE DATA
            # ---------------------------------------------

            people: list[
                PersonDetection
            ] = []

            for index, face in enumerate(
                face_results
            ):

                people.append(
                    PersonDetection(
                        face_id=(
                            f"face-{index + 1}"
                        ),
                        person_index=None,
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
                        relative_position=None,
                        vertical_position=None,
                    )
                )

            # ---------------------------------------------
            # LLM SCENE DESCRIPTION
            # ---------------------------------------------

            response = await describe_scene(
                objects=objects,
                people=people,
            )

    # =====================================================
    # PERSON LOCATION
    # =====================================================

    elif intent == "person_location":

        # -------------------------------------------------
        # CHECK PERSON REFERENCE
        # -------------------------------------------------

        if parsed.person is None:

            response = (
                "Tell me which person "
                "you want me to find."
            )

        else:

            response = (
                await find_person_location(
                    session_id=session_id,
                    person_reference=parsed.person,
                )
            )

    # =====================================================
    # OBJECT SEARCH
    # =====================================================

    elif intent == "object_search":

        if parsed.object_name is None:

            response = (
                "Tell me what object "
                "you want me to find."
            )

        else:

            response = (
                f"I need the object memory "
                f"system to find your "
                f"{parsed.object_name}."
            )

    # =====================================================
    # MEMORY
    # =====================================================

    elif intent == "memory":

        response = (
            "I need the memory system "
            "to check that."
        )

    # =====================================================
    # SAFETY
    # =====================================================

    else:

        raise RuntimeError(
            "Unsupported intent."
        )

    # =====================================================
    # SAVE CONVERSATION
    # =====================================================

    add_message(
        session_id=session_id,
        role="user",
        content=message,
    )

    add_message(
        session_id=session_id,
        role="assistant",
        content=response,
    )

    return response, intent
