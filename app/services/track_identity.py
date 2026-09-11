import cv2

from app.services.face_recognition import (
    recognize_faces,
)
from app.services.tracking import (
    TrackedDetection,
)


def is_face_inside_person(
    face_box: tuple[
        float,
        float,
        float,
        float,
    ],
    person_box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> bool:

    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box

    face_center_x = (
        fx1 + fx2
    ) / 2

    face_center_y = (
        fy1 + fy2
    ) / 2

    return (
        px1 <= face_center_x <= px2
        and
        py1 <= face_center_y <= py2
    )


def associate_faces_with_tracks(
    image_path: str,
    tracked_detections: list[
        TrackedDetection
    ],
) -> dict[int, tuple[
    str | None,
    float | None,
]]:
    """
    Returns:

        track_id -> (
            recognized_name,
            similarity
        )
    """

    face_results = recognize_faces(
        image_path
    )

    person_tracks = [
        detection
        for detection in tracked_detections
        if detection.class_name == "person"
    ]

    associations: dict[
        int,
        tuple[str | None, float | None],
    ] = {}

    for face in face_results:

        face_box = (
            face.x1,
            face.y1,
            face.x2,
            face.y2,
        )

        for person in person_tracks:

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

                associations[
                    person.track_id
                ] = (
                    face.name,
                    face.similarity,
                )

                break

    return associations