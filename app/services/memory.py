from dataclasses import dataclass

from app.schemas.vision import Detection


@dataclass(frozen=True)
class Message:
    role: str
    content: str


_sessions: dict[str, list[Message]] = {}
_vision_results: dict[str, list[Detection]] = {}


def get_history(session_id: str) -> list[Message]:
    return _sessions.get(session_id, [])


def add_message(
    session_id: str,
    role: str,
    content: str,
) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = []

    _sessions[session_id].append(
        Message(
            role=role,
            content=content,
        )
    )


def save_vision_result(
    session_id: str,
    detections: list[Detection],
) -> None:
    _vision_results[session_id] = detections


def get_vision_result(
    session_id: str,
) -> list[Detection]:
    return _vision_results.get(session_id, [])


def clear_vision_result(
    session_id: str,
) -> None:
    _vision_results.pop(session_id, None)
    
_latest_frames: dict[str, str] = {}


def save_latest_frame(
    session_id: str,
    image_path: str,
) -> None:
    _latest_frames[session_id] = image_path


def get_latest_frame(
    session_id: str,
) -> str | None:
    return _latest_frames.get(session_id)


def clear_latest_frame(
    session_id: str,
) -> None:
    _latest_frames.pop(
        session_id,
        None,
    )