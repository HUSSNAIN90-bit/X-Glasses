from pydantic import BaseModel, Field


# =========================================================
# OBJECT DETECTION
# =========================================================

class Detection(BaseModel):
    class_name: str = Field(min_length=1)

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    x1: float
    y1: float
    x2: float
    y2: float

    relative_position: str | None = None
    vertical_position: str | None = None


class VisionResponse(BaseModel):
    success: bool
    detections: list[Detection]


# =========================================================
# PERSON / FACE DETECTION
# =========================================================

class PersonDetection(BaseModel):
    face_id: str

    person_index: int | None

    name: str | None

    similarity: float | None

    recognized: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    x1: float
    y1: float
    x2: float
    y2: float

    relative_position: str | None = None

    vertical_position: str | None = None


# =========================================================
# COMBINED SINGLE-FRAME VISION
# =========================================================

class CombinedVisionResponse(BaseModel):
    success: bool

    scene: str

    objects: list[Detection]

    people: list[PersonDetection]


# =========================================================
# FRAME QUALITY
# =========================================================

class FrameQuality(BaseModel):
    index: int

    good: bool

    blur_score: float

    brightness: float


class FrameQualityBatchResponse(BaseModel):
    success: bool

    accepted_frames: int

    total_frames: int

    frame_quality: list[FrameQuality]

    message: str


# =========================================================
# MULTI-FRAME ANALYSIS
# =========================================================

class FrameAnalysis(BaseModel):
    index: int

    blur_score: float

    brightness: float

    objects: list[Detection]

    people: list[PersonDetection]


class MultiFrameAnalysisResponse(BaseModel):
    success: bool

    total_frames: int

    analyzed_frames: int

    frames: list[FrameAnalysis]
    
class MovementResult(BaseModel):
    label: str
    from_position: str
    to_position: str
    movement_distance: float
    direction: str


class MultiFrameComparisonResponse(BaseModel):
    success: bool
    frames_compared: int
    person_movements: list[MovementResult]
    object_movements: list[MovementResult]

class TrackedDetectionResponse(BaseModel):
    track_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float



    
class TrackedPersonResponse(BaseModel):
    track_id: int

    name: str | None

    recognized: bool

    recognition_similarity: float | None

    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float


class TrackedFrameResponse(BaseModel):
    index: int

    detections: list[
        TrackedDetectionResponse
    ]

    people: list[
        TrackedPersonResponse
    ]


class TrackingResponse(BaseModel):
    success: bool

    frames: list[
        TrackedFrameResponse
    ]