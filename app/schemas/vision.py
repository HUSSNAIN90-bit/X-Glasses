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

    @property
    def bbox(self) -> list[float]:
        return [
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        ]


class VisionResponse(BaseModel):
    success: bool
    detections: list[Detection]


class PoseDetection(BaseModel):
    person_index: int
    posture: str
    confidence: float
    bbox: list[float]
    keypoints: list[list[float]]


class Relationship(BaseModel):
    subject: str
    relation: str
    object: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


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

    person_bbox: list[float] | None = None

    relative_position: str | None = None

    vertical_position: str | None = None

    posture: str | None = None

    posture_confidence: float | None = None

    @property
    def bbox(self) -> list[float]:
        return [
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        ]


# =========================================================
# COMBINED SINGLE-FRAME VISION
# =========================================================

class CombinedVisionResponse(BaseModel):
    success: bool

    scene: str

    objects: list[Detection]

    people: list[PersonDetection]

    poses: list[PoseDetection]

    relationships: list[Relationship] = Field(
        default_factory=lambda: list[Relationship](),
    )


# =========================================================
# FRAME QUALITY
# =========================================================

class FrameQuality(BaseModel):
    index: int

    good: bool

    blur_score: float

    brightness: float

    width: int = 0

    height: int = 0

    exposure_score: float = 0.0

    quality_score: float = 0.0

    reason: str | None = None


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
    posture: str | None
    posture_confidence: float | None
    keypoints: list[list[float]] | None

    posture: str | None = None

    posture_confidence: float | None = None

    keypoints: list[list[float]] | None = None


class TrackedFrameResponse(BaseModel):
    index: int

    detections: list[
        TrackedDetectionResponse
    ]

    people: list[
        TrackedPersonResponse
    ]

    relationships: list[
        Relationship
    ] = Field(
        default_factory=lambda: list[Relationship](),
    )


class TrackingResponse(BaseModel):
    success: bool

    frames: list[
        TrackedFrameResponse
    ]


class FaceMatch(BaseModel):
    name: str | None = None
    recognized: bool
    confidence: float | None = None
    detection_confidence: float


class OCRResult(BaseModel):
    text: str
    confidence: float | None = None


class CodePrice(BaseModel):
    value: float
    currency: str | None = None
    source: str
    kind: str


class CodeProduct(BaseModel):
    name: str | None = None
    brand: str | None = None
    category: str | None = None
    source: str | None = None
    prices: list[CodePrice] = Field(default_factory=list)
    price_note: str | None = None


class CodeDetection(BaseModel):
    format: str
    value: str
    product: CodeProduct | None = None


class VisionProcessing(BaseModel):
    mode: str = "command_triggered"
    request_id: str
    frame_count: int = 0
    selected_frame: int | None = None
    quality_score: float | None = None
    llm_attempted: bool = False
    llm_used: bool = False
    ai_called: bool = False
    yolo_verified: bool = False
    detectors: dict[str, str] = Field(default_factory=dict)
    durations_ms: dict[str, float] = Field(default_factory=dict)
    frame_quality: list[FrameQuality] = Field(default_factory=list)


class CommandVisionResponse(BaseModel):
    success: bool
    session_id: str
    command: str
    reply: str
    language: str = "en"
    objects: list[Detection] = Field(default_factory=list)
    face_matches: list[FaceMatch] = Field(default_factory=list)
    ocr: list[OCRResult] = Field(default_factory=list)
    barcodes: list[CodeDetection] = Field(default_factory=list)
    processing: VisionProcessing
