import base64
from functools import lru_cache
import json
import re
from typing import cast

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from app.core.config import settings
from app.schemas.vision import (
    CodeDetection,
    Detection,
    FaceMatch,
    OCRResult,
    PersonDetection,
)
from app.services.memory import Message


client = AsyncGroq(
    api_key=settings.llm_api_key,
)
from app.schemas.chat import IntentResult


class VisionLLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider_status_code: int | None = None,
        provider_error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_status_code = provider_status_code
        self.provider_error_type = provider_error_type


class VisionLLMConfigurationError(VisionLLMError):
    pass


class VisionLLMAuthenticationError(VisionLLMError):
    pass


class VisionLLMTimeoutError(VisionLLMError):
    pass


class VisionLLMRequestError(VisionLLMError):
    pass


VISION_ASSISTANT_INSTRUCTIONS = """You are the vision assistant for X-Glasses.

Analyze the supplied image and answer the user's command directly.
Use the image as the primary visual source. Use YOLO, OCR, barcode, QR, and
face-recognition metadata only as supporting context. Do not invent facts.
Do not identify a person unless trusted face-recognition metadata provides the
identity. Never invent an SKU, model number, price, barcode, or serial number.
Do not expose internal reasoning, analysis, planning, chain-of-thought, hidden
thoughts, or model deliberation. Return only the final answer intended for the
user. Keep normal answers to 1-2 short sentences unless the user explicitly
requests more detail. Use simple natural language suitable for speech. If the
requested information cannot be determined from the image or verified
metadata, say so briefly."""


@lru_cache(maxsize=1)
def get_openai_vision_client() -> AsyncOpenAI:
    api_key = settings.openai_api_key.strip()
    if not api_key:
        raise VisionLLMConfigurationError(
            "OPENAI_API_KEY is missing from the backend environment."
        )
    return AsyncOpenAI(
        api_key=api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=0,
    )


def clean_llm_response(text: str) -> str:
    """Return only user-facing text; never expose model reasoning."""
    if not text:
        return ""

    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    unclosed_think = re.search(r"<think\b[^>]*>", text, flags=re.IGNORECASE)
    if unclosed_think:
        hidden_tail = text[unclosed_think.end():]
        answer_match = re.search(
            r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*",
            hidden_tail,
            flags=re.IGNORECASE,
        )
        text = (
            text[:unclosed_think.start()] + hidden_tail[answer_match.end():]
            if answer_match
            else text[:unclosed_think.start()]
        )

    def clean_fence(match: re.Match[str]) -> str:
        language = match.group(1).strip().lower()
        body = match.group(2)
        if language in {
            "analysis",
            "reasoning",
            "think",
            "thought",
            "planning",
        }:
            return ""
        return body

    text = re.sub(
        r"```([^\n`]*)\n?(.*?)```",
        clean_fence,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"^\s*(?:analysis|reasoning|chain\s+of\s+thought|internal\s+analysis|"
        r"internal\s+planning|planning|model\s+deliberation)\s*:\s*.*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*(?:final\s+answer|answer)\s*:\s*",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


async def describe_image_command(
    image_bytes: bytes,
    command: str,
    detected_objects: list[Detection] | None = None,
    face_matches: list[FaceMatch] | None = None,
    ocr_results: list[OCRResult] | None = None,
    detected_codes: list[CodeDetection] | None = None,
) -> str:
    if not image_bytes:
        raise VisionLLMRequestError("The selected image is empty.")

    metadata = {
        "objects": [
            {
                "label": item.class_name,
                "confidence": round(item.confidence, 3),
                "horizontal_position": item.relative_position,
                "vertical_position": item.vertical_position,
            }
            for item in (detected_objects or [])[:30]
        ],
        "recognized_people": [
            {
                "name": match.name,
                "confidence": (
                    round(match.confidence, 3)
                    if match.confidence is not None
                    else None
                ),
            }
            for match in (face_matches or [])
            if match.recognized and match.name
        ],
        "ocr": [
            {"text": item.text, "confidence": item.confidence}
            for item in (ocr_results or [])[:30]
        ],
        "codes": [
            {"format": item.format, "value": item.value}
            for item in (detected_codes or [])[:10]
        ],
    }
    prompt = (
        f"User command: {command.strip()}\n"
        "Verified local detector metadata (supporting context only):\n"
        f"{json.dumps(metadata, ensure_ascii=True, separators=(',', ':'))}"
    )
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    request_options = {
        "model": settings.openai_vision_model,
        "instructions": VISION_ASSISTANT_INSTRUCTIONS,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "auto",
                    },
                ],
            }
        ],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 110,
        "store": False,
    }
    if settings.openai_vision_service_tier.strip():
        request_options["service_tier"] = settings.openai_vision_service_tier.strip()

    try:
        response = await get_openai_vision_client().responses.create(
            **request_options,
        )
    except AuthenticationError as exc:
        raise VisionLLMAuthenticationError(
            "OpenAI rejected the backend credentials."
        ) from exc
    except (APITimeoutError, TimeoutError) as exc:
        raise VisionLLMTimeoutError("OpenAI vision request timed out.") from exc
    except BadRequestError as exc:
        raise VisionLLMRequestError(
            "OpenAI rejected the vision request.",
            provider_status_code=exc.status_code,
            provider_error_type=type(exc).__name__,
        ) from exc
    except (APIConnectionError, RateLimitError, APIStatusError) as exc:
        raise VisionLLMError(
            "OpenAI vision is temporarily unavailable.",
            provider_status_code=getattr(exc, "status_code", None),
            provider_error_type=type(exc).__name__,
        ) from exc

    content = clean_llm_response(response.output_text or "")
    if not content:
        raise VisionLLMError("OpenAI vision returned an empty response.")
    return content

# =========================================================
# GENERAL CHAT
# =========================================================

async def generate_response(
    message: str,
    history: list[Message],
) -> str:

    messages: list[ChatCompletionMessageParam] = [
        cast(
            ChatCompletionMessageParam,
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant for smart glasses. "
                    "Give short, natural spoken responses. "
                    "Usually answer in 1 to 3 sentences. "
                    "Be friendly and conversational. "
                    "Never expose reasoning or internal thoughts."
                ),
            },
        )
    ]

    for item in history:
        messages.append(
            cast(
                ChatCompletionMessageParam,
                {
                    "role": item.role,
                    "content": item.content,
                },
            )
        )

    messages.append(
        cast(
            ChatCompletionMessageParam,
            {
                "role": "user",
                "content": message,
            },
        )
    )

    response = await client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=messages,
        max_completion_tokens=300,
        reasoning_format="hidden",
        reasoning_effort="none",
        temperature=0.7,
    )

    content = clean_llm_response(
        response.choices[0].message.content or ""
    )

    if not content:
        raise RuntimeError(
            "LLM returned an empty response."
        )

    return content.strip()


# =========================================================
# OBJECT DETECTION DESCRIPTION
# =========================================================

async def describe_detections(
    detections: list[Detection],
) -> str:

    if not detections:
        return (
            "I don't see any recognizable objects."
        )

    object_counts: dict[str, int] = {}

    for detection in detections:
        object_counts[detection.class_name] = (
            object_counts.get(
                detection.class_name,
                0,
            )
            + 1
        )

    detected_objects: list[str] = []

    for class_name, count in object_counts.items():

        if count == 1:
            detected_objects.append(
                class_name
            )
        else:
            detected_objects.append(
                f"{count} {class_name}s"
            )

    scene: str = ", ".join(
        detected_objects
    )

    messages: list[ChatCompletionMessageParam] = [
        cast(
            ChatCompletionMessageParam,
            {
                "role": "system",
                "content": (
                    "You are the vision assistant for AI glasses. "
                    "Describe detected objects naturally and briefly. "
                    "The response will be spoken aloud. "
                    "Usually answer in one sentence. "
                    "Do not mention confidence scores, coordinates, "
                    "JSON, models, or internal reasoning."
                ),
            },
        ),
        cast(
            ChatCompletionMessageParam,
            {
                "role": "user",
                "content": (
                    f"The camera detected: {scene}. "
                    "Describe what is visible."
                ),
            },
        ),
    ]

    response = await client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=messages,
        max_completion_tokens=100,
        reasoning_format="hidden",
        reasoning_effort="none",
        temperature=0.3,
    )

    content = clean_llm_response(
        response.choices[0].message.content or ""
    )

    if not content:
        raise RuntimeError(
            "LLM returned an empty vision response."
        )

    return content.strip()


# =========================================================
# SCENE DESCRIPTION
# =========================================================

async def describe_scene(
    objects: list[Detection],
    people: list[PersonDetection],
) -> str:

    known_people: list[str] = []
    unknown_count: int = 0

    # ---------------------------------------------
    # PEOPLE
    # ---------------------------------------------

    for person in people:

        if person.name is not None:
            known_people.append(
                person.name
            )

        else:
            unknown_count += 1

    # Remove duplicate names
    known_people = list(
        dict.fromkeys(
            known_people
        )
    )

    scene_parts: list[str] = []

    # ---------------------------------------------
    # KNOWN PEOPLE
    # ---------------------------------------------

    if known_people:

        if len(known_people) == 1:

            scene_parts.append(
                known_people[0]
            )

        else:

            scene_parts.append(
                ", ".join(
                    known_people[:-1]
                )
                + " and "
                + known_people[-1]
            )

    # ---------------------------------------------
    # UNKNOWN PEOPLE
    # ---------------------------------------------

    if unknown_count == 1:

        scene_parts.append(
            "one other person"
        )

    elif unknown_count > 1:

        scene_parts.append(
            f"{unknown_count} other people"
        )

    # ---------------------------------------------
    # OTHER OBJECTS
    # ---------------------------------------------

    other_objects: dict[str, int] = {}

    for detection in objects:

        if detection.class_name == "person":
            continue

        other_objects[
            detection.class_name
        ] = (
            other_objects.get(
                detection.class_name,
                0,
            )
            + 1
        )

    for class_name, count in (
        other_objects.items()
    ):

        if count == 1:

            scene_parts.append(
                f"a {class_name}"
            )

        else:

            scene_parts.append(
                f"{count} {class_name}s"
            )

    # ---------------------------------------------
    # NOTHING DETECTED
    # ---------------------------------------------

    if not scene_parts:

        return (
            "I don't see anything clearly "
            "recognizable."
        )

    # ---------------------------------------------
    # BUILD SCENE
    # ---------------------------------------------

    scene_description = ", ".join(
        scene_parts
    )

    messages: list[ChatCompletionMessageParam] = [
        cast(
            ChatCompletionMessageParam,
            {
                "role": "system",
                "content": (
                    "You are the vision assistant "
                    "for smart glasses. "
                    "Describe only what is actually "
                    "detected. "
                    "Use natural spoken language. "
                    "Do not invent people, objects, "
                    "locations, distances, or relationships. "
                    "Do not mention models, JSON, "
                    "coordinates, confidence scores, "
                    "or internal reasoning."
                ),
            },
        ),
        cast(
            ChatCompletionMessageParam,
            {
                "role": "user",
                "content": (
                    f"The camera detected: "
                    f"{scene_description}."
                ),
            },
        ),
    ]

    response = await client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=messages,
        max_completion_tokens=100,
        reasoning_format="hidden",
        reasoning_effort="none",
        temperature=0.3,
    )

    content = clean_llm_response(
        response.choices[0].message.content or ""
    )

    if not content:
        raise RuntimeError(
            "LLM returned an empty scene response."
        )

    return content.strip()

async def parse_intent(
    message: str,
) -> IntentResult:

    messages: list[ChatCompletionMessageParam] = [
        cast(
            ChatCompletionMessageParam,
            {
                "role": "system",
                "content": (
                    "You are an intent parser for an AI smart-glasses assistant. "
                    "Classify the user's request into exactly one intent:\n\n"
                    "general = normal conversation or general questions.\n"
                    "vision = asking what the camera sees or what is visible.\n"
                    "person_location = asking where a person is, "
                    "whether they are visible, or where a known person is.\n"
                    "object_search = asking to find a physical object "
                    "such as a phone, keys, bag, or wallet.\n"
                    "memory = asking about something previously remembered "
                    "or stored.\n\n"
                    "Extract the person's name or reference when the user "
                    "is asking about a person.\n"
                    "Examples:\n"
                    "\"Where is Yash?\" -> "
                    "{\"intent\":\"person_location\",\"person\":\"Yash\",\"object_name\":null}\n"
                    "\"Can you find my friend?\" -> "
                    "{\"intent\":\"person_location\",\"person\":\"my friend\",\"object_name\":null}\n"
                    "\"What do you see?\" -> "
                    "{\"intent\":\"vision\",\"person\":null,\"object_name\":null}\n"
                    "\"Where are my keys?\" -> "
                    "{\"intent\":\"object_search\",\"person\":null,\"object_name\":\"keys\"}\n"
                    "\"How are you?\" -> "
                    "{\"intent\":\"general\",\"person\":null,\"object_name\":null}\n\n"
                    "Return ONLY valid JSON. "
                    "Do not explain anything."
                ),
            },
        ),
        cast(
            ChatCompletionMessageParam,
            {
                "role": "user",
                "content": message,
            },
        ),
    ]

    response = await client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=messages,
        response_format={
            "type": "json_object",
        },
        max_completion_tokens=120,
        reasoning_format="hidden",
        reasoning_effort="none",
        temperature=0,
    )

    content = clean_llm_response(
        response.choices[0].message.content or ""
    )

    if not content:
        raise RuntimeError(
            "Intent parser returned an empty response."
        )

    try:
        data = json.loads(content)

        return IntentResult.model_validate(
            data
        )

    except (
        json.JSONDecodeError,
        ValueError,
    ) as exc:

        raise RuntimeError(
            "Intent parser returned invalid data."
        ) from exc
