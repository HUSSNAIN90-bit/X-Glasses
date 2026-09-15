import base64
import json
import re
from functools import lru_cache
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
from app.schemas.chat import IntentResult
from app.schemas.vision import (
    CodeDetection,
    Detection,
    FaceMatch,
    OCRResult,
    PersonDetection,
)
from app.services.memory import Message


groq_client = AsyncGroq(api_key=settings.llm_api_key)


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
Return only the final answer intended for the user. Keep normal answers to 1-2
short sentences unless the user explicitly requests more detail. Use simple
natural language suitable for speech. If the requested information cannot be
determined from the image or verified metadata, say so briefly."""


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
    if not text:
        return ""
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:final answer|answer)\s*:\s*", "", text, flags=re.MULTILINE | re.IGNORECASE)
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
                "confidence": round(match.confidence, 3) if match.confidence is not None else None,
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
        response = await get_openai_vision_client().responses.create(**request_options)
    except AuthenticationError as exc:
        raise VisionLLMAuthenticationError("OpenAI rejected the backend credentials.") from exc
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


async def _groq_chat(
    messages: list[ChatCompletionMessageParam],
    *,
    max_completion_tokens: int,
    temperature: float,
    json_mode: bool = False,
) -> str:
    kwargs = {
        "model": settings.groq_chat_model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await groq_client.chat.completions.create(**kwargs)
    return clean_llm_response(response.choices[0].message.content or "")


async def generate_response(message: str, history: list[Message]) -> str:
    messages: list[ChatCompletionMessageParam] = [
        cast(ChatCompletionMessageParam, {
            "role": "system",
            "content": (
                "You are a helpful AI assistant for Smart Eyes. "
                "Give short, natural spoken responses. Usually answer in 1 to 3 sentences. "
                "Be friendly and conversational. Never expose reasoning or internal thoughts."
            ),
        })
    ]
    for item in history:
        messages.append(cast(ChatCompletionMessageParam, {"role": item.role, "content": item.content}))
    messages.append(cast(ChatCompletionMessageParam, {"role": "user", "content": message}))

    content = await _groq_chat(messages, max_completion_tokens=300, temperature=0.7)
    if not content:
        raise RuntimeError("LLM returned an empty response.")
    return content


async def describe_detections(detections: list[Detection]) -> str:
    if not detections:
        return "I don't see any recognizable objects."

    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
    scene = ", ".join(
        name if count == 1 else f"{count} {name}s"
        for name, count in counts.items()
    )

    messages = [
        cast(ChatCompletionMessageParam, {
            "role": "system",
            "content": (
                "You are the vision assistant for Smart Eyes. Describe detected objects naturally and briefly. "
                "The response will be spoken aloud. Usually answer in one sentence. "
                "Do not mention confidence scores, coordinates, JSON, models, or internal reasoning."
            ),
        }),
        cast(ChatCompletionMessageParam, {
            "role": "user",
            "content": f"The camera detected: {scene}. Describe what is visible.",
        }),
    ]
    content = await _groq_chat(messages, max_completion_tokens=100, temperature=0.3)
    return content or "I don't see anything clearly recognizable."


async def describe_scene(objects: list[Detection], people: list[PersonDetection]) -> str:
    known_people = list(dict.fromkeys(p.name for p in people if p.name))
    unknown_count = sum(1 for p in people if not p.name)
    scene_parts: list[str] = []

    if known_people:
        scene_parts.append(
            known_people[0] if len(known_people) == 1
            else ", ".join(known_people[:-1]) + " and " + known_people[-1]
        )
    if unknown_count == 1:
        scene_parts.append("one other person")
    elif unknown_count > 1:
        scene_parts.append(f"{unknown_count} other people")

    other_objects: dict[str, int] = {}
    for detection in objects:
        if detection.class_name != "person":
            other_objects[detection.class_name] = other_objects.get(detection.class_name, 0) + 1
    for name, count in other_objects.items():
        scene_parts.append(f"a {name}" if count == 1 else f"{count} {name}s")

    if not scene_parts:
        return "I don't see anything clearly recognizable."

    description = ", ".join(scene_parts)
    messages = [
        cast(ChatCompletionMessageParam, {
            "role": "system",
            "content": (
                "You are the vision assistant for Smart Eyes. Describe only what is actually detected. "
                "Use natural spoken language. Do not invent people, objects, locations, distances, or relationships. "
                "Do not mention models, JSON, coordinates, confidence scores, or internal reasoning."
            ),
        }),
        cast(ChatCompletionMessageParam, {
            "role": "user",
            "content": f"The camera detected: {description}.",
        }),
    ]
    content = await _groq_chat(messages, max_completion_tokens=100, temperature=0.3)
    return content or "I don't see anything clearly recognizable."


async def parse_intent(message: str) -> IntentResult:
    messages = [
        cast(ChatCompletionMessageParam, {
            "role": "system",
            "content": (
                "You are an intent parser for an AI smart-glasses assistant. "
                "Classify the user's request into exactly one intent: "
                "general, vision, person_location, object_search, or memory. "
                "Extract person or object_name when relevant. Return ONLY valid JSON with keys "
                "intent, person, object_name. "
                "Examples: "
                "Where is Yash? -> {\"intent\":\"person_location\",\"person\":\"Yash\",\"object_name\":null}; "
                "What do you see? -> {\"intent\":\"vision\",\"person\":null,\"object_name\":null}; "
                "Where are my keys? -> {\"intent\":\"object_search\",\"person\":null,\"object_name\":\"keys\"}; "
                "How are you? -> {\"intent\":\"general\",\"person\":null,\"object_name\":null}."
            ),
        }),
        cast(ChatCompletionMessageParam, {"role": "user", "content": message}),
    ]
    content = await _groq_chat(
        messages,
        max_completion_tokens=120,
        temperature=0,
        json_mode=True,
    )
    if not content:
        raise RuntimeError("Intent parser returned an empty response.")
    try:
        return IntentResult.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("Intent parser returned invalid data.") from exc
