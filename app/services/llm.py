from typing import cast
import json

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam

from app.core.config import settings
from app.schemas.vision import (
    Detection,
    PersonDetection,
)
from app.services.memory import Message


client = AsyncGroq(
    api_key=settings.llm_api_key,
)
from app.schemas.chat import IntentResult

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

    content: str | None = (
        response.choices[0].message.content
    )

    if content is None:
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

    content: str | None = (
        response.choices[0].message.content
    )

    if content is None:
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

    content: str | None = (
        response.choices[0]
        .message
        .content
    )

    if content is None:
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

    content = response.choices[0].message.content

    if content is None:
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