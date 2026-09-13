import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings
from app.schemas.vision import Detection, FaceMatch
from app.services.code_detection import detect_codes
from app.services.face_enrollment import FaceIntroduction
from app.services.frame_quality import (
    evaluate_frame_quality,
    optimize_image_for_vision,
)
from app.services.llm import (
    VisionLLMConfigurationError,
    VisionLLMTimeoutError,
    clean_llm_response,
    describe_image_command,
    get_openai_vision_client,
)


class FrameQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests")
        self.created_paths: list[Path] = []

    def tearDown(self) -> None:
        for path in self.created_paths:
            path.unlink(missing_ok=True)

    def write_image(self, name: str, image: np.ndarray) -> Path:
        path = self.root / f".tmp-vision-{name}"
        encoded, buffer = cv2.imencode(Path(name).suffix, image)
        self.assertTrue(encoded)
        path.write_bytes(buffer.tobytes())
        self.created_paths.append(path)
        return path

    def test_good_and_blurry_frames(self) -> None:
        rng = np.random.default_rng(42)
        clear = rng.integers(25, 231, size=(480, 640, 3), dtype=np.uint8)
        blurry = np.full((480, 640, 3), 127, dtype=np.uint8)

        clear_quality = evaluate_frame_quality(str(self.write_image("clear.jpg", clear)))
        blurry_quality = evaluate_frame_quality(str(self.write_image("blurry.jpg", blurry)))

        self.assertTrue(clear_quality.good)
        self.assertFalse(blurry_quality.good)
        self.assertEqual(blurry_quality.reason, "too_blurry")
        self.assertGreater(clear_quality.quality_score, blurry_quality.quality_score)

    def test_best_of_multiple_frames_and_image_optimization(self) -> None:
        rng = np.random.default_rng(7)
        clear = rng.integers(35, 221, size=(1200, 2400, 3), dtype=np.uint8)
        soft = cv2.GaussianBlur(clear, (31, 31), 0)
        clear_path = self.write_image("clear-large.jpg", clear)
        soft_path = self.write_image("soft-large.jpg", soft)

        qualities = [
            evaluate_frame_quality(str(soft_path)),
            evaluate_frame_quality(str(clear_path)),
        ]
        selected = max(range(len(qualities)), key=lambda index: qualities[index].quality_score)
        self.assertEqual(selected, 1)

        optimized = optimize_image_for_vision(str(clear_path), max_dimension=1000)
        decoded = cv2.imdecode(np.frombuffer(optimized, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(max(decoded.shape[:2]), 1000)


class SanitizerTests(unittest.TestCase):
    def test_removes_hidden_reasoning_without_damaging_answer(self) -> None:
        raw = (
            "<think>private deliberation</think>\n"
            "Analysis: internal planning\n"
            "Final answer: There is a bottle on the table."
        )
        self.assertEqual(
            clean_llm_response(raw),
            "There is a bottle on the table.",
        )

    def test_removes_reasoning_fence_and_preserves_normal_text(self) -> None:
        raw = "```reasoning\nprivate steps\n```\nThe sign says Exit."
        self.assertEqual(clean_llm_response(raw), "The sign says Exit.")


class OpenAIVisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_responses_call_contains_verified_metadata(self) -> None:
        create = AsyncMock(
            return_value=SimpleNamespace(
                output_text="<think>hidden</think>Hussnain is holding a bottle."
            )
        )
        fake_client = SimpleNamespace(responses=SimpleNamespace(create=create))
        detection = Detection(
            class_name="bottle",
            confidence=0.91,
            x1=1,
            y1=2,
            x2=3,
            y2=4,
            relative_position="center",
        )
        face = FaceMatch(
            name="Hussnain",
            recognized=True,
            confidence=0.92,
            detection_confidence=0.99,
        )

        with patch(
            "app.services.llm.get_openai_vision_client",
            return_value=fake_client,
        ):
            answer = await describe_image_command(
                image_bytes=b"jpeg-data",
                command="Who is in front of me?",
                detected_objects=[detection],
                face_matches=[face],
            )

        self.assertEqual(answer, "Hussnain is holding a bottle.")
        create.assert_awaited_once()
        request = create.await_args.kwargs
        self.assertEqual(request["model"], settings.openai_vision_model)
        self.assertFalse(request["store"])
        content = request["input"][0]["content"]
        self.assertEqual(sum(item["type"] == "input_image" for item in content), 1)
        metadata_text = content[0]["text"].split("\n", 2)[-1]
        metadata = json.loads(metadata_text)
        self.assertEqual(metadata["recognized_people"][0]["name"], "Hussnain")

    async def test_missing_key_and_timeout_are_mapped(self) -> None:
        original_key = settings.openai_api_key
        settings.openai_api_key = ""
        get_openai_vision_client.cache_clear()
        try:
            with self.assertRaises(VisionLLMConfigurationError):
                await describe_image_command(b"jpeg", "What is here?")
        finally:
            settings.openai_api_key = original_key
            get_openai_vision_client.cache_clear()

        create = AsyncMock(side_effect=TimeoutError("simulated timeout"))
        fake_client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with patch(
            "app.services.llm.get_openai_vision_client",
            return_value=fake_client,
        ):
            with self.assertRaises(VisionLLMTimeoutError):
                await describe_image_command(b"jpeg", "What is here?")


class CommandEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from main import app

        cls.client = TestClient(app)

    def test_no_usable_frame_does_not_call_openai(self) -> None:
        image = np.full((480, 640, 3), 127, dtype=np.uint8)
        encoded, buffer = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)

        with patch(
            "app.api.vision.describe_image_command",
            new=AsyncMock(),
        ) as describe:
            response = self.client.post(
                "/vision/command",
                data={"session_id": "test-session", "command": "What is here?"},
                files=[("frames", ("frame.jpg", buffer.tobytes(), "image/jpeg"))],
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["success"])
        self.assertFalse(response.json()["processing"]["llm_used"])
        describe.assert_not_awaited()

    def test_three_frames_select_one_and_call_openai_once(self) -> None:
        rng = np.random.default_rng(99)
        images = [
            np.full((480, 640, 3), 127, dtype=np.uint8),
            rng.integers(30, 226, size=(480, 640, 3), dtype=np.uint8),
            cv2.GaussianBlur(
                rng.integers(30, 226, size=(480, 640, 3), dtype=np.uint8),
                (15, 15),
                0,
            ),
        ]
        files = []
        for index, image in enumerate(images):
            encoded, buffer = cv2.imencode(".jpg", image)
            self.assertTrue(encoded)
            files.append(
                ("frames", (f"frame-{index}.jpg", buffer.tobytes(), "image/jpeg"))
            )

        describe = AsyncMock(return_value="There is a bottle ahead of you.")
        object_result = [
            Detection(
                class_name="bottle",
                confidence=0.9,
                x1=1,
                y1=2,
                x2=3,
                y2=4,
            )
        ]
        face_result = [
            SimpleNamespace(
                name="Hussnain",
                similarity=0.91,
                confidence=0.98,
            )
        ]

        with (
            patch(
                "app.api.vision.detect_objects",
                return_value=object_result,
            ) as detect_objects,
            patch("app.api.vision.recognize_faces", return_value=face_result),
            patch("app.api.vision.detect_codes", return_value=[]),
            patch("app.api.vision.describe_image_command", new=describe),
            patch("app.api.vision.shutil.copyfile"),
            patch("app.api.vision.save_latest_frame"),
        ):
            response = self.client.post(
                "/vision/command",
                data={"session_id": "test-multi", "command": "What is ahead?"},
                files=files,
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["processing"]["frame_count"], 3)
        self.assertEqual(body["processing"]["selected_frame"], 1)
        self.assertTrue(body["processing"]["llm_used"])
        self.assertEqual(body["objects"][0]["class_name"], "bottle")
        self.assertEqual(body["face_matches"][0]["name"], "Hussnain")
        self.assertTrue(
            detect_objects.call_args.kwargs["image_path"].endswith(".jpg")
        )
        describe.assert_awaited_once()

    def test_spoken_introduction_enrolls_without_openai_or_yolo(self) -> None:
        image = np.random.default_rng(15).integers(
            30,
            226,
            size=(480, 640, 3),
            dtype=np.uint8,
        )
        encoded, buffer = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)
        introduction = FaceIntroduction(name="Yash", relationship="friend")
        enrollment = SimpleNamespace(
            success=True,
            reply="I've saved Yash as your friend.",
        )

        with (
            patch(
                "app.api.vision.parse_face_introduction",
                return_value=introduction,
            ),
            patch(
                "app.api.vision.enroll_introduced_person",
                return_value=enrollment,
            ),
            patch("app.api.vision.detect_objects") as detect_objects,
            patch("app.api.vision.recognize_faces") as recognize_faces,
            patch("app.api.vision.detect_codes") as detect_codes,
            patch(
                "app.api.vision.describe_image_command",
                new=AsyncMock(),
            ) as describe,
            patch("app.api.vision.shutil.copyfile"),
            patch("app.api.vision.save_latest_frame"),
        ):
            response = self.client.post(
                "/vision/command",
                data={"session_id": "intro-command", "command": "This is my friend Yash."},
                files=[("frames", ("frame.jpg", buffer.tobytes(), "image/jpeg"))],
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["reply"], "I've saved Yash as your friend.")
        self.assertEqual(body["processing"]["mode"], "face_enrollment")
        self.assertFalse(body["processing"]["llm_used"])
        detect_objects.assert_not_called()
        recognize_faces.assert_not_called()
        detect_codes.assert_not_called()
        describe.assert_not_awaited()

    def test_local_qr_detector_returns_exact_value(self) -> None:
        value = "x-glasses-test-123"
        qr_image = cv2.QRCodeEncoder_create().encode(value)
        qr_image = cv2.resize(
            qr_image,
            (600, 600),
            interpolation=cv2.INTER_NEAREST,
        )
        path = Path("tests/.tmp-vision-qr.png")
        encoded, buffer = cv2.imencode(".png", qr_image)
        self.assertTrue(encoded)
        path.write_bytes(buffer.tobytes())
        try:
            codes = detect_codes(str(path))
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(len(codes), 1)
        self.assertEqual(codes[0].value, value)


if __name__ == "__main__":
    unittest.main()
