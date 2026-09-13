import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from app.services.face_enrollment import (
    FaceIntroduction,
    enroll_introduced_person,
    parse_face_introduction,
)


class FaceIntroductionParserTests(unittest.TestCase):
    def test_parses_spoken_friend_introductions(self) -> None:
        introduction = parse_face_introduction("This is my friend Yash.")
        self.assertEqual(
            introduction,
            FaceIntroduction(name="Yash", relationship="friend"),
        )

    def test_parses_common_speech_to_text_spelling(self) -> None:
        introduction = parse_face_introduction("He is my frnd yash")
        self.assertEqual(
            introduction,
            FaceIntroduction(name="Yash", relationship="friend"),
        )

    def test_does_not_treat_normal_vision_request_as_enrollment(self) -> None:
        self.assertIsNone(parse_face_introduction("Who is this person?"))


class FaceEnrollmentServiceTests(unittest.TestCase):
    def test_creates_person_and_stores_one_embedding(self) -> None:
        embedding = np.ones(512, dtype=np.float32)
        introduction = FaceIntroduction(name="Yash", relationship="friend")

        with (
            patch(
                "app.services.face_enrollment.extract_face_embeddings",
                return_value=[embedding],
            ),
            patch(
                "app.services.face_enrollment.get_person_by_name",
                return_value=None,
            ),
            patch(
                "app.services.face_enrollment.create_person",
                return_value="person-1",
            ) as create_person,
            patch(
                "app.services.face_enrollment.add_embedding_to_person",
            ) as add_embedding,
        ):
            result = enroll_introduced_person("frame.jpg", introduction)

        self.assertTrue(result.success)
        self.assertEqual(result.reply, "I've saved Yash as your friend.")
        create_person.assert_called_once_with(name="Yash", relationship="friend")
        add_embedding.assert_called_once_with(person_id="person-1", embedding=embedding)

    def test_rejects_multiple_faces_without_writing(self) -> None:
        introduction = FaceIntroduction(name="Yash", relationship="friend")
        with (
            patch(
                "app.services.face_enrollment.extract_face_embeddings",
                return_value=[np.ones(2), np.ones(2)],
            ),
            patch("app.services.face_enrollment.create_person") as create_person,
        ):
            result = enroll_introduced_person("frame.jpg", introduction)

        self.assertFalse(result.success)
        self.assertIn("only one face", result.reply)
        create_person.assert_not_called()


class OrchestratorEnrollmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_intro_bypasses_llm_intent_parser(self) -> None:
        from app.services.orchestrator import process_request

        introduction = FaceIntroduction(name="Yash", relationship="friend")
        enrollment = SimpleNamespace(reply="I've saved Yash as your friend.")
        with (
            patch(
                "app.services.orchestrator.parse_face_introduction",
                return_value=introduction,
            ),
            patch(
                "app.services.orchestrator.get_latest_frame",
                return_value="frame.jpg",
            ),
            patch(
                "app.services.orchestrator.enroll_introduced_person",
                return_value=enrollment,
            ),
            patch("app.services.orchestrator.add_message"),
            patch(
                "app.services.orchestrator.parse_intent",
                new=AsyncMock(),
            ) as parse_intent,
        ):
            reply, intent = await process_request(
                session_id="intro-session",
                message="This is my friend Yash.",
            )

        self.assertEqual(reply, "I've saved Yash as your friend.")
        self.assertEqual(intent, "face_enrollment")
        parse_intent.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
