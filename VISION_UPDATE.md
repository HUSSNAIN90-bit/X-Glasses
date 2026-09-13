# X-Glasses command vision

The phone keeps its live preview local. On a user command it uploads one to three
still frames to `POST /vision/command` (also available at
`POST /api/vision/command`). The legacy
`POST /api/vision/analyze-command-multi` route remains compatible.

The backend rejects invalid or unusable frames, selects the best quality frame,
runs YOLO, InsightFace, and barcode/QR detection, and sends one optimized image
to GPT-6 Astra through the OpenAI Responses API. Detector failures are isolated,
and no OpenAI call is made when every frame is unusable.

## Environment

Create `.env` beside `main.py`:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_VISION_MODEL=gpt-6-astra
```

`LLM_API_KEY` remains optional for the existing Groq-backed general chat and
intent routes. Never put either key in the mobile application or commit `.env`.

## Fast Introductions

When a live-camera request includes an explicit introduction such as `This is my
friend Yash`, `He is my frnd Yash`, or `Save this person as Yash`, the backend
skips LLM intent parsing and scene analysis. It saves the one visible face to the
existing face database and responds immediately. Enrollment is rejected if zero
or multiple faces are visible.

Normal vision requests use a 1280px image limit, JPEG quality 82, and a short
110-token answer. Set `OPENAI_VISION_SERVICE_TIER=fast` only when you want to
use OpenAI's faster paid processing tier.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The multipart request fields are `session_id`, `command`, optional `language`,
and one to three repeated `frames` files. The legacy route also accepts repeated
`images` files.
