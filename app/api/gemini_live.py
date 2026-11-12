from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, HTTPException

from app.core.config import settings

router = APIRouter(prefix="/api/gemini", tags=["Gemini Live"])

TOKEN_URL = "https://generativelanguage.googleapis.com/v1beta/auth_tokens"


@router.post("/live-token")
async def create_live_token():
    """Issue a short-lived, single-use token; the Gemini API key never reaches mobile."""
    api_key = settings.gemini_api_key.strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Gemini Live is not configured on the backend.")

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=2)
    payload = {
        "uses": 1,
        "expireTime": expires_at.isoformat().replace("+00:00", "Z"),
        "newSessionExpireTime": expires_at.isoformat().replace("+00:00", "Z"),
        "liveConnectConstraints": {
            "model": f"models/{settings.gemini_live_model}",
            "config": {"responseModalities": ["AUDIO"], "outputAudioTranscription": {}, "sessionResumption": {}},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(TOKEN_URL, headers={"x-goog-api-key": api_key}, json=payload)
        response.raise_for_status()
        data = response.json()
        token = (data.get("name") or data.get("authToken", {}).get("name") or "").strip()
        if not token:
            raise ValueError("Gemini did not return an ephemeral token.")
        return {"token": token, "model": settings.gemini_live_model, "expires_at": data.get("expireTime")}
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail="Gemini Live token service rejected the request.") from exc
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Unable to initialize Gemini Live right now.") from exc
