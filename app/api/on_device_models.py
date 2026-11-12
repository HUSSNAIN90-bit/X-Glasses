import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/models", tags=["On-device models"])
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "yolo26n.tflite"
MANIFEST_PATH = ROOT / "models" / "yolo26n_manifest.json"


@router.get("/yolo26n/manifest")
async def yolo26n_manifest():
    if not MODEL_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise HTTPException(status_code=503, detail="On-device YOLO26 LiteRT model has not been exported on this backend.")
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="YOLO26 model manifest is unavailable.") from exc


@router.get("/yolo26n.tflite")
async def download_yolo26n_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="On-device YOLO26 LiteRT model has not been exported on this backend.")
    return FileResponse(
        MODEL_PATH,
        media_type="application/octet-stream",
        filename="yolo26n.tflite",
        headers={"Cache-Control": "public, max-age=86400"},
    )
