import asyncio
import time
import uuid

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.schemas.vision import CodeDetection, CommandVisionResponse, FrameQuality, VisionProcessing
from app.services.code_detection import detect_codes_from_frames
from app.services.frame_quality import evaluate_frame_quality, optimize_image_for_vision
from app.services.llm import describe_image_command

router = APIRouter(prefix="/api/product", tags=["Product"])
RETRY_SESSIONS: set[str] = set()

IMAGE_SUFFIXES = {
    "image/bmp": ".bmp",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


def _reply_for_product_codes(codes) -> str | None:
    product_codes = [c for c in codes if c.format.upper() not in {"QR_CODE", "QR", "QRCODE"}]
    for code in product_codes:
        product = code.product
        if not product:
            continue
        price_text = None
        if product.prices:
            first = product.prices[0]
            currency = f" {first.currency}" if first.currency else ""
            price_text = f" Price is {first.value:.2f}{currency}."
        if product.name:
            return f"{product.name}{(' by ' + product.brand) if product.brand else ''}.{price_text or ''}".strip()

    qr_codes = [c for c in codes if c.format.upper() in {"QR_CODE", "QR", "QRCODE"}]
    if qr_codes:
        value = qr_codes[0].value
        return "I found a QR code with a web link." if value.startswith(("http://", "https://")) else "I found a QR code."
    return None


async def _llm_fallback(selected_path: str, command: str):
    optimized = await asyncio.to_thread(
        optimize_image_for_vision,
        selected_path,
        settings.vision_max_dimension,
        settings.vision_jpeg_quality,
    )
    return await describe_image_command(image_bytes=optimized, command=command, detected_codes=[])


@router.post("/command", response_model=CommandVisionResponse)
async def product_command(
    session_id: str = Form(...),
    command: str = Form(...),
    language: str = Form("en"),
    barcode_retry: bool = Form(False),
    frames: list[UploadFile] = File(...),
) -> CommandVisionResponse:
    started = time.perf_counter()
    session_id = session_id.strip()
    command = command.strip()
    language = language.strip() or "en"
    if not session_id or not command:
        raise HTTPException(status_code=400, detail="session_id and command are required.")
    if not frames or len(frames) > 6:
        raise HTTPException(status_code=400, detail="Send between 1 and 6 frames.")

    barcode_retry = barcode_retry or session_id in RETRY_SESSIONS
    RETRY_SESSIONS.discard(session_id)

    saved: list[tuple[int, str]] = []
    quality: list[FrameQuality] = []
    try:
        for index, upload in enumerate(frames):
            if upload.content_type and not upload.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Every frame must be an image.")
            data = await upload.read(settings.max_vision_frame_bytes + 1)
            await upload.close()
            if not data or len(data) > settings.max_vision_frame_bytes:
                raise HTTPException(status_code=413, detail="Invalid or oversized image frame.")
            suffix = IMAGE_SUFFIXES.get(upload.content_type or ".jpg", ".jpg")
            with NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(data)
                saved.append((index, temp.name))

        metrics = await asyncio.gather(*[
            asyncio.to_thread(evaluate_frame_quality, path, 55.0, 30.0, 225.0)
            for _, path in saved
        ])
        for (index, _), item in zip(saved, metrics):
            quality.append(FrameQuality(index=index, good=item.good, blur_score=item.blur_score, brightness=item.brightness, width=item.width, height=item.height, exposure_score=item.exposure_score, quality_score=item.quality_score, reason=item.reason))

        candidates = [(idx, path, q) for (idx, path), q in zip(saved, quality) if q.good] or [(idx, path, q) for (idx, path), q in zip(saved, quality)]
        selected_idx, selected_path, selected_quality = max(candidates, key=lambda item: item[2].quality_score)
        code_paths = [path for _, path, q in candidates if q.good] or [p for _, p in saved]
        codes = await asyncio.to_thread(detect_codes_from_frames, code_paths)

        processing = VisionProcessing(
            mode="barcode_scan",
            request_id=str(uuid.uuid4()),
            frame_count=len(saved),
            selected_frame=selected_idx,
            quality_score=selected_quality.quality_score,
            ai_called=False,
            detectors={"barcode_qr": "ok" if codes else "not_found"},
            durations_ms={"barcode_qr": round((time.perf_counter() - started) * 1000, 1)},
            frame_quality=quality,
        )

        direct_reply = _reply_for_product_codes(codes)
        if direct_reply:
            return CommandVisionResponse(success=True, session_id=session_id, command=command, reply=direct_reply, language=language, barcodes=[CodeDetection(format=c.format, value=c.value, product=None) for c in codes], processing=processing)

        # A decoded barcode/QR with no product match should go straight to the
        # visual model. Only a completely unreadable barcode gets the closer prompt.
        if codes or barcode_retry:
            llm_started = time.perf_counter()
            reply = await _llm_fallback(selected_path, command)
            processing.mode = "barcode_llm_fallback"
            processing.llm_attempted = True
            processing.llm_used = True
            processing.ai_called = True
            processing.durations_ms["llm"] = round((time.perf_counter() - llm_started) * 1000, 1)
            processing.durations_ms["total"] = round((time.perf_counter() - started) * 1000, 1)
            return CommandVisionResponse(success=True, session_id=session_id, command=command, reply=reply, language=language, barcodes=[CodeDetection(format=c.format, value=c.value, product=None) for c in codes], processing=processing)

        RETRY_SESSIONS.add(session_id)
        processing.mode = "barcode_retry"
        processing.detectors["barcode_qr"] = "retry_requested"
        return CommandVisionResponse(success=True, session_id=session_id, command=command, reply="I can't read the barcode clearly. Please bring the product a little closer.", language=language, barcodes=[], processing=processing)
    finally:
        for _, path in saved:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
