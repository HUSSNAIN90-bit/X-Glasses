from fastapi import FastAPI

from app.core.config import settings
from app.api.chat import router as chat_router
from app.api.vision import command_router, router as vision_router
from app.api.faces import router as faces_router
from app.api.enrollment import router as enrollment_router
from app.api.session import router as session_router
from app.api.product_command import router as product_router

from app.services.face_database import (
    initialize_database,
)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0"
)

initialize_database()

app.include_router(chat_router)
app.include_router(vision_router)
app.include_router(command_router)
app.include_router(faces_router)
app.include_router(enrollment_router)
app.include_router(session_router)
app.include_router(product_router)

@app.get("/")
async def root():
    return {
        "message": "AI Glasses Backend is running"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "environment": settings.environment
    }
