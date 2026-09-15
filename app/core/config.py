from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Glasses Backend"
    environment: str = "development"

    llm_api_key: str = ""
    groq_chat_model: str = "llama-3.1-8b-instant"

    openai_api_key: str = ""
    openai_vision_model: str = "gpt-6-astra"
    openai_timeout_seconds: float = 30.0
    openai_vision_service_tier: str = ""

    max_vision_frame_bytes: int = 8 * 1024 * 1024
    max_vision_frame_pixels: int = 24_000_000
    vision_max_dimension: int = 1280
    vision_jpeg_quality: int = 82

    database_url: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @field_validator("openai_vision_model", mode="before")
    @classmethod
    def normalize_openai_vision_model(cls, value: object) -> str:
        return str(value).strip().lower()

    @field_validator("groq_chat_model", mode="before")
    @classmethod
    def normalize_groq_chat_model(cls, value: object) -> str:
        return str(value).strip()


settings = Settings()
