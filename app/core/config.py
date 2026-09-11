from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI Glasses Backend"
    environment: str = "development"

    llm_api_key: str = ""
    database_url: str = ""

    class Config:
        env_file = ".env"


settings = Settings()