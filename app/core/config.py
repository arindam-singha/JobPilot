from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment and .env files."""

    app_env: str = Field(default="development", alias="APP_ENV")
    database_url: str = Field(default="postgresql+asyncpg://jobpilot:jobpilot@postgres:5432/jobpilot", alias="DATABASE_URL")
    cv_storage_dir: str = Field(default="storage/cvs", alias="CV_STORAGE_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
