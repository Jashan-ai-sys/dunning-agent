from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql+asyncpg://recovery:recovery@localhost:5433/recovery"

    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""
    razorpay_api_base: str = "https://api.razorpay.com/v1"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
