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

    # --- Recovery policy ---
    # Below this, a voice call costs more than the debt is worth.
    min_recoverable_amount_paise: int = 5_000  # Rs 50
    # Minimum gap between contact attempts on the same case.
    retry_backoff_hours: int = 24
    # Contact window in the customer's local time. TRAI restricts commercial
    # calls to 09:00-21:00; staying inside it is the compliance story.
    contact_window_start_hour: int = 9
    contact_window_end_hour: int = 21
    contact_timezone: str = "Asia/Kolkata"

    worker_interval_seconds: int = 60
    worker_batch_size: int = 50

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
