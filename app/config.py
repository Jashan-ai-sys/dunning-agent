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

    # Payment links expire so a stale link cannot be paid weeks later against
    # a case that has since been closed. Razorpay requires at least 15 minutes.
    payment_link_expiry_hours: int = 48

    worker_interval_seconds: int = 60
    worker_batch_size: int = 50

    # --- Voice ---
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_agent_name: str = "dunning-agent"
    # Created in LiveKit against a SIP provider (Twilio/Plivo/Exotel). Without
    # it there is no way to dial a phone number.
    livekit_sip_trunk_id: str = ""
    # Name the agent introduces itself with.
    company_name: str = "Acme"
    # Sarvam / Gemini / Cartesia keys are read from the environment by their
    # own plugins (SARVAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY).
    # saaras:v4 supersedes saarika:v2.5, which Sarvam is deprecating. Beta
    # access is per-key, so this is only usable with a whitelisted key.
    sarvam_stt_model: str = "saaras:v4"
    # "unknown" lets Sarvam detect the language per utterance. Forcing en-IN
    # mis-transcribes the Hindi half of a Hinglish sentence.
    sarvam_language: str = "unknown"
    # "vertex" | "local". "local" points at any OpenAI-compatible server --
    # SGLang and vLLM both expose one -- so a self-hosted SLM needs no code
    # change, only these three values.
    llm_provider: str = "vertex"
    local_llm_base_url: str = "http://localhost:30000/v1"
    local_llm_model: str = "LiquidAI/LFM2.5-8B-A1B"
    local_llm_api_key: str = "not-needed"

    gemini_model: str = "gemini-2.5-flash"
    # Vertex AI rather than the Gemini Developer API: authenticates with ADC or
    # a service account instead of an API key, and keeps the LLM inside the same
    # GCP project as the rest of the deployment.
    google_use_vertex: bool = True
    google_cloud_project: str = ""
    # Mumbai: verified to serve gemini-2.5-flash, and every extra round trip
    # is audible on a phone call.
    google_cloud_location: str = "asia-south1"
    # "Arushi - Hinglish Speaker". The flow code-switches by default, so a voice
    # trained on Hinglish handles it better than a generic Hindi or English one.
    # Alternatives: Kavita - Customer Care Agent 56e35e2d-6eb6-4226-ab8b-9776515a7094,
    # Devansh - Warm Support Agent 1259b7e3-cb8a-43df-9446-30971a46b8b0.
    cartesia_voice: str = "95d51f79-c397-46f9-b49a-23763d3eaa2d"

    log_level: str = "INFO"

    @property
    def livekit_configured(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
