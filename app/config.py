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

    # Optional shared cache. Empty means no Redis, which is the default and
    # a supported configuration -- every cache call degrades to a miss and
    # the service behaves exactly as it did before Redis existed.
    #
    # Redis is never the authority here. Postgres decides; this only saves
    # the round trip when the answer is already known, and coordinates the
    # contact cooldown across workers, which a row lock cannot do.
    redis_url: str = ""
    # How long a webhook event id is remembered. Longer than any
    # plausible redelivery window; the unique constraint covers forever.
    redis_event_ttl_seconds: int = 86_400

    # Push delivery for webhook events. Empty means cron-only, which is the
    # default and how this service has always worked -- the sweep finds
    # every envelope either way, just up to five minutes later.
    # Full resource name: projects/PROJECT/topics/TOPIC
    pubsub_topic: str = ""
    pubsub_publish_timeout_seconds: float = 5.0
    # Service account Pub/Sub signs its push requests as. Empty disables
    # the push endpoint outright: an unverified endpoint that dispatches
    # handlers is not something to leave open by default.
    pubsub_push_service_account: str = ""
    # The audience the OIDC token must carry -- normally this service URL.
    pubsub_push_audience: str = ""

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

    # How long after a failed checkout payment we wait before chasing it. A
    # customer who simply retries with another card two minutes later has not
    # abandoned anything, and contacting them mid-purchase would be absurd.
    checkout_grace_minutes: int = 30

    # How many times a webhook envelope is retried before it is dead-lettered.
    # A retry that can never succeed is not resilience, it is a queue jam.
    webhook_max_attempts: int = 5
    # How many consecutive delivery failures a case tolerates before it is
    # closed as undeliverable. Distinct from max_attempts, which counts
    # contacts the customer actually received.
    max_delivery_failures: int = 5

    # Re-charge a still-valid mandate instead of sending a link, when the only
    # thing that went wrong was that the money was not there. Off by default:
    # it is the one path that takes money without the customer doing anything,
    # so a deployment should turn it on deliberately rather than inherit it.
    mandate_retry_enabled: bool = False

    # Minimum gap between contacting the same *person*, whatever the debt.
    # Distinct from retry_backoff_hours, which bounds one case: a customer with
    # three failed charges should still hear from us once, not three times.
    customer_contact_cooldown_hours: int = 24

    # How long we defer to Razorpay's own retry sequence before working a
    # bank-side failure ourselves. Deliberately a clock and not a webhook: the
    # `subscription.halted` event is the *ideal* signal, but a case that never
    # receives one -- cancelled instead of halted, event not subscribed,
    # delivery lost past the replay window -- must still eventually be worked
    # rather than waiting forever.
    bank_retry_grace_hours: int = 72

    # Appended to every payment link reference_id. Razorpay enforces
    # reference_id uniqueness across the whole account, forever -- and case ids
    # restart at 1 on a fresh database, so a redeploy against a new Cloud SQL
    # instance collides with links the account has held since the last one.
    # That is a real 400 we hit: "payment link with given reference_id:
    # recovery-1-0 already exists".
    #
    # Empty by default, which reproduces the old format exactly so links
    # already out in the world keep reconciling. Set it to anything short and
    # distinct per deployment -- a date, an environment name, a random suffix.
    payment_reference_namespace: str = ""

    # Payment links expire so a stale link cannot be paid weeks later against
    # a case that has since been closed. Razorpay requires at least 15 minutes.
    payment_link_expiry_hours: int = 48
    # How long a customer who says "yes, I will pay" has before that promise
    # counts as broken. Tracks the link's own lifetime: a promise cannot
    # outlive the instrument it was made against.
    promise_window_hours: int = 48

    worker_interval_seconds: int = 60
    worker_batch_size: int = 50

    # --- Sarvam server-side VAD (saaras:v3 only) ---
    #
    # Distinct from the Silero settings below. Silero decides whether we are
    # listening; these decide whether Sarvam turns what it hears into words.
    # Both matter, because a barge-in needs BOTH: Silero must open the turn and
    # the audio must transcribe into enough words to pass min_words.
    #
    # Every one is None by default, meaning Sarvam's own default applies. They
    # are exposed as settings rather than constants because the right values
    # depend on the line -- a call from a shop is not a call from an office --
    # and finding them means changing a number and listening, not redeploying.
    #
    # high_vad_sensitivity is the blunt version of all of these. Keeping it on
    # while raising the floors below is deliberate: hear the quiet customer,
    # ignore the quiet background.
    #
    # Volume (dB) below which audio is too quiet to be speech. The most direct
    # answer to a television in the next room.
    sarvam_start_speech_volume_threshold: float | None = None
    # Consecutive speech frames needed to open a segment. Raise it and a short
    # burst -- a cough, a door -- never becomes a word.
    sarvam_min_speech_frames: int | None = None
    # The same, for registering a barge-in specifically.
    sarvam_interrupt_min_speech_frames: int | None = None
    # VAD probability (0-1) above which a frame counts as speech. Raise to make
    # Sarvam more certain before it commits.
    sarvam_positive_speech_threshold: float | None = None
    sarvam_negative_speech_threshold: float | None = None
    # Frames prepended before detected onset -- recovers a clipped first
    # syllable, which is what a raised threshold tends to cost.
    sarvam_pre_speech_pad_frames: int | None = None

    # --- Turn detection (VAD) ---
    # Values from the Blostem production telephony backend, whose comments carry
    # the reasoning. They tuned against 8 kHz mu-law with background noise; a
    # clean browser stream may tolerate a lower threshold, so these are knobs.
    #
    #   confidence 0.7  -> activation_threshold
    #   stop_secs  0.3  -> min_silence_duration. They lowered it from 0.6 on
    #                      2026-04-17 "to cut ~300ms of perceived bot thinking
    #                      delay". LiveKit's own default is 0.55, i.e. slower
    #                      than the value they moved away from.
    #   start_secs 0.4  -> min_speech_duration. Deliberately high to suppress
    #                      false barge-ins; they reverted an 0.2 experiment.
    #                      LiveKit's default is 0.05, so this is the one to
    #                      raise carefully rather than copy outright.
    #
    # WARNING carried over with the number: their own smart_turn_shadow notes
    # that Silero at confidence=0.7 "never fired on the customer channel" across
    # three recordings while firing readily on the bot's TTS in the same files.
    # Narrowband carrier audio measures 0.05-0.86% of its energy above 4 kHz, so
    # the upper half a 16 kHz VAD expects is simply empty. 0.7 is right for the
    # wideband browser path we run today; when the SIP leg lands, expect to drop
    # this and verify the VAD actually fires on the customer, not just on us.
    vad_activation_threshold: float = 0.7
    vad_min_silence_duration: float = 0.3
    vad_min_speech_duration: float = 0.05

    # Speak a rendered opening line instead of asking the LLM for one.
    # The greeting is the only turn with no input to reason about, so the
    # round trip buys nothing and costs the customer half a second at the
    # moment they are listening hardest. Off makes the model generate it.
    cached_greeting_enabled: bool = True

    # --- Voice ---
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_agent_name: str = "dunning-agent"
    # Created in LiveKit against a SIP provider (Twilio/Plivo/Exotel). Without
    # it there is no way to dial a phone number.
    livekit_sip_trunk_id: str = ""

    # --- Twilio (outbound telephony) ---
    # The reachable alternative to a LiveKit SIP trunk: a trial account issues a
    # number immediately, and Media Streams carry the audio over a websocket. The
    # number need not be Indian -- a US trial number dialling a verified mobile is
    # enough to exercise the loop end to end.
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    # E.164, the Twilio number that appears as the caller.
    twilio_from_number: str = ""
    # Public wss:// address of the agent's telephony websocket -- an ngrok tunnel
    # in development, the deployed service in production. Twilio dials out from
    # its own network, so localhost is never reachable.
    twilio_stream_url: str = ""
    twilio_api_base: str = "https://api.twilio.com/2010-04-01"
    # Answering machine detection. Without it a voicemail greeting answers,
    # the agent talks to a recording, and the attempt is spent -- three of
    # those and the case closes as max_attempts_reached having never reached
    # a person.
    #
    # Asynchronous on purpose: synchronous AMD holds the call while Twilio
    # listens, which is several seconds of silence for a human who did answer.
    # Async connects immediately and tells us afterwards.
    machine_detection_enabled: bool = True
    # Seconds Twilio may spend deciding. Its own range is 3-59.
    machine_detection_timeout: int = 30
    # Public URL Twilio posts the verdict to. Empty disables AMD regardless of
    # the flag above -- detection nobody can hear the result of is pointless.
    twilio_amd_callback_url: str = ""
    # Name the agent introduces itself with.
    company_name: str = "Acme"
    # Sarvam / Gemini / Cartesia keys are read from the environment by their
    # own plugins (SARVAM_API_KEY, GOOGLE_API_KEY, CARTESIA_API_KEY).
    # saaras:v4 supersedes saarika:v2.5, which Sarvam is deprecating. Beta
    # access is per-key, so this is only usable with a whitelisted key.
    sarvam_stt_model: str = "saaras:v4"
    # Sarvam needs BCP-47, not a bare two-letter code: "hi" is rejected with a
    # validation error naming the legal set, "hi-IN" is accepted.
    #
    # "hi-IN" rather than "unknown" (auto-detect) is deliberate. Auto-detect
    # decides per utterance, and on a short Hinglish reply it can land on
    # English and hand back romanised text -- which the model then mirrors,
    # producing exactly the "haan ji" output we do not want. Pinning Hindi
    # makes Sarvam decode toward Devanagari at the source. Set "unknown" if a
    # campaign genuinely needs language detection.
    sarvam_language: str = "hi-IN"
    # "vertex" | "local". "local" points at any OpenAI-compatible server --
    # SGLang and vLLM both expose one -- so a self-hosted SLM needs no code
    # change, only these three values.
    llm_provider: str = "vertex"
    local_llm_base_url: str = "http://localhost:30000/v1"
    local_llm_model: str = "LiquidAI/LFM2.5-8B-A1B"
    local_llm_api_key: str = "not-needed"

    # NeMo Guardrails output rails: "off" | "audit" | "block".
    #
    # Defaults to off so installing the dependency changes nothing about a live
    # call until somebody decides it should. "audit" streams normally and logs
    # what it would have stopped, costing no latency; "block" holds each turn
    # until the rail clears it, which forfeits progressive TTS and with it the
    # 0.058s time to first byte. See `app/voice/guardrails/processor.py`.
    guardrails_mode: str = "off"

    # Blostem's production debt-recovery template runs 0.2 / 120. A dunning
    # call wants consistency over flair, and a cap keeps turns to the one or
    # two sentences a phone call can carry.
    llm_temperature: float = 0.2
    llm_max_tokens: int = 120

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
    #: Base URL of the self-hosted STT/TTS service (scripts/modal_speech.py).
    #: Empty means the vendor path: Sarvam for hearing, Cartesia for speaking.
    #: Set it and both swap together, because they are one deployment -- a
    #: call that heard locally and spoke through a vendor would tell us
    #: nothing useful about either.
    local_speech_url: str = ""
    cartesia_voice: str = "95d51f79-c397-46f9-b49a-23763d3eaa2d"

    log_level: str = "INFO"

    @property
    def livekit_configured(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

    @property
    def twilio_sms_configured(self) -> bool:
        """Enough to send a text, which is less than enough to place a call.

        Placing a call needs somewhere for Twilio to stream the audio to. An
        SMS needs a sender and a credential. Conflating the two meant the voice
        service -- which receives calls and therefore has no stream URL -- could
        not send the mandate link, and failed at construction before it reached
        the API.
        """
        return bool(
            self.twilio_account_sid and self.twilio_auth_token and self.twilio_from_number
        )

    @property
    def twilio_configured(self) -> bool:
        return bool(
            self.twilio_account_sid
            and self.twilio_auth_token
            and self.twilio_from_number
            and self.twilio_stream_url
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
