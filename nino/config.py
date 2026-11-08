"""Application configuration for Nino using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings (bind to 0.0.0.0 to allow access from local and other devices on LAN/Wi-Fi)
    host: str = "0.0.0.0"
    port: int = 8000

    # Authentication
    auth_token: str = "anjana@021"

    # Model settings
    model_adapter: str = "ollama"
    model_name: str = "gemma2:2b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"

    # Embeddings & Vector Search
    embed_model: str = "nomic-embed-text"

    # Memory & Database
    db_path: str = "data/nino.db"
    sensitive_memory: bool = False
    debug_retrieval: bool = False
    token_budget_total: int = 4096
    token_budget_memory: int = 1200
    memory_encryption_key: str | None = None

    # Security & Privacy Gateway
    allow_external_providers: bool = False

    # Tool Execution & Permissions
    max_tool_iterations: int = 5
    tool_timeout_seconds: float = 30.0
    total_tool_timeout_seconds: float = 120.0
    tool_output_max_chars: int = 4000
    allow_restricted_tools: bool = False
    confirmation_expiry_minutes: int = 5
    allowed_dirs: str = ".,test_notes,data"
    enable_web_search: bool = False
    searxng_base_url: str = "http://127.0.0.1:8080"
    enable_code_execution: bool = False
    code_execution_timeout: float = 10.0

    # Location & Weather (GPS) & Timezone
    default_latitude: float | None = 11.5978
    default_longitude: float | None = 75.5814
    default_city: str | None = "Vadakara, Calicut, Kerala, India"
    user_timezone: str = "Asia/Kolkata"

    # Audit Logging
    log_content: bool = False
    audit_log_path: str = "logs/audit.jsonl"

    # Voice Pipeline (Phase 4)
    voice_enabled: bool = True
    wake_word_model: str = "hey_jarvis"
    wake_word_threshold: float = 0.5
    vad_threshold: float = 0.5
    vad_silence_timeout_ms: int = 600
    stt_model: str = "base.en"
    stt_compute_type: str = "int8"
    stt_device: str = "auto"
    tts_adapter: str = "piper"
    tts_voice: str = "en_US-kristin-medium"
    tts_speed: float = 1.0
    store_audio: bool = False
    voice_duplex_mode: str = "half_duplex"

    # Proactivity & Automations (Phase 5)
    proactive_enabled: bool = True
    max_proactive_per_hour: int = 3
    max_proactive_per_day: int = 10
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "08:00"
    daily_briefing_time: str = "08:30"
    weather_enabled: bool = False
    weather_coarse_location: str = ""

    # Home Assistant Integration (Phase 5)
    ha_url: str = "http://127.0.0.1:8123"
    ha_token: str = ""
    ha_allowed_entities: str = ""
    ha_dry_run: bool = False
    ha_enable_restricted: bool = False

    # Security & Auth Lockout (Phase 5)
    auth_max_failed_attempts: int = 5
    auth_lockout_seconds: int = 300

    # Private Push Notifications (ntfy / Tailscale)
    ntfy_url: str | None = None
    ntfy_topic: str | None = None
    ntfy_token: str | None = None

    # Vision & Camera (Phase 6A)
    vision_enabled: bool = True
    vision_model: str = "moondream:latest"
    allow_external_images: bool = False
    ocr_fallback_enabled: bool = False
    temp_image_dir: str = "data/tmp_vision"


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()

