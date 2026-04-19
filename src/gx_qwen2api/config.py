"""Configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    port: int = 31998
    address: str = "0.0.0.0"

    # Auth
    api_key: str = ""
    qwen_code_auth_use: bool = True

    # Model
    default_model: str = "coder-model"

    # Retry
    max_retries: int = 5
    retry_delay_ms: int = 1000
    quota_cooldown_seconds: int = 14400 # 4 hours by default

    # Qwen API (hardcoded)
    qwen_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_oauth_token_url: str = "https://chat.qwen.ai/api/v1/oauth2/token"
    qwen_oauth_client_id: str = "f0304373b74a44d2b584a3fb70ca9e56"
    token_refresh_buffer_s: int = 30

    # Freebuff / Codebuff
    freebuff_api_base: str = "https://codebuff.com"
    freebuff_rotation_interval_seconds: int = 21600
    freebuff_model_source_url: str = (
        "https://raw.githubusercontent.com/CodebuffAI/codebuff/main/common/src/constants/free-agents.ts"
    )

    # Credentials directory — scanned for *.json files
    creds_dir: Path = Path.home() / ".qwen"

    # Logging
    log_level: str = "info"
    log_requests: bool = True

    # Web admin — set to empty string to disable
    admin_enabled: bool = True
    # Simple admin auth password (set in production!)
    admin_password: str = ""

    # Auto-refresh (background keep-alive)
    auto_refresh_enabled: bool = True
    auto_refresh_interval_seconds: int = 300  # 5 minutes
    auto_refresh_threshold_minutes: int = 30  # refresh if < 30 min left

    @property
    def api_keys(self) -> list[str] | None:
        if not self.api_key:
            return None
        return [k.strip() for k in self.api_key.split(",") if k.strip()]

    @property
    def retry_delay_s(self) -> float:
        return self.retry_delay_ms / 1000


settings = Settings()
