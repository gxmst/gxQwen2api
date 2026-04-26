"""DeepSeek internal models and OpenAI mapping.

Aligned with ds-free-api:
- Internal model types: default, expert
- Exposed OpenAI model ids: deepseek-flash, deepseek-pro
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# DeepSeek internal model types -> exposed OpenAI model ids
DS_MODEL_MAPPING: dict[str, str] = {
    "default": "deepseek-flash",
    "expert": "deepseek-pro",
}

# Reverse mapping: OpenAI model id -> DeepSeek internal type
OPENAI_TO_DS_MODEL: dict[str, str] = {
    v: k for k, v in DS_MODEL_MAPPING.items()
}

# All exposed models for /v1/models
EXPOSED_MODELS: list[str] = list(DS_MODEL_MAPPING.values())


def ds_model_type(openai_model: str) -> str | None:
    """Map OpenAI model name to DeepSeek internal model type."""
    return OPENAI_TO_DS_MODEL.get(openai_model)


def openai_model_id(ds_type: str) -> str | None:
    """Map DeepSeek internal model type to OpenAI model id."""
    return DS_MODEL_MAPPING.get(ds_type)


def is_supported_model(model: str) -> bool:
    return model in OPENAI_TO_DS_MODEL


@dataclass
class DeepseekAccount:
    """DeepSeek account credentials and state (persisted to JSON)."""

    account_id: str
    email: str = ""
    password: str = ""
    mobile: str = ""
    area_code: str = ""
    enabled: bool = True

    # Session tokens
    access_token: str = ""
    refresh_token: str = ""

    # Metadata
    created_at: float = field(default_factory=lambda: __import__("time").time())
    last_login_at: float | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "email": self.email,
            "password": self.password,
            "mobile": self.mobile,
            "area_code": self.area_code,
            "enabled": self.enabled,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeepseekAccount:
        return cls(
            account_id=data.get("account_id", ""),
            email=data.get("email", ""),
            password=data.get("password", ""),
            mobile=data.get("mobile", ""),
            area_code=data.get("area_code", ""),
            enabled=data.get("enabled", True),
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            created_at=data.get("created_at", 0.0) or __import__("time").time(),
            last_login_at=data.get("last_login_at"),
            last_error=data.get("last_error"),
        )


@dataclass
class DeepseekSession:
    """Active chat session for a DeepSeek account."""

    session_id: str
    created_at: float = field(default_factory=lambda: __import__("time").time())
    last_used_at: float = field(default_factory=lambda: __import__("time").time())
    request_count: int = 0
    inflight: int = 0

    def touch(self) -> None:
        self.last_used_at = __import__("time").time()

    def is_expired(self, ttl_seconds: float = 3600.0) -> bool:
        return (__import__("time").time() - self.last_used_at) > ttl_seconds
