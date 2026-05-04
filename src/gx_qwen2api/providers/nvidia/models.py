"""NVIDIA NIM API models, key state and health tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NvidiaKeyStatus(str, Enum):
    VALID = "valid"
    AUTH_ERROR = "auth_error"
    RATE_LIMITED = "rate_limited"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


@dataclass
class NvidiaKeyState:
    key_id: str
    name: str = ""
    api_key: str = ""
    enabled: bool = True
    status: NvidiaKeyStatus = NvidiaKeyStatus.UNKNOWN
    status_reason: str = ""
    cooldown_until: float = 0.0
    last_success_at: float = 0.0
    last_error_at: float = 0.0
    last_error: str = ""
    models: list[dict[str, Any]] = field(default_factory=list)
    request_count: int = 0
    error_count: int = 0

    @property
    def masked_key(self) -> str:
        k = self.api_key
        if not k:
            return ""
        if len(k) <= 8:
            return k[:2] + "****"
        return k[:4] + "****" + k[-4:]

    @property
    def is_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def cooldown_remaining(self) -> float:
        if not self.is_cooldown:
            return 0.0
        return max(0.0, self.cooldown_until - time.time())

    def mark_success(self) -> None:
        self.status = NvidiaKeyStatus.VALID
        self.last_success_at = time.time()
        self.last_error = ""

    def mark_rate_limited(self, cooldown: float = 120.0) -> None:
        self.status = NvidiaKeyStatus.RATE_LIMITED
        self.cooldown_until = time.time() + cooldown
        self.last_error_at = time.time()
        self.error_count += 1

    def mark_auth_error(self, reason: str = "") -> None:
        self.status = NvidiaKeyStatus.AUTH_ERROR
        self.last_error_at = time.time()
        self.last_error = reason
        self.error_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "name": self.name,
            "masked_key": self.masked_key,
            "enabled": self.enabled,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "cooldown_remaining": round(self.cooldown_remaining, 1),
            "last_success_at": self.last_success_at or None,
            "last_error_at": self.last_error_at or None,
            "last_error": self.last_error[:200] if self.last_error else None,
            "model_count": len(self.models),
            "request_count": self.request_count,
            "error_count": self.error_count,
        }


# Pre-defined NVIDIA NIM models (also used as fallback)
_BUILTIN_MODELS: list[dict[str, str]] = [
    {"upstream": "deepseek-ai/deepseek-v4-flash", "local": "nvidia-deepseek-v4-flash"},
    {"upstream": "deepseek-ai/deepseek-v4-pro", "local": "nvidia-deepseek-v4-pro"},
    {"upstream": "meta/llama-4-maverick-17b-128e-instruct", "local": "nvidia-llama4-maverick"},
    {"upstream": "meta/llama-4-scout-17b-16e-instruct", "local": "nvidia-llama4-scout"},
    {"upstream": "qwen/qwen3-235b-a22b", "local": "nvidia-qwen3-235b"},
]


def get_builtin_models() -> list[dict[str, Any]]:
    return [{"upstream": m["upstream"], "local": m["local"], "enabled": True} for m in _BUILTIN_MODELS]
