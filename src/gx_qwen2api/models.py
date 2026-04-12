"""Qwen model definitions, aliases, and token limits."""

from typing import Any
from .config import settings


MODEL_ALIASES: dict[str, str] = {
    "qwen3.5-plus": "coder-model",
    "qwen3.6-plus": "coder-model",
}


def is_auth_error(status: int | None, message: str) -> bool:
    """Check if error is authentication-related (401/403 or specific auth message).
    
    We specifically exclude 400 and 504 from the status-only list as they are
    often validation or network issues, not auth issues.
    """
    if status in (401, 403):
        return True
    msg_lower = message.lower()
    return any(
        x in msg_lower
        for x in [
            "unauthorized", "forbidden", "token expired", "invalid api key",
            "invalid access token", "authentication", "access denied",
        ]
    )


def is_quota_error(status: int | None, message: str) -> bool:
    """Check if error is quota/rate limit related (429 or quota message)."""
    if status == 429:
        return True
    msg_lower = message.lower()
    return any(
        x in msg_lower
        for x in ["insufficient_quota", "quota exceeded", "rate limit", "too many requests"]
    )


def is_validation_error(message: str) -> bool:
    """Check if error is a validation error."""
    return "validation error" in message.lower() or "invalid" in message.lower()


def make_error_response(
    message: str,
    error_type: str = "api_error",
    code: str | None = None,
) -> dict[str, dict[str, str | int] | dict[str, str]]:
    """Create OpenAI-compatible error response."""
    error: dict[str, str | int] = {"message": message, "type": error_type}
    if code:
        error["code"] = code
    return {"error": error}


MODELS: list[dict[str, str | int]] = [
    {"id": "qwen3-coder-plus", "object": "model", "created": 1754686206, "owned_by": "qwen"},
    {"id": "qwen3-coder-flash", "object": "model", "created": 1754686206, "owned_by": "qwen"},
    {"id": "qwen3.6-plus", "object": "model", "created": 1754686206, "owned_by": "qwen"},
    {"id": "coder-model", "object": "model", "created": 1754686206, "owned_by": "qwen"},
    {"id": "vision-model", "object": "model", "created": 1754686206, "owned_by": "qwen"},
]

MODEL_MAX_TOKENS: dict[str, int] = {
    "vision-model": 32768,
    "qwen3-vl-plus": 32768,
    "qwen3-vl-max": 32768,
    "qwen3.5-plus": 65536,
    "qwen3.6-plus": 65536,
    "coder-model": 65536,
}

EFFORT_BUDGET_MAP: dict[str, int | None] = {"low": 1024, "medium": 8192, "high": None}


def resolve_thinking_params(body: dict[str, Any]) -> dict[str, Any]:
    """Map reasoning.effort to enable_thinking/thinking_budget for the Qwen API."""
    result: dict[str, Any] = {}
    if "enable_thinking" in body:
        result["enable_thinking"] = body["enable_thinking"]
    if "thinking_budget" in body:
        result["thinking_budget"] = body["thinking_budget"]
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        effort = reasoning["effort"]
        if effort == "none":
            result["enable_thinking"] = False
        elif effort in EFFORT_BUDGET_MAP:
            result["enable_thinking"] = True
            budget = EFFORT_BUDGET_MAP[effort]
            if budget is not None:
                result["thinking_budget"] = budget
    return result


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model) or settings.default_model


def clamp_max_tokens(model: str, max_tokens: int) -> int:
    limit = MODEL_MAX_TOKENS.get(model)
    if limit and max_tokens > limit:
        return limit
    return max_tokens
