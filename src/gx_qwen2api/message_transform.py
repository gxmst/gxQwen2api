"""Message transformations: cache_control tagging, system prompt injection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import settings


def _add_cache_control(message: dict[str, Any]) -> dict[str, Any]:
    """Add cache_control to the last content item of a message."""
    content = message.get("content")

    if isinstance(content, str):
        return {
            **message,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }

    if isinstance(content, list) and content:
        new_parts: list[Any] = [*content]
        last: Any = new_parts[-1]
        if isinstance(last, dict):
            new_parts[-1] = {**last, "cache_control": {"type": "ephemeral"}}
        return {**message, "content": new_parts}

    return message


def _load_system_prompt() -> str | None:
    """Load custom system prompt from QWEN_SYSTEM_PROMPT or sys-prompt.txt."""
    env_val = os.getenv("QWEN_SYSTEM_PROMPT")
    if env_val:
        return env_val
    prompt_file = os.getenv("QWEN_SYSTEM_PROMPT_FILE")
    if prompt_file:
        p = Path(prompt_file)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    # Default: search for sys-prompt.txt starting from nearest parent directory
    # to avoid hardcoding path depth (works both in Docker /app/ and local dev)
    base = Path(__file__).resolve().parent
    for parent in [base, *base.parents]:
        candidate = parent / "sys-prompt.txt"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    return None


def transform_messages(
    messages: list[dict[str, Any]],
    model: str,
    *,
    streaming: bool = False,
) -> list[dict[str, Any]]:
    """Add cache_control matching the real client.

    Streaming: cache_control on system message + last message.
    Non-streaming: cache_control on system message only.
    Also inject custom system prompt if configured.
    """
    transformed = list(messages)

    sys_idx = next(
        (i for i, m in enumerate(transformed) if m.get("role") == "system"), None
    )

    # DashScope API requires a system message to be present
    if sys_idx is None:
        transformed.insert(0, {"role": "system", "content": ""})
        sys_idx = 0

    # Inject custom system prompt
    custom_prompt = _load_system_prompt()
    if custom_prompt and sys_idx is not None:
        existing = transformed[sys_idx].get("content", "")
        if isinstance(existing, str):
            transformed[sys_idx]["content"] = f"{custom_prompt}\n\n{existing}" if existing else custom_prompt

    # Apply cache_control to system message (always) and last message (streaming only)
    if sys_idx is not None:
        transformed[sys_idx] = _add_cache_control(transformed[sys_idx])

    if streaming and transformed:
        last_idx = len(transformed) - 1
        if last_idx != sys_idx:
            transformed[last_idx] = _add_cache_control(transformed[last_idx])

    return transformed
