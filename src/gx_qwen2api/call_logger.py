"""Ring-buffer call logger for per-request traceability in the admin UI."""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("gx2api.call_logger")

_MASK_LENGTH = 4


def _mask_key(key: str) -> str:
    """Mask an API key or token showing only prefix and suffix."""
    key = str(key)
    if len(key) <= 8:
        return key[:2] + "****"
    return key[:_MASK_LENGTH] + "****" + key[-_MASK_LENGTH:]


def _mask_error(msg: str) -> str:
    """Truncate and strip sensitive patterns from error messages."""
    msg = str(msg)[:200]
    for pattern in ("Bearer ", "nvapi-", "sk-", "dsk-"):
        idx = msg.lower().find(pattern.lower())
        if idx >= 0:
            end = msg.find(" ", idx)
            if end < 0:
                end = len(msg)
            token = msg[idx:end]
            msg = msg.replace(token, _mask_key(token))
    return msg


class CallLogger:
    """Lightweight ring-buffer for chat completion call records."""

    MAX_ENTRIES = 500

    def __init__(self) -> None:
        self._ring: deque[dict[str, Any]] = deque(maxlen=self.MAX_ENTRIES)

    def log(
        self,
        request_id: str,
        provider: str,
        account_id: str,
        model: str,
        stream: bool,
        status: str,
        http_status: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: int | None = None,
        error_message: str = "",
        request_preview: str = "",
    ) -> None:
        entry = {
            "id": request_id,
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "timestamp": time.time(),
            "provider": provider,
            "account_id": account_id,
            "model": model,
            "stream": stream,
            "status": status,
            "http_status": http_status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "error_message": _mask_error(error_message) if error_message else None,
        }
        if request_preview:
            entry["request_preview"] = request_preview[:200]
        self._ring.append(entry)

    def get_logs(
        self,
        limit: int = 100,
        provider: str = "",
        status: str = "",
        model: str = "",
    ) -> list[dict[str, Any]]:
        entries = list(self._ring)
        entries.reverse()

        if provider:
            entries = [e for e in entries if e["provider"] == provider]
        if status:
            entries = [e for e in entries if e["status"] == status]
        if model:
            entries = [e for e in entries if model in e["model"]]

        return entries[:limit]

    def get_stats(self) -> dict[str, Any]:
        entries = list(self._ring)
        if not entries:
            return {"total": 0, "by_provider": {}, "by_status": {}}

        by_provider: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for e in entries:
            p = e["provider"]
            s = e["status"]
            by_provider[p] = by_provider.get(p, 0) + 1
            by_status[s] = by_status.get(s, 0) + 1

        return {
            "total": len(entries),
            "by_provider": by_provider,
            "by_status": by_status,
        }

    def clear(self) -> None:
        self._ring.clear()


call_logger = CallLogger()
