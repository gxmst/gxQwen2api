"""Ring-buffer event logger for the admin UI + structured stderr logging."""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("gx_qwen2api")


class EventLogger:
    """Dual-purpose logger: structured logs to stderr + ring buffer for UI."""

    MAX_ENTRIES = 200

    def __init__(self) -> None:
        self._ring: deque[dict[str, Any]] = deque(maxlen=self.MAX_ENTRIES)

    def _emit(self, level: int, event: str, data: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "event": event,
            **data,
        }
        self._ring.append(record)
        logger.log(level, f"[{event}] {data.get('account_id', '-')} {data.get('detail', '') or data.get('reason', '')}")

    # ── Ring buffer access ────────────────────────────────────────

    def get_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._ring)[-limit:]

    def clear_logs(self) -> None:
        self._ring.clear()

    # ── Request / response ────────────────────────────────────────

    def proxy_request(
        self,
        request_id: str,
        model: str,
        account_id: str,
        token_count: int,
        is_streaming: bool = False,
    ) -> None:
        self._emit(
            logging.INFO, "request",
            {
                "account_id": account_id,
                "request_id": request_id,
                "model": model,
                "token_count": token_count,
                "is_streaming": is_streaming,
                "detail": f"→ {model} ({'stream' if is_streaming else 'regular'}) {token_count}tok",
            },
        )

    def proxy_response(
        self,
        request_id: str,
        status_code: int,
        account_id: str,
        latency_ms: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        qwen_id: str | None = None,
    ) -> None:
        tok_part = ""
        if input_tokens is not None:
            tok_part = f" {input_tokens}+{output_tokens or '?'}tok"
        self._emit(
            logging.INFO, "response",
            {
                "account_id": account_id,
                "request_id": request_id,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "qwen_id": qwen_id,
                "detail": f"← {status_code} {latency_ms}ms{tok_part}",
            },
        )

    def proxy_error(
        self,
        request_id: str,
        status_code: int,
        account_id: str,
        error_message: str,
    ) -> None:
        self._emit(
            logging.ERROR, "error",
            {
                "account_id": account_id,
                "request_id": request_id,
                "status_code": status_code,
                "error_message": error_message,
                "detail": f"✗ {status_code} {error_message[:200]}",
            },
        )

    # ── Token refresh ─────────────────────────────────────────────

    def refresh_started(self, account_id: str, url: str) -> None:
        self._emit(
            logging.INFO, "refresh_start",
            {"account_id": account_id, "url": url, "detail": f"Refreshing token → {url}"},
        )

    def refresh_succeeded(
        self, account_id: str, elapsed_ms: float, expires_at: str
    ) -> None:
        self._emit(
            logging.INFO, "refresh_ok",
            {
                "account_id": account_id,
                "elapsed_ms": round(elapsed_ms, 1),
                "expires_at": expires_at,
                "detail": f"OK in {elapsed_ms:.0f}ms, expires {expires_at}",
            },
        )

    def refresh_failed(
        self,
        account_id: str,
        reason: str,
        url: str = "",
        status_code: int | None = None,
        content_type: str = "",
        elapsed_ms: float | None = None,
        body_preview: str = "",
        error: str = "",
    ) -> None:
        detail = f"FAIL {reason}"
        if status_code:
            detail += f" HTTP{status_code}"
        if elapsed_ms is not None:
            detail += f" {elapsed_ms:.0f}ms"
        if error:
            detail += f" — {error[:150]}"
        elif body_preview:
            detail += f" — {body_preview[:150]}"
        self._emit(
            logging.ERROR, "refresh_fail",
            {
                "account_id": account_id,
                "reason": reason,
                "url": url,
                "status_code": status_code,
                "content_type": content_type,
                "elapsed_ms": elapsed_ms,
                "error": error[:500] if error else body_preview[:500],
                "detail": detail,
            },
        )

    # ── Server ────────────────────────────────────────────────────

    def server_started(self, host: str, port: int) -> None:
        self._emit(
            logging.INFO, "server",
            {"host": host, "port": port, "detail": f"Server started on {host}:{port}"},
        )

    def shutdown(self, reason: str) -> None:
        self._emit(
            logging.INFO, "shutdown",
            {"reason": reason, "detail": f"Shutdown: {reason}"},
        )

    # ── Auto-refresh ──────────────────────────────────────────────

    def auto_refresh_start(self) -> None:
        self._emit(
            logging.INFO, "auto_refresh_start",
            {"detail": "Auto-refresh cycle started"},
        )

    def auto_refresh_ok(self, account_id: str, elapsed_ms: float, expires_at: str) -> None:
        self._emit(
            logging.INFO, "auto_refresh_ok",
            {
                "account_id": account_id,
                "elapsed_ms": round(elapsed_ms, 1),
                "expires_at": expires_at,
                "detail": f"Auto-refresh OK: {account_id}, expires {expires_at}",
            },
        )

    def auto_refresh_skip(self, account_id: str, reason: str) -> None:
        self._emit(
            logging.INFO, "auto_refresh_skip",
            {
                "account_id": account_id,
                "reason": reason,
                "detail": f"Auto-refresh skipped: {account_id} ({reason})",
            },
        )

    def auto_refresh_fail(self, account_id: str, reason: str, error: str = "") -> None:
        detail = f"Auto-refresh failed: {account_id} ({reason})"
        if error:
            detail += f" — {error[:150]}"
        self._emit(
            logging.ERROR, "auto_refresh_fail",
            {
                "account_id": account_id,
                "reason": reason,
                "error": error[:500] if error else "",
                "detail": detail,
            },
        )


event_logger = EventLogger()
