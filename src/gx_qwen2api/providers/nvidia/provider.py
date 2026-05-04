"""NVIDIA NIM API provider with multi-key round-robin and OpenAI compatibility."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from ...call_logger import call_logger
from ...config import settings
from ...event_logger import event_logger
from .models import (
    NvidiaKeyState,
    NvidiaKeyStatus,
    get_builtin_models,
)

logger = logging.getLogger("gx2api.nvidia")

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"


class NvidiaProvider:
    """NVIDIA NIM API provider supporting multi-key round-robin.

    Credentials are stored as JSON files in CREDS_DIR with format:
    {
        "provider": "nvidia",
        "name": "my-key",
        "api_key": "nvapi-...",
        "enabled": true,
        "models": [{"upstream": "...", "local": "...", "enabled": true}]
    }
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self._keys: dict[str, NvidiaKeyState] = {}
        self._last_success_id: str | None = None

    async def start(self) -> None:
        self._scan_keys()

    async def stop(self) -> None:
        self._keys.clear()

    def has_accounts(self) -> bool:
        return any(k.enabled for k in self._keys.values())

    def can_handle_model(self, model: str) -> bool:
        for key_state in self._keys.values():
            if not key_state.enabled or key_state.status == NvidiaKeyStatus.DISABLED:
                continue
            for m in key_state.models:
                if m.get("local") == model and m.get("enabled", True):
                    return True
        return False

    def list_models_payload(self) -> list[dict[str, str | int]]:
        now = int(time.time())
        seen: set[str] = set()
        models: list[dict[str, str | int]] = []
        for key_state in self._keys.values():
            if not key_state.enabled:
                continue
            for m in key_state.models:
                local = m.get("local", "")
                if local and m.get("enabled", True) and local not in seen:
                    seen.add(local)
                    models.append({"id": local, "object": "model", "created": now, "owned_by": "nvidia"})
        return models

    def get_upstream_model(self, local_model: str) -> str | None:
        for key_state in self._keys.values():
            for m in key_state.models:
                if m.get("local") == local_model and m.get("enabled", True):
                    return m.get("upstream")
        return None

    def verify_account(self, key_id: str) -> dict[str, Any]:
        key_state = self._keys.get(key_id)
        if not key_state:
            return {"status": "failed", "key_id": key_id, "detail": "Key not found", "valid": False}
        return {"status": "ok", "key_id": key_id, "detail": "Key loaded", "valid": key_state.enabled}

    # ── Key management ──────────────────────────────────────────

    def _scan_keys(self) -> None:
        """Scan creds dir for NVIDIA *.json credential files."""
        try:
            files = sorted(settings.creds_dir.glob("nvidia_*.json"))
        except (OSError, PermissionError):
            return

        for fp in files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("provider") != "nvidia":
                continue

            api_key = data.get("api_key", "")
            if not api_key:
                continue

            key_id = fp.stem
            key_state = NvidiaKeyState(
                key_id=key_id,
                name=data.get("name", key_id),
                api_key=api_key,
                enabled=data.get("enabled", True),
            )

            models = data.get("models")
            if isinstance(models, list) and models:
                key_state.models = models
            else:
                key_state.models = get_builtin_models()

            self._keys[key_id] = key_state

    def reload_key(self, key_id: str) -> bool:
        fp = settings.creds_dir / f"{key_id}.json"
        if not fp.exists():
            return False
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(data, dict) or data.get("provider") != "nvidia":
            return False

        api_key = data.get("api_key", "")
        if not api_key:
            return False

        ks = self._keys.get(key_id)
        if ks:
            ks.api_key = api_key
            ks.name = data.get("name", key_id)
            ks.enabled = data.get("enabled", True)
            models = data.get("models")
            if isinstance(models, list) and models:
                ks.models = models
        else:
            ks = NvidiaKeyState(
                key_id=key_id,
                name=data.get("name", key_id),
                api_key=api_key,
                enabled=data.get("enabled", True),
            )
            models = data.get("models")
            if isinstance(models, list) and models:
                ks.models = models
            else:
                ks.models = get_builtin_models()
            self._keys[key_id] = ks
        return True

    def get_all_keys(self) -> list[dict[str, Any]]:
        return [k.to_dict() for k in self._keys.values()]

    def get_key(self, key_id: str) -> NvidiaKeyState | None:
        return self._keys.get(key_id)

    def enable_key(self, key_id: str) -> bool:
        ks = self._keys.get(key_id)
        if ks:
            ks.enabled = True
            self._save_key_state(ks)
            return True
        return False

    def disable_key(self, key_id: str) -> bool:
        ks = self._keys.get(key_id)
        if ks:
            ks.enabled = False
            self._save_key_state(ks)
            return True
        return False

    def delete_key(self, key_id: str) -> bool:
        if key_id in self._keys:
            del self._keys[key_id]
        fp = settings.creds_dir / f"{key_id}.json"
        if fp.exists():
            try:
                fp.unlink()
            except OSError:
                pass
        return True

    def add_key(self, key_id: str, api_key: str, name: str = "", models: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not re.match(r"^[A-Za-z0-9_-]+$", key_id):
            return {"status": "failed", "detail": "key_id must be alphanumeric with dashes/underscores"}

        settings.creds_dir.mkdir(parents=True, exist_ok=True)
        target = settings.creds_dir / f"{key_id}.json"

        data: dict[str, Any] = {
            "provider": "nvidia",
            "name": name or key_id,
            "api_key": api_key,
            "enabled": True,
        }
        if models:
            data["models"] = models
        else:
            data["models"] = get_builtin_models()

        action = "new"
        if target.exists():
            action = "overwrite"

        try:
            target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            return {"status": "failed", "detail": f"Failed to write file: {exc}"}

        ks = NvidiaKeyState(
            key_id=key_id,
            name=name or key_id,
            api_key=api_key,
            enabled=True,
        )
        if models:
            ks.models = models
        else:
            ks.models = get_builtin_models()
        self._keys[key_id] = ks

        return {"status": "ok", "key_id": key_id, "action": action}

    def _save_key_state(self, ks: NvidiaKeyState) -> None:
        fp = settings.creds_dir / f"{ks.key_id}.json"
        try:
            data = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["provider"] = "nvidia"
        data["name"] = ks.name
        data["api_key"] = ks.api_key
        data["enabled"] = ks.enabled
        data["models"] = ks.models
        try:
            fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ── Model management ──────────────────────────────────────

    async def fetch_upstream_models(self, key_id: str) -> dict[str, Any]:
        """Fetch available models from NVIDIA NIM API for a given key."""
        ks = self._keys.get(key_id)
        if not ks:
            return {"status": "failed", "detail": "Key not found"}

        try:
            resp = await self.client.get(
                f"{NVIDIA_API_BASE}/models",
                headers={"Authorization": f"Bearer {ks.api_key}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                upstream_models: list[dict[str, str]] = []
                model_list = data.get("data", data) if isinstance(data, dict) else data
                if isinstance(model_list, list):
                    for m in model_list:
                        if isinstance(m, dict) and "id" in m:
                            upstream_models.append({
                                "upstream": m["id"],
                                "local": _suggest_local_name(m["id"]),
                                "enabled": True,
                            })
                return {"status": "ok", "models": upstream_models}
            return {
                "status": "failed",
                "detail": f"HTTP {resp.status_code}",
                "error": resp.text[:500],
                "models": get_builtin_models(),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "detail": f"Network error: {exc}",
                "models": get_builtin_models(),
            }

    def set_key_models(self, key_id: str, models: list[dict[str, Any]]) -> dict[str, Any]:
        ks = self._keys.get(key_id)
        if not ks:
            return {"status": "failed", "detail": "Key not found"}
        ks.models = models
        self._save_key_state(ks)
        return {"status": "ok", "key_id": key_id}

    # ── Key verification ─────────────────────────────────────

    async def verify_key(self, key_id: str) -> dict[str, Any]:
        ks = self._keys.get(key_id)
        if not ks:
            return {"status": "failed", "key_id": key_id, "detail": "Key not found", "valid": False}

        try:
            resp = await self.client.get(
                f"{NVIDIA_API_BASE}/models",
                headers={"Authorization": f"Bearer {ks.api_key}"},
            )
            if resp.status_code == 200:
                ks.mark_success()
                self._last_success_id = key_id
                return {"status": "ok", "key_id": key_id, "detail": "API key is valid", "valid": True}
            elif resp.status_code in (401, 403):
                error_body = resp.text[:300]
                ks.mark_auth_error(f"HTTP {resp.status_code}: {error_body}")
                return {"status": "failed", "key_id": key_id, "detail": f"Authentication failed (HTTP {resp.status_code})", "valid": False, "error_type": "auth_error"}
            elif resp.status_code == 429:
                ks.mark_rate_limited(120)
                return {"status": "failed", "key_id": key_id, "detail": "Rate limited", "valid": False, "error_type": "rate_limited"}
            else:
                return {"status": "failed", "key_id": key_id, "detail": f"Unexpected HTTP {resp.status_code}", "valid": False}
        except Exception as exc:
            return {"status": "failed", "key_id": key_id, "detail": f"Network error: {exc}", "valid": False, "error_type": "network"}

    # ── Key selection ────────────────────────────────────────

    # ── Chat completions ─────────────────────────────────────

    def _get_eligible_keys(self, model: str) -> list[NvidiaKeyState]:
        """Return all enabled, non-auth-error, non-cooldown keys supporting the model."""
        eligible: list[NvidiaKeyState] = []
        for ks in self._keys.values():
            if not ks.enabled:
                continue
            if ks.status == NvidiaKeyStatus.DISABLED:
                continue
            if ks.status == NvidiaKeyStatus.AUTH_ERROR:
                continue
            if ks.is_cooldown:
                continue
            has_model = any(m.get("local") == model and m.get("enabled", True) for m in ks.models)
            if has_model:
                eligible.append(ks)
        return eligible

    async def chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        request_id: str = "",
    ) -> JSONResponse | StreamingResponse:
        upstream_model = self.get_upstream_model(model)
        if not upstream_model:
            call_logger.log(request_id=request_id, provider="nvidia", account_id="", model=model,
                            stream=stream, status="error", http_status=400, latency_ms=0,
                            error_message=f"Unsupported model: {model}")
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Unsupported NVIDIA model: {model}", "type": "invalid_request_error"}},
            )

        payload: dict[str, Any] = {
            "model": upstream_model,
            "messages": messages,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        tried: set[str] = set()

        while True:
            eligible = self._get_eligible_keys(model)
            eligible = [k for k in eligible if k.key_id not in tried]
            if not eligible:
                call_logger.log(request_id=request_id, provider="nvidia", account_id="",
                                model=model, stream=stream, status="error", http_status=503,
                                latency_ms=int((time.time() - __import__("time").time()) * 1000) if False else 0,
                                error_message="No healthy NVIDIA key available")
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "No healthy NVIDIA key available", "type": "api_error"}},
                )

            # Round-robin: start after last success
            ks = eligible[0]
            if self._last_success_id:
                idx = next((i for i, k in enumerate(eligible) if k.key_id == self._last_success_id), -1)
                if idx >= 0:
                    ks = eligible[(idx + 1) % len(eligible)]

            tried.add(ks.key_id)
            t0 = time.time()

            try:
                if stream:
                    return StreamingResponse(
                        self._stream_with_failover(payload, upstream_model, model, request_id, t0),
                        media_type="text/event-stream",
                    )

                resp = await self.client.post(
                    f"{NVIDIA_API_BASE}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {ks.api_key}", "Content-Type": "application/json"},
                    timeout=httpx.Timeout(300, connect=10),
                )
                if resp.status_code == 200:
                    ks.mark_success()
                    self._last_success_id = ks.key_id
                    ks.request_count += 1
                    latency_ms = int((time.time() - t0) * 1000)
                    data = resp.json()
                    usage = data.get("usage", {})
                    call_logger.log(request_id=request_id, provider="nvidia", account_id=ks.key_id,
                                    model=model, stream=False, status="success", http_status=200,
                                    input_tokens=usage.get("prompt_tokens"),
                                    output_tokens=usage.get("completion_tokens"),
                                    total_tokens=usage.get("total_tokens"), latency_ms=latency_ms)
                    return JSONResponse(content=data)

                body = resp.text[:500]
                await resp.aclose()

                if resp.status_code in (401, 403):
                    ks.mark_auth_error(f"HTTP {resp.status_code}")
                    continue  # try next key
                if resp.status_code == 429:
                    ks.mark_rate_limited(120)
                    continue  # try next key

                latency_ms = int((time.time() - t0) * 1000)
                call_logger.log(request_id=request_id, provider="nvidia", account_id=ks.key_id,
                                model=model, stream=False, status="error",
                                http_status=resp.status_code, latency_ms=latency_ms, error_message=body)
                return JSONResponse(status_code=resp.status_code,
                                    content={"error": {"message": f"NVIDIA upstream error: HTTP {resp.status_code}", "type": "api_error"}})

            except Exception as exc:
                latency_ms = int((time.time() - t0) * 1000)
                call_logger.log(request_id=request_id, provider="nvidia", account_id=ks.key_id,
                                model=model, stream=stream, status="error",
                                http_status=None, latency_ms=latency_ms, error_message=str(exc))
                if len(eligible) > 1:
                    continue  # try next key
                return JSONResponse(status_code=502,
                                    content={"error": {"message": f"NVIDIA request failed: {exc}", "type": "api_error"}})

    async def _stream_with_failover(
        self,
        payload: dict[str, Any],
        upstream_model: str,
        model: str,
        request_id: str,
        t0: float,
    ) -> AsyncGenerator[bytes, None]:
        tried: set[str] = set()
        last_error = ""

        while True:
            eligible = self._get_eligible_keys(model)
            eligible = [k for k in eligible if k.key_id not in tried]
            if not eligible:
                latency_ms = int((time.time() - t0) * 1000)
                call_logger.log(request_id=request_id, provider="nvidia", account_id="",
                                model=model, stream=True, status="error", http_status=503,
                                latency_ms=latency_ms, error_message="No healthy NVIDIA key available")
                err_json = json.dumps({"error": {"message": "No healthy NVIDIA key available", "type": "api_error"}})
                yield f"data: {err_json}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

            ks = eligible[0]
            if self._last_success_id:
                idx = next((i for i, k in enumerate(eligible) if k.key_id == self._last_success_id), -1)
                if idx >= 0:
                    ks = eligible[(idx + 1) % len(eligible)]

            tried.add(ks.key_id)

            try:
                async with self.client.stream(
                    "POST",
                    f"{NVIDIA_API_BASE}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {ks.api_key}", "Content-Type": "application/json"},
                    timeout=httpx.Timeout(300, connect=10),
                ) as resp:
                    if resp.status_code == 200:
                        ks.mark_success()
                        self._last_success_id = ks.key_id
                        ks.request_count += 1
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                        latency_ms = int((time.time() - t0) * 1000)
                        call_logger.log(request_id=request_id, provider="nvidia", account_id=ks.key_id,
                                        model=model, stream=True, status="success", http_status=200, latency_ms=latency_ms)
                        return

                    body = (await resp.aread()).decode(errors="replace")[:500]
                    last_error = f"HTTP {resp.status_code}: {body}"

                    if resp.status_code in (401, 403):
                        ks.mark_auth_error(f"HTTP {resp.status_code}")
                        continue  # try next key
                    if resp.status_code == 429:
                        ks.mark_rate_limited(120)
                        continue  # try next key

                    # Non-retryable error (400, 500, etc.) — stop immediately
                    latency_ms = int((time.time() - t0) * 1000)
                    call_logger.log(request_id=request_id, provider="nvidia", account_id=ks.key_id,
                                    model=model, stream=True, status="error",
                                    http_status=resp.status_code, latency_ms=latency_ms, error_message=body)
                    err_json = json.dumps({"error": {"message": f"NVIDIA upstream error: HTTP {resp.status_code}", "type": "api_error"}})
                    yield f"data: {err_json}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
            except Exception as exc:
                last_error = str(exc)
                logger.error("NVIDIA stream probe error for %s: %s", ks.key_id, exc)

        # All keys exhausted
        latency_ms = int((time.time() - t0) * 1000)
        call_logger.log(request_id=request_id, provider="nvidia", account_id="",
                        model=model, stream=True, status="error", http_status=503,
                        latency_ms=latency_ms, error_message=last_error[:200])
        err_json = json.dumps({"error": {"message": f"NVIDIA request failed: {last_error[:200]}", "type": "api_error"}})
        yield f"data: {err_json}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def _handle_error(
        self,
        resp: httpx.Response,
        ks: NvidiaKeyState,
        request_id: str,
        model: str,
        t0: float,
        stream: bool,
    ) -> JSONResponse:
        body = resp.text[:500]
        latency_ms = int((time.time() - t0) * 1000)
        status_code = resp.status_code

        if status_code in (401, 403):
            ks.mark_auth_error(f"HTTP {status_code}")
            call_logger.log(
                request_id=request_id, provider="nvidia", account_id=ks.key_id,
                model=model, stream=stream, status="auth_error",
                http_status=status_code, latency_ms=latency_ms, error_message=body,
            )
            await resp.aclose()
            return JSONResponse(status_code=status_code, content={"error": {"message": f"NVIDIA auth error: HTTP {status_code}", "type": "auth_error"}})

        if status_code == 429:
            ks.mark_rate_limited(120)
            call_logger.log(
                request_id=request_id, provider="nvidia", account_id=ks.key_id,
                model=model, stream=stream, status="rate_limited",
                http_status=status_code, latency_ms=latency_ms, error_message=body,
            )
            await resp.aclose()
            return JSONResponse(status_code=503, content={"error": {"message": "NVIDIA key rate limited", "type": "rate_limit_error"}})

        call_logger.log(
            request_id=request_id, provider="nvidia", account_id=ks.key_id,
            model=model, stream=stream, status="error",
            http_status=status_code, latency_ms=latency_ms, error_message=body,
        )
        await resp.aclose()
        return JSONResponse(status_code=status_code, content={"error": {"message": f"NVIDIA upstream error: HTTP {status_code}", "type": "api_error"}})


def _suggest_local_name(upstream: str) -> str:
    """Derive a local model name from upstream model ID."""
    name = upstream.replace("/", "-").replace(".", "-")
    return f"nvidia-{name}"
