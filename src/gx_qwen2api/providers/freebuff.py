"""Freebuff / Codebuff provider support."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from ..account_pool import AccountPool, AccountState
from ..config import settings
from ..event_logger import event_logger
from ..models import ChatCompletionRequest, make_error_response

logger = logging.getLogger("gx_qwen2api.freebuff")

FREEBUFF_FALLBACK_MODELS: dict[str, list[str]] = {
    "base2-free": ["minimax/minimax-m2.7", "z-ai/glm-5.1"],
    "file-picker": ["google/gemini-2.5-flash-lite"],
    "file-picker-max": ["google/gemini-3.1-flash-lite-preview"],
    "file-lister": ["google/gemini-3.1-flash-lite-preview"],
    "researcher-web": ["google/gemini-3.1-flash-lite-preview"],
    "researcher-docs": ["google/gemini-3.1-flash-lite-preview"],
    "basher": ["google/gemini-3.1-flash-lite-preview"],
    "editor-lite": ["minimax/minimax-m2.7", "z-ai/glm-5.1"],
    "code-reviewer-lite": ["minimax/minimax-m2.7", "z-ai/glm-5.1"],
}


def _generate_client_session_id() -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(alphabet) for _ in range(13))


def _retry_after_duration(header_value: str | None) -> int:
    if not header_value:
        return 0
    try:
        seconds = int(header_value.strip())
    except (TypeError, ValueError):
        return 0
    return max(0, seconds)


def _is_run_invalid(status_code: int, body: bytes) -> bool:
    if status_code != 400:
        return False
    message = body.decode("utf-8", errors="ignore").lower()
    return "runid not found" in message or "runid not running" in message


def _openai_error_from_upstream(status_code: int, body: bytes) -> JSONResponse:
    trimmed = body.strip()
    if trimmed:
        try:
            payload = json.loads(trimmed)
        except Exception:
            payload = {
                "error": {
                    "message": trimmed.decode("utf-8", errors="ignore")[:500],
                    "type": "upstream_error",
                }
            }
        else:
            if isinstance(payload, dict) and "error" in payload:
                error = payload["error"]
                if isinstance(error, dict):
                    message = str(error.get("message") or payload.get("message") or trimmed.decode("utf-8", errors="ignore"))
                    error_type = str(error.get("type") or "upstream_error")
                    code = error.get("code")
                    return JSONResponse(
                        status_code=status_code,
                        content=make_error_response(message, error_type, str(code) if code else None),
                    )
            if isinstance(payload, dict) and "message" in payload:
                return JSONResponse(
                    status_code=status_code,
                    content=make_error_response(str(payload["message"]), "upstream_error"),
                )
    return JSONResponse(
        status_code=status_code,
        content=make_error_response("Upstream request failed", "upstream_error"),
    )


@dataclass
class _ManagedRun:
    run_id: str
    agent_id: str
    started_at: float
    inflight: int = 0
    request_count: int = 0


@dataclass
class _RunLease:
    account: AccountState
    run: _ManagedRun


class FreebuffModelRegistry:
    _block_re = re.compile(r"'([^']+)':\s*new\s+Set\(\[([^\]]*)\]\)")
    _model_re = re.compile(r"'([^']+)'")

    def __init__(self) -> None:
        self._agent_models: dict[str, list[str]] = {}
        self._model_to_agent: dict[str, str] = {}
        self._all_models: list[str] = []

    async def refresh(self, client: httpx.AsyncClient) -> None:
        try:
            resp = await client.get(settings.freebuff_model_source_url, timeout=30.0)
            resp.raise_for_status()
            agent_models = self._parse_all_free_models(resp.text)
            if not agent_models:
                raise ValueError("no free agents found")
        except Exception as exc:
            logger.warning("Freebuff model registry refresh failed, using fallback: %s", exc)
            agent_models = FREEBUFF_FALLBACK_MODELS

        model_to_agent = self._build_model_mapping(agent_models)
        self._agent_models = agent_models
        self._model_to_agent = model_to_agent
        self._all_models = sorted(model_to_agent.keys())

    def models(self) -> list[str]:
        return list(self._all_models)

    def agent_ids(self) -> list[str]:
        return sorted(self._agent_models.keys())

    def has_model(self, model: str) -> bool:
        return model in self._model_to_agent

    def agent_for_model(self, model: str) -> str | None:
        return self._model_to_agent.get(model)

    def default_model(self) -> str | None:
        return self._all_models[0] if self._all_models else None

    @classmethod
    def _parse_all_free_models(cls, source: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for match in cls._block_re.finditer(source):
            agent_id = match.group(1)
            models_str = match.group(2)
            models = [m.group(1).strip() for m in cls._model_re.finditer(models_str) if m.group(1).strip()]
            if models:
                result[agent_id] = models
        return result

    @staticmethod
    def _build_model_mapping(agent_models: dict[str, list[str]]) -> dict[str, str]:
        model_agents: dict[str, list[str]] = {}
        for agent_id, models in agent_models.items():
            for model in models:
                model_agents.setdefault(model, []).append(agent_id)
        return {model: random.choice(agents) for model, agents in model_agents.items()}


class FreebuffProvider:
    """Provider implementation using Freebuff / Codebuff local auth tokens."""

    def __init__(self, pool: AccountPool, client: httpx.AsyncClient) -> None:
        self.pool = pool
        self.client = client
        self.registry = FreebuffModelRegistry()
        self._runs: dict[str, dict[str, _ManagedRun]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        await self.registry.refresh(self.client)

    async def stop(self) -> None:
        # Best-effort shutdown of active runs.
        for account_id, agent_runs in list(self._runs.items()):
            acct = self.pool.get_account(account_id)
            if not acct:
                continue
            for run in list(agent_runs.values()):
                try:
                    await self._finish_run(acct, run)
                except Exception:
                    logger.debug("Freebuff finish_run failed for %s/%s", account_id, run.run_id, exc_info=True)

    def has_accounts(self) -> bool:
        return any(a.provider == "freebuff" and a.enabled and a.has_credential for a in self.pool.all_accounts())

    def can_handle_model(self, model: str) -> bool:
        if not self.has_accounts():
            return False
        return self.registry.has_model(model) or model in {"", settings.default_model, "coder-model"}

    def list_models_payload(self) -> list[dict[str, str | int]]:
        now = int(time.time())
        return [
            {"id": model, "object": "model", "created": now, "owned_by": "freebuff"}
            for model in self.registry.models()
        ]

    async def verify_account(self, acct: AccountState) -> dict[str, Any]:
        if acct.provider != "freebuff":
            return {
                "status": "unsupported",
                "account_id": acct.account_id,
                "detail": f"Unsupported provider: {acct.provider}",
                "valid": None,
                "error_type": "unsupported",
            }
        if not acct.access_token:
            return {
                "status": "failed",
                "account_id": acct.account_id,
                "detail": "No auth token loaded",
                "valid": False,
                "error_type": "no_token",
            }

        agent_id = self.registry.agent_ids()[0] if self.registry.agent_ids() else "base2-free"
        acct.last_auth_check_at = time.time()
        run = await self._start_run(acct, agent_id)
        if not run:
            acct.last_auth_failure_at = time.time()
            err = acct.last_error or acct.last_auth_error or "Freebuff verification failed"
            return {
                "status": "failed",
                "account_id": acct.account_id,
                "detail": err,
                "valid": False,
                "error_type": "permission" if acct.health_status.value == "auth_error" else "endpoint",
            }
        try:
            acct.clear_auth_error()
            acct.last_auth_success_at = time.time()
            acct.update_health()
            return {
                "status": "ok",
                "account_id": acct.account_id,
                "detail": "Freebuff token is valid",
                "valid": True,
                "error_type": None,
            }
        finally:
            await self._finish_run(acct, run)

    def _lock_for(self, account_id: str) -> asyncio.Lock:
        if account_id not in self._locks:
            self._locks[account_id] = asyncio.Lock()
        return self._locks[account_id]

    async def chat_completions(
        self,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> JSONResponse | StreamingResponse:
        requested_model = request.model or settings.default_model
        if not self.registry.has_model(requested_model):
            default_model = self.registry.default_model()
            if requested_model in {"", settings.default_model, "coder-model"} and default_model:
                requested_model = default_model
            else:
                return JSONResponse(
                    status_code=400,
                    content=make_error_response(
                        f"Unsupported Freebuff model: {requested_model}",
                        "invalid_request_error",
                        "model_not_found",
                    ),
                )

        agent_id = self.registry.agent_for_model(requested_model)
        if not agent_id:
            return JSONResponse(status_code=502, content=make_error_response("No agent mapping available", "api_error"))

        tried_accounts: set[str] = set()
        for _ in range(max(1, len([a for a in self.pool.all_accounts() if a.provider == "freebuff"]))):
            lease = await self._acquire_run(agent_id, tried_accounts)
            if not lease:
                break

            acct = lease.account
            event_logger.proxy_request(
                request_id=request_id,
                model=requested_model,
                account_id=acct.account_id,
                token_count=0,
                is_streaming=request.stream,
            )

            payload = self._inject_upstream_metadata(request, requested_model, lease.run.run_id)
            req = self.client.build_request(
                "POST",
                f"{settings.freebuff_api_base.rstrip('/')}/api/v1/chat/completions",
                json=payload,
                headers=self._build_headers(acct.access_token),
            )
            resp = await self.client.send(req, stream=request.stream)

            if 200 <= resp.status_code < 300:
                acct.record_success()
                self.pool.last_success_account_id = acct.account_id
                event_logger.rr_anchor_update(acct.account_id)
                if request.stream:
                    return self._create_streaming_response(resp, acct.account_id, request_id, lease)
                try:
                    body = resp.json()
                except Exception:
                    raw = await resp.aread()
                    return JSONResponse(
                        status_code=200,
                        content={"id": str(uuid.uuid4()), "object": "chat.completion", "choices": [], "raw": raw.decode("utf-8", errors="ignore")},
                    )
                finally:
                    await resp.aclose()
                    self._release_lease(lease)
                event_logger.proxy_response(request_id, 200, acct.account_id, 0)
                return JSONResponse(content=body)

            error_body = await resp.aread()
            status_code = resp.status_code
            await resp.aclose()

            if _is_run_invalid(status_code, error_body):
                self._release_lease(lease)
                await self._invalidate_run(lease)
                continue

            if status_code == 401:
                self._release_lease(lease)
                acct.mark_auth_error("Freebuff upstream rejected token")
                tried_accounts.add(acct.account_id)
                continue

            if status_code == 429:
                self._release_lease(lease)
                retry_after = _retry_after_duration(resp.headers.get("retry-after"))
                cooldown = retry_after or 120
                acct.mark_rate_limited("Upstream 429", cooldown)
                event_logger.rate_limit_hit(acct.account_id, retry_after, "Freebuff upstream 429")
                tried_accounts.add(acct.account_id)
                continue

            self._release_lease(lease)
            event_logger.proxy_error(request_id, status_code, acct.account_id, error_body.decode("utf-8", errors="ignore")[:200])
            return _openai_error_from_upstream(status_code, error_body)

        return JSONResponse(
            status_code=502,
            content=make_error_response("No healthy Freebuff account or run available", "api_error"),
        )

    async def _acquire_run(self, agent_id: str, exclude_ids: set[str]) -> _RunLease | None:
        local_excludes = set(exclude_ids)
        account_count = len([a for a in self.pool.all_accounts() if a.provider == "freebuff"])
        for _ in range(max(1, account_count)):
            acct = self.pool.select_account(exclude_ids=local_excludes, provider="freebuff")
            if not acct:
                return None
            local_excludes.add(acct.account_id)
            lock = self._lock_for(acct.account_id)
            async with lock:
                run = self._runs.setdefault(acct.account_id, {}).get(agent_id)
                if not run or time.time() - run.started_at >= settings.freebuff_rotation_interval_seconds:
                    if run:
                        try:
                            await self._finish_run(acct, run)
                        except Exception:
                            logger.debug("Freebuff rotate finish_run failed for %s", acct.account_id, exc_info=True)
                    run = await self._start_run(acct, agent_id)
                    if not run:
                        continue
                    self._runs.setdefault(acct.account_id, {})[agent_id] = run

                run.inflight += 1
                run.request_count += 1
                return _RunLease(account=acct, run=run)
        return None

    async def _invalidate_run(self, lease: _RunLease) -> None:
        acct = lease.account
        run = lease.run
        current = self._runs.setdefault(acct.account_id, {}).get(run.agent_id)
        if current and current.run_id == run.run_id:
            self._runs[acct.account_id].pop(run.agent_id, None)
        try:
            await self._finish_run(acct, run)
        except Exception:
            logger.debug("Freebuff invalidate finish failed for %s", run.run_id, exc_info=True)

    async def _start_run(self, acct: AccountState, agent_id: str) -> _ManagedRun | None:
        payload = {"action": "START", "agentId": agent_id}
        resp = await self.client.post(
            f"{settings.freebuff_api_base.rstrip('/')}/api/v1/agent-runs",
            json=payload,
            headers=self._build_headers(acct.access_token),
            timeout=30.0,
            follow_redirects=True,
        )
        if resp.status_code == 401:
            acct.mark_auth_error("Freebuff start_run unauthorized")
            return None
        if resp.status_code == 429:
            retry_after = _retry_after_duration(resp.headers.get("retry-after"))
            acct.mark_rate_limited("Freebuff start_run rate limited", retry_after or 120)
            return None
        try:
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            acct.record_error(f"Freebuff start_run failed: {exc}")
            return None
        run_id = str(data.get("runId", "")).strip()
        if not run_id:
            acct.record_error("Freebuff start_run missing runId")
            return None
        return _ManagedRun(run_id=run_id, agent_id=agent_id, started_at=time.time())

    async def _finish_run(self, acct: AccountState, run: _ManagedRun) -> None:
        payload = {
            "action": "FINISH",
            "runId": run.run_id,
            "status": "completed",
            "totalSteps": run.request_count,
            "directCredits": 0,
            "totalCredits": 0,
        }
        try:
            await self.client.post(
                f"{settings.freebuff_api_base.rstrip('/')}/api/v1/agent-runs",
                json=payload,
                headers=self._build_headers(acct.access_token),
                timeout=15.0,
                follow_redirects=True,
            )
        except Exception:
            logger.debug("Freebuff finish_run request failed for %s", run.run_id, exc_info=True)

    def _build_headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "ai-sdk/openai-compatible/1.0.25/codebuff",
        }

    def _inject_upstream_metadata(
        self,
        request: ChatCompletionRequest,
        requested_model: str,
        run_id: str,
    ) -> dict[str, Any]:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = requested_model
        metadata = payload.get("codebuff_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["run_id"] = run_id
        metadata["cost_mode"] = "free"
        metadata["client_id"] = _generate_client_session_id()
        payload["codebuff_metadata"] = metadata
        return payload

    def _create_streaming_response(
        self,
        resp: httpx.Response,
        account_id: str,
        request_id: str,
        lease: _RunLease,
    ) -> StreamingResponse:
        async def generate():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                event_logger.proxy_response(request_id, 200, account_id, 0)
            except Exception as exc:
                logger.error("Freebuff stream error for %s: %s", request_id, exc)
                err_json = json.dumps(
                    {"error": {"message": f"Streaming error: {exc}", "type": "api_error"}}
                )
                yield f"data: {err_json}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            finally:
                self._release_lease(lease)
                await resp.aclose()

        return StreamingResponse(generate(), media_type="text/event-stream")

    def _release_lease(self, lease: _RunLease) -> None:
        if lease.run.inflight > 0:
            lease.run.inflight -= 1
