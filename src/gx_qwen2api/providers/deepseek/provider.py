"""DeepSeek provider implementation with account pool and OpenAI compatibility.

Aligned with ds-free-api protocol:
- Each request creates a temporary chat session, deleted after completion.
- Chat uses edit_message with per-request PoW (X-Ds-Pow-Response).
- Payload uses prompt, model_type, message_id, chat_session_id.
- SSE responses are parsed through a patch state machine and converted
  to OpenAI-compatible SSE format.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from ...account_pool import AccountPool, AccountState
from ...event_logger import event_logger
from ...models import ChatCompletionRequest, make_error_response
from .auth import (
    completion,
    create_chat_session,
    create_pow_challenge,
    delete_chat_session,
    edit_message,
    refresh_token_if_needed,
    stop_stream,
    build_pow_header,
)
from .models import (
    EXPOSED_MODELS,
    DeepseekAccount,
    ds_model_type,
    is_supported_model,
    openai_model_id,
)
from .pow import get_solver, solve_pow_challenge
from .tool_calls import (
    ToolCallStreamSieve,
    build_prompt_with_tools,
    parse_tool_calls_from_text,
)

logger = logging.getLogger("gx_qwen2api.deepseek.provider")


# ======================================================================
# DeepSeek Patch State Machine (ported from ds-free-api state.rs)
# ======================================================================

_FRAG_THINK = "THINK"
_FRAG_RESPONSE = "RESPONSE"


class DsFrame:
    """Incremental frame parsed from DeepSeek SSE patch stream."""

    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: Any = None) -> None:
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:
        return f"DsFrame({self.kind!r}, {self.value!r})"


class _Fragment:
    __slots__ = ("ty", "content")

    def __init__(self, ty: str, content: str) -> None:
        self.ty = ty
        self.content = content


class DsPatchState:
    """State machine that consumes DeepSeek SSE events and produces DsFrames.

    Mirrors the Rust DsState from ds-free-api openai_adapter/response/state.rs.
    Handles:
    - event: ready → DsFrame("role")
    - event: finish → DsFrame("finish")
    - JSON patch operations: {p, o, v} with paths like
      response/fragments/-1/content, response/status, etc.
    - Snapshot: bare {v: {response: {fragments: [...]}}}
    """

    def __init__(self) -> None:
        self.current_path: str | None = None
        self.fragments: list[_Fragment] = []
        self.status: str | None = None
        self.accumulated_token_usage: int | None = None

    def apply_event(self, event_type: str | None, data: str) -> list[DsFrame]:
        frames: list[DsFrame] = []

        if event_type == "ready":
            frames.append(DsFrame("role"))
        if event_type == "finish":
            frames.append(DsFrame("finish"))

        if data:
            try:
                val = json.loads(data)
            except Exception:
                return frames
            if isinstance(val, dict):
                frames.extend(self._apply_patch_value(val))

        return frames

    def _apply_patch_value(self, val: dict[str, Any]) -> list[DsFrame]:
        frames: list[DsFrame] = []
        has_p = "p" in val
        op = val.get("o")
        if isinstance(op, str):
            pass
        else:
            op = None

        if has_p:
            p = val.get("p")
            if isinstance(p, str):
                self.current_path = p

        v = val.get("v")
        if v is None:
            return frames

        if has_p or op is not None:
            if self.current_path is not None:
                if self.current_path == "response" and op == "BATCH":
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                frames.extend(self._apply_patch_value(item))
                else:
                    frames.extend(self._apply_path(self.current_path, op, v))
        elif self.current_path is not None:
            frames.extend(self._apply_path(self.current_path, "APPEND", v))
        else:
            if isinstance(v, dict):
                response = v.get("response")
                if isinstance(response, dict):
                    frag_arr = response.get("fragments")
                    if isinstance(frag_arr, list):
                        self.fragments.clear()
                        for frag in frag_arr:
                            if isinstance(frag, dict):
                                ty = frag.get("type", "")
                                content = frag.get("content", "")
                                if isinstance(ty, str) and isinstance(content, str):
                                    self.fragments.append(_Fragment(ty, content))
                                    if content:
                                        if ty == _FRAG_THINK:
                                            frames.append(DsFrame("think_delta", content))
                                        elif ty == _FRAG_RESPONSE:
                                            frames.append(DsFrame("content_delta", content))

        return frames

    def _apply_path(self, path: str, op: str | None, val: Any) -> list[DsFrame]:
        frames: list[DsFrame] = []

        if path in ("response/status", "response/quasi_status", "quasi_status"):
            if isinstance(val, str):
                self.status = val
                frames.append(DsFrame("status", val))

        elif path in ("response/accumulated_token_usage", "accumulated_token_usage"):
            if isinstance(val, (int, float)):
                u = int(val)
                self.accumulated_token_usage = u
                frames.append(DsFrame("usage", u))

        elif path == "response/fragments/-1/content":
            if isinstance(val, str) and self.fragments:
                frag = self.fragments[-1]
                if frag.ty == _FRAG_THINK:
                    frag.content += val
                    frames.append(DsFrame("think_delta", val))
                elif frag.ty == _FRAG_RESPONSE:
                    frag.content += val
                    frames.append(DsFrame("content_delta", val))

        elif path == "response/fragments" and op in ("APPEND", "SET"):
            if isinstance(val, list):
                if op == "SET":
                    self.fragments.clear()
                for item in val:
                    if isinstance(item, dict):
                        ty = item.get("type", "")
                        content = item.get("content", "")
                        if isinstance(ty, str) and isinstance(content, str):
                            self.fragments.append(_Fragment(ty, content))
                            if content:
                                if ty == _FRAG_THINK:
                                    frames.append(DsFrame("think_delta", content))
                                elif ty == _FRAG_RESPONSE:
                                    frames.append(DsFrame("content_delta", content))

        return frames


# ======================================================================
# SSE Parser (ported from ds-free-api sse_parser.rs)
# ======================================================================

class SseEvent:
    __slots__ = ("event", "data")

    def __init__(self, event: str | None, data: str) -> None:
        self.event = event
        self.data = data


def parse_sse_events(buf: str) -> list[SseEvent]:
    """Parse a text buffer into a list of SSE events.

    Each event is delimited by \\n\\n. Within an event block:
    - Lines starting with "event:" set the event type.
    - Lines starting with "data:" append to the data field.
    """
    events: list[SseEvent] = []
    blocks = buf.split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        event_type: str | None = None
        data_parts: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                part = line[5:]
                if part.startswith(" "):
                    part = part[1:]
                data_parts.append(part)
        if data_parts:
            events.append(SseEvent(event_type, "\n".join(data_parts)))
    return events


# ======================================================================
# OpenAI Chunk Converter (ported from ds-free-api converter.rs)
# ======================================================================

_chatcmpl_counter = 0


def _next_chatcmpl_id() -> str:
    global _chatcmpl_counter
    _chatcmpl_counter += 1
    return f"chatcmpl-{_chatcmpl_counter:016x}"


def _make_openai_chunk(
    model: str,
    request_id: str,
    role: str | None = None,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    tool_call_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_call_delta is not None:
        delta["tool_calls"] = [tool_call_delta]

    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _chunk_has_tool_calls(chunk: dict[str, Any]) -> bool:
    """Check if an OpenAI chunk dict contains tool_calls in its delta."""
    choices = chunk.get("choices", [])
    if choices:
        delta = choices[0].get("delta", {})
        if delta.get("tool_calls"):
            return True
    return False


def ds_frames_to_openai_chunks(
    frames: list[DsFrame],
    model: str,
    request_id: str,
    include_usage: bool = True,
) -> list[dict[str, Any]]:
    """Convert a list of DsFrames into OpenAI ChatCompletionChunk dicts."""
    chunks: list[dict[str, Any]] = []
    finished = False
    usage_value: int | None = None

    for frame in frames:
        if finished and frame.kind not in ("usage",):
            continue

        if frame.kind == "role":
            chunks.append(_make_openai_chunk(model, request_id, role="assistant"))

        elif frame.kind == "think_delta":
            chunks.append(_make_openai_chunk(model, request_id, reasoning_content=frame.value))

        elif frame.kind == "content_delta":
            chunks.append(_make_openai_chunk(model, request_id, content=frame.value))

        elif frame.kind == "status":
            if frame.value == "FINISHED" and not finished:
                finished = True
                chunks.append(_make_openai_chunk(model, request_id, finish_reason="stop"))

        elif frame.kind == "finish":
            if not finished:
                finished = True
                chunks.append(_make_openai_chunk(model, request_id, finish_reason="stop"))

        elif frame.kind == "usage":
            usage_value = frame.value
            if finished and include_usage:
                chunks.append(_make_openai_chunk(
                    model, request_id,
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": usage_value,
                        "total_tokens": usage_value,
                    },
                ))

    if finished and include_usage and usage_value is not None:
        has_usage_chunk = any(c.get("usage") for c in chunks)
        if not has_usage_chunk:
            chunks.append(_make_openai_chunk(
                model, request_id,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": usage_value,
                    "total_tokens": usage_value,
                },
            ))

    return chunks


def _extract_deepseek_text_fallback(raw_text: str) -> tuple[str, str]:
    """Best-effort extraction for DeepSeek event shapes not covered by the patch state."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    def visit(value: Any, current_type: str | None = None) -> None:
        if isinstance(value, dict):
            frag_type = value.get("type")
            if isinstance(frag_type, str):
                current_type = frag_type

            text = value.get("content")
            if isinstance(text, str) and text:
                if current_type == _FRAG_THINK:
                    reasoning_parts.append(text)
                elif current_type == _FRAG_RESPONSE or current_type is None:
                    content_parts.append(text)

            for child in value.values():
                visit(child, current_type)
        elif isinstance(value, list):
            for item in value:
                visit(item, current_type)

    for event in parse_sse_events(raw_text):
        if not event.data or event.data.strip() == "[DONE]":
            continue
        try:
            payload = json.loads(event.data)
        except Exception:
            continue
        visit(payload)

    return "".join(content_parts), "".join(reasoning_parts)


# ======================================================================
# Session / Runtime dataclasses
# ======================================================================

@dataclass
class _AccountRuntime:
    account: DeepseekAccount
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ======================================================================
# Provider
# ======================================================================

class DeepseekProvider:
    """DeepSeek Web provider with multi-account scheduling and OpenAI API compatibility."""

    def __init__(self, pool: AccountPool, client: httpx.AsyncClient) -> None:
        self.pool = pool
        self.client = client
        self._runtimes: dict[str, _AccountRuntime] = {}
        self._runtime_lock = asyncio.Lock()
        self._wasm_initialized = False

    async def start(self) -> None:
        async with self._runtime_lock:
            for acct_state in self.pool.all_accounts():
                if acct_state.provider == "deepseek":
                    self._ensure_runtime(acct_state)

        solver = await get_solver(self.client)
        if solver:
            self._wasm_initialized = True
            logger.info("DeepSeek PoW WASM solver ready")
        else:
            logger.warning("DeepSeek PoW WASM solver not available, will use brute-force fallback")

    async def stop(self) -> None:
        async with self._runtime_lock:
            self._runtimes.clear()

    def remove_account_runtime(self, account_id: str) -> None:
        runtime = self._runtimes.pop(account_id, None)
        if runtime:
            logger.info("DeepSeek runtime removed for %s", account_id)

    def has_accounts(self) -> bool:
        return any(
            a.provider == "deepseek" and a.enabled and a.has_credential
            for a in self.pool.all_accounts()
        )

    def can_handle_model(self, model: str) -> bool:
        if not self.has_accounts():
            return False
        return is_supported_model(model)

    def list_models_payload(self) -> list[dict[str, str | int]]:
        now = int(time.time())
        return [
            {"id": model, "object": "model", "created": now, "owned_by": "deepseek"}
            for model in EXPOSED_MODELS
        ]

    async def verify_account(self, acct: AccountState) -> dict[str, Any]:
        if acct.provider != "deepseek":
            return {
                "status": "unsupported",
                "account_id": acct.account_id,
                "detail": f"Unsupported provider: {acct.provider}",
                "valid": None,
                "error_type": "unsupported",
            }
        if not acct.access_token and not ((acct.email or acct.mobile) and acct.password):
            return {
                "status": "failed",
                "account_id": acct.account_id,
                "detail": "No credentials available",
                "valid": False,
                "error_type": "no_token",
            }

        acct.last_auth_check_at = time.time()
        runtime = await self._get_runtime(acct.account_id)
        if not runtime:
            return {
                "status": "failed",
                "account_id": acct.account_id,
                "detail": "Runtime not found",
                "valid": False,
                "error_type": "runtime_error",
            }

        ok = await refresh_token_if_needed(self.client, runtime.account)
        if ok:
            acct.clear_auth_error()
            acct.last_auth_success_at = time.time()
            acct.update_health()
            acct.access_token = runtime.account.access_token
            return {
                "status": "ok",
                "account_id": acct.account_id,
                "detail": "DeepSeek token is valid",
                "valid": True,
                "error_type": None,
            }
        else:
            acct.last_auth_failure_at = time.time()
            err = runtime.account.last_error or "DeepSeek verification failed"
            return {
                "status": "failed",
                "account_id": acct.account_id,
                "detail": err,
                "valid": False,
                "error_type": "auth_error",
            }

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    async def chat_completions(
        self,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> JSONResponse | StreamingResponse:
        requested_model = request.model or ""
        if not is_supported_model(requested_model):
            return JSONResponse(
                status_code=400,
                content=make_error_response(
                    f"Unsupported DeepSeek model: {requested_model}",
                    "invalid_request_error",
                    "model_not_found",
                    ),
                )

        ds_type = ds_model_type(requested_model)
        if not ds_type:
            return JSONResponse(
                status_code=400,
                content=make_error_response(
                    f"Model mapping not found: {requested_model}",
                    "invalid_request_error",
                ),
            )

        thinking_enabled = request.enable_thinking if request.enable_thinking is not None else (ds_type == "expert")
        search_enabled = False

        tools = request.tools
        tool_names = [t.get("function", {}).get("name", "") for t in (tools or []) if t.get("function", {}).get("name")] if tools else None
        prompt = build_prompt_with_tools(request.messages, tools)

        tried_accounts: set[str] = set()
        auth_retried_accounts: set[str] = set()
        max_attempts = max(1, len([a for a in self.pool.all_accounts() if a.provider == "deepseek"]))

        for _ in range(max_attempts):
            lease = await self._acquire_account(tried_accounts, ds_type)
            if not lease:
                break

            runtime = lease["runtime"]
            acct_state = lease["account_state"]
            session_id = lease["session_id"]

            event_logger.proxy_request(
                request_id=request_id,
                model=requested_model,
                account_id=acct_state.account_id,
                token_count=0,
                is_streaming=request.stream,
            )

            pow_header = await self._compute_pow(runtime.account.access_token)
            if not pow_header:
                await self._delete_session(runtime, session_id)
                acct_state.record_error("PoW computation failed")
                tried_accounts.add(acct_state.account_id)
                continue

            payload = {
                "chat_session_id": session_id,
                "parent_message_id": None,
                "prompt": prompt,
                "ref_file_ids": [],
                "search_enabled": search_enabled,
                "thinking_enabled": thinking_enabled,
                "model_type": ds_type,
                "preempt": False,
            }

            try:
                resp = await completion(
                    self.client,
                    runtime.account.access_token,
                    pow_header,
                    payload,
                )
            except Exception as exc:
                await self._delete_session(runtime, session_id)
                logger.warning("DeepSeek completion request send error for %s: %s", acct_state.account_id, exc)
                acct_state.record_error(f"Request send failed: {exc}")
                tried_accounts.add(acct_state.account_id)
                continue

            if 200 <= resp.status_code < 300:
                acct_state.record_success()
                self.pool.last_success_account_id = acct_state.account_id
                event_logger.rr_anchor_update(acct_state.account_id)

                if request.stream:
                    return await self._create_streaming_response(
                        resp, acct_state, request_id, runtime, session_id, requested_model,
                        tools, tool_names,
                    )

                try:
                    return await self._create_nonstream_response(
                        resp, acct_state, request_id, runtime, session_id, requested_model,
                        tools, tool_names,
                    )
                finally:
                    await self._delete_session(runtime, session_id)

            error_body = await resp.aread()
            status_code = resp.status_code
            await resp.aclose()
            await self._delete_session(runtime, session_id)

            if status_code == 401:
                if acct_state.account_id in auth_retried_accounts:
                    acct_state.mark_auth_error("DeepSeek upstream rejected token after forced re-login")
                    tried_accounts.add(acct_state.account_id)
                    continue

                auth_retried_accounts.add(acct_state.account_id)
                refreshed = await refresh_token_if_needed(self.client, runtime.account, force=True)
                acct_state.access_token = runtime.account.access_token
                if refreshed and runtime.account.access_token:
                    acct_state.clear_auth_error()
                    logger.info("DeepSeek forced re-login succeeded for %s; retrying request once", acct_state.account_id)
                    continue
                acct_state.mark_auth_error("DeepSeek upstream rejected token and refresh failed")
                tried_accounts.add(acct_state.account_id)
                continue

            if status_code == 429:
                retry_after_hdr = resp.headers.get("retry-after")
                cooldown = 120
                if retry_after_hdr:
                    try:
                        cooldown = max(0, int(retry_after_hdr.strip()))
                    except (TypeError, ValueError):
                        pass
                acct_state.mark_rate_limited("Upstream 429", cooldown)
                event_logger.rate_limit_hit(acct_state.account_id, cooldown, "DeepSeek upstream 429")
                tried_accounts.add(acct_state.account_id)
                continue

            event_logger.proxy_error(
                request_id, status_code, acct_state.account_id,
                error_body.decode("utf-8", errors="ignore")[:200]
            )
            return self._openai_error_from_upstream(status_code, error_body)

        return JSONResponse(
            status_code=502,
            content=make_error_response("No healthy DeepSeek account or session available", "api_error"),
        )

    # ------------------------------------------------------------------
    # PoW
    # ------------------------------------------------------------------

    async def _compute_pow(self, token: str) -> str | None:
        challenge = await create_pow_challenge(self.client, token)
        if not challenge:
            logger.error("DeepSeek PoW challenge request failed")
            return None

        if not self._wasm_initialized:
            solver = await get_solver(self.client)
            if solver:
                self._wasm_initialized = True

        answer = solve_pow_challenge(challenge)
        if answer is None:
            logger.error("DeepSeek PoW solve failed")
            return None
        return build_pow_header(challenge, answer)

    # ------------------------------------------------------------------
    # Account / session acquisition
    # ------------------------------------------------------------------

    async def _acquire_account(self, exclude_ids: set[str], model_type: str) -> dict[str, Any] | None:
        local_excludes = set(exclude_ids)
        account_count = len([a for a in self.pool.all_accounts() if a.provider == "deepseek"])

        for _ in range(max(1, account_count)):
            acct_state = self.pool.select_account(exclude_ids=local_excludes, provider="deepseek")
            if not acct_state:
                return None
            local_excludes.add(acct_state.account_id)

            runtime = await self._get_runtime(acct_state.account_id)
            if not runtime:
                continue

            async with runtime.lock:
                ok = await refresh_token_if_needed(self.client, runtime.account)
                if not ok:
                    acct_state.mark_auth_error("DeepSeek token refresh/re-login failed")
                    continue
                acct_state.clear_auth_error()
                acct_state.access_token = runtime.account.access_token

                session_id = await create_chat_session(self.client, runtime.account.access_token)
                if not session_id:
                    acct_state.record_error("DeepSeek session creation failed")
                    continue

                return {
                    "runtime": runtime,
                    "account_state": acct_state,
                    "session_id": session_id,
                }

        return None

    async def _delete_session(self, runtime: _AccountRuntime, session_id: str) -> None:
        try:
            await delete_chat_session(self.client, runtime.account.access_token, session_id)
        except Exception:
            logger.debug("DeepSeek session delete failed for %s", session_id, exc_info=True)

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def _ensure_runtime(self, acct_state: AccountState) -> _AccountRuntime:
        account_id = acct_state.account_id
        if account_id not in self._runtimes:
            ds_account = DeepseekAccount(
                account_id=account_id,
                email=acct_state.email or "",
                password=acct_state.password or "",
                enabled=acct_state.enabled,
                access_token=acct_state.access_token or "",
                refresh_token=acct_state._raw_creds.get("refresh_token", "") if isinstance(acct_state._raw_creds, dict) else "",
                mobile=acct_state._raw_creds.get("mobile", "") if isinstance(acct_state._raw_creds, dict) else "",
                area_code=acct_state._raw_creds.get("area_code", "") if isinstance(acct_state._raw_creds, dict) else "",
                device_id=acct_state.device_id or (acct_state._raw_creds.get("device_id", "") if isinstance(acct_state._raw_creds, dict) else ""),
            )
            self._runtimes[account_id] = _AccountRuntime(account=ds_account)
        return self._runtimes[account_id]

    async def _get_runtime(self, account_id: str) -> _AccountRuntime | None:
        acct_state = self.pool.get_account(account_id)
        if not acct_state or acct_state.provider != "deepseek":
            return None
        async with self._runtime_lock:
            return self._ensure_runtime(acct_state)

    def sync_account_to_pool(self, account_id: str) -> None:
        runtime = self._runtimes.get(account_id)
        if not runtime:
            return
        acct_state = self.pool.get_account(account_id)
        if acct_state:
            acct_state.access_token = runtime.account.access_token
            if isinstance(acct_state._raw_creds, dict):
                acct_state._raw_creds["access_token"] = runtime.account.access_token
                acct_state._raw_creds["refresh_token"] = runtime.account.refresh_token

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        return build_prompt_with_tools(messages, tools)

    def _process_stream_frames_with_sieve(
        self,
        frames: list[DsFrame],
        model: str,
        request_id: str,
        sieve: ToolCallStreamSieve,
    ) -> list[dict[str, Any]]:
        """Feed DsFrame content through the ToolCallStreamSieve and convert to OpenAI chunks."""
        chunks: list[dict[str, Any]] = []
        finished = False
        usage_value: int | None = None

        for frame in frames:
            if finished and frame.kind not in ("usage",):
                continue

            if frame.kind == "role":
                chunks.append(_make_openai_chunk(model, request_id, role="assistant"))

            elif frame.kind == "think_delta":
                feed = sieve.feed(frame.value)
                if feed.content:
                    chunks.append(_make_openai_chunk(model, request_id, reasoning_content=feed.content))
                for delta in feed.tool_call_chunks:
                    chunks.append(_make_openai_chunk(model, request_id, tool_call_delta=delta))

            elif frame.kind == "content_delta":
                feed = sieve.feed(frame.value)
                if feed.content:
                    chunks.append(_make_openai_chunk(model, request_id, content=feed.content))
                for delta in feed.tool_call_chunks:
                    chunks.append(_make_openai_chunk(model, request_id, tool_call_delta=delta))

            elif frame.kind == "status":
                if frame.value == "FINISHED" and not finished:
                    finished = True

            elif frame.kind == "finish":
                if not finished:
                    finished = True

            elif frame.kind == "usage":
                usage_value = frame.value

        if usage_value is not None:
            chunks.append(_make_openai_chunk(
                model, request_id,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": usage_value,
                    "total_tokens": usage_value,
                },
            ))

        return chunks

    # ------------------------------------------------------------------
    # Streaming response: DeepSeek SSE → OpenAI SSE
    # ------------------------------------------------------------------

    async def _create_streaming_response(
        self,
        resp: httpx.Response,
        acct_state: AccountState,
        request_id: str,
        runtime: _AccountRuntime,
        session_id: str,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_names: list[str] | None = None,
    ) -> StreamingResponse:
        state = DsPatchState()
        text_buf = ""
        stop_id = 0
        ready_parsed = False
        pre_ready_frames: list[DsFrame] = []
        sieve = ToolCallStreamSieve(tool_names) if tools else None
        finish_reason = "stop"
        tool_calls_emitted = False

        async def generate() -> AsyncGenerator[bytes, None]:
            nonlocal text_buf, stop_id, ready_parsed, pre_ready_frames, finish_reason, tool_calls_emitted

            try:
                async for chunk in resp.aiter_bytes():
                    text_buf += chunk.decode("utf-8", errors="replace")

                    while "\n\n" in text_buf:
                        event_end = text_buf.index("\n\n")
                        block = text_buf[:event_end]
                        text_buf = text_buf[event_end + 2:]

                        sse_events = parse_sse_events(block)
                        for sse_evt in sse_events:
                            frames = state.apply_event(sse_evt.event, sse_evt.data)

                            if not ready_parsed:
                                found_role = False
                                for f in frames:
                                    if f.kind == "role":
                                        ready_parsed = True
                                        found_role = True
                                    elif f.kind == "status" and "rate_limit" in str(f.value):
                                        err_json = json.dumps({
                                            "error": {
                                                "message": "DeepSeek rate limit",
                                                "type": "rate_limit_error",
                                            }
                                        })
                                        yield f"data: {err_json}\n\n".encode("utf-8")
                                        yield b"data: [DONE]\n\n"
                                        return
                                    elif f.kind not in ("role",):
                                        pre_ready_frames.append(f)

                                if sse_evt.event == "ready":
                                    try:
                                        ready_data = json.loads(sse_evt.data)
                                        stop_id = ready_data.get("response_message_id", 0)
                                    except Exception:
                                        pass

                                if found_role:
                                    openai_chunks = ds_frames_to_openai_chunks(
                                        [f for f in frames if f.kind == "role"], model, request_id,
                                    )
                                    for chunk_dict in openai_chunks:
                                        line = f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                                        yield line.encode("utf-8")

                                    if pre_ready_frames:
                                        if sieve is not None:
                                            for chunk_dict in self._process_stream_frames_with_sieve(pre_ready_frames, model, request_id, sieve):
                                                if _chunk_has_tool_calls(chunk_dict):
                                                    tool_calls_emitted = True
                                                line = f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                                                yield line.encode("utf-8")
                                        else:
                                            openai_chunks = ds_frames_to_openai_chunks(pre_ready_frames, model, request_id)
                                            for chunk_dict in openai_chunks:
                                                line = f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                                                yield line.encode("utf-8")
                                        pre_ready_frames = []

                                continue

                            if sieve is not None:
                                for chunk_dict in self._process_stream_frames_with_sieve(frames, model, request_id, sieve):
                                    if _chunk_has_tool_calls(chunk_dict):
                                        tool_calls_emitted = True
                                    line = f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                                    yield line.encode("utf-8")
                            else:
                                openai_chunks = ds_frames_to_openai_chunks(frames, model, request_id)
                                for chunk_dict in openai_chunks:
                                    line = f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                                    yield line.encode("utf-8")

                if text_buf.strip():
                    sse_events = parse_sse_events(text_buf)
                    for sse_evt in sse_events:
                        frames = state.apply_event(sse_evt.event, sse_evt.data)
                        if sieve is not None:
                            for chunk_dict in self._process_stream_frames_with_sieve(frames, model, request_id, sieve):
                                if _chunk_has_tool_calls(chunk_dict):
                                    tool_calls_emitted = True
                                line = f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                                yield line.encode("utf-8")
                        else:
                            openai_chunks = ds_frames_to_openai_chunks(frames, model, request_id)
                            for chunk_dict in openai_chunks:
                                line = f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                                yield line.encode("utf-8")

                if sieve is not None:
                    flush_result = sieve.flush()
                    if flush_result.tool_call_chunks:
                        tool_calls_emitted = True
                        for delta in flush_result.tool_call_chunks:
                            chunk = _make_openai_chunk(model, request_id, tool_call_delta=delta)
                            line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            yield line.encode("utf-8")
                    if flush_result.content:
                        chunk = _make_openai_chunk(model, request_id, content=flush_result.content)
                        line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        yield line.encode("utf-8")

                    if tool_calls_emitted:
                        finish_reason = "tool_calls"

                    final_chunk = _make_openai_chunk(model, request_id, finish_reason=finish_reason)
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode("utf-8")

                yield b"data: [DONE]\n\n"

                event_logger.proxy_response(request_id, 200, acct_state.account_id, 0)

            except Exception as exc:
                logger.error("DeepSeek stream error for %s: %s", request_id, exc)
                if stop_id:
                    try:
                        await stop_stream(self.client, runtime.account.access_token, session_id, stop_id)
                    except Exception:
                        pass
                err_json = json.dumps({"error": {"message": f"Streaming error: {exc}", "type": "api_error"}})
                yield f"data: {err_json}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            finally:
                await self._delete_session(runtime, session_id)
                await resp.aclose()

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Non-streaming response: aggregate DeepSeek SSE → OpenAI JSON
    # ------------------------------------------------------------------

    async def _create_nonstream_response(
        self,
        resp: httpx.Response,
        acct_state: AccountState,
        request_id: str,
        runtime: _AccountRuntime,
        session_id: str,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_names: list[str] | None = None,
    ) -> JSONResponse:
        state = DsPatchState()
        text_buf = ""

        try:
            async for chunk in resp.aiter_bytes():
                text_buf += chunk.decode("utf-8", errors="replace")
        finally:
            await resp.aclose()

        if text_buf.strip():
            sse_events = parse_sse_events(text_buf)
            for sse_evt in sse_events:
                state.apply_event(sse_evt.event, sse_evt.data)

        content = ""
        reasoning = ""
        usage_tokens = 0
        for frag in state.fragments:
            if frag.ty == _FRAG_RESPONSE and frag.content:
                content += frag.content
            elif frag.ty == _FRAG_THINK and frag.content:
                reasoning += frag.content

        if state.accumulated_token_usage is not None:
            usage_tokens = state.accumulated_token_usage

        if not content and not reasoning and text_buf.strip():
            content, reasoning = _extract_deepseek_text_fallback(text_buf)
            logger.warning(
                "DeepSeek non-stream response parsed empty; fallback content_len=%d reasoning_len=%d raw_sample=%r",
                len(content),
                len(reasoning),
                text_buf[:2000],
            )

        finish_reason = "stop"
        tool_calls_result: list[dict[str, Any]] | None = None

        if tools and (content or reasoning):
            tool_calls_result, clean_content, clean_reasoning = parse_tool_calls_from_text(
                content, reasoning, tool_names,
            )
            if tool_calls_result:
                finish_reason = "tool_calls"
                content = clean_content
                reasoning = clean_reasoning

        message: dict[str, Any] = {"role": "assistant"}
        if finish_reason == "tool_calls":
            message["content"] = None
            message["tool_calls"] = tool_calls_result
        else:
            message["content"] = content or None
        if reasoning:
            message["reasoning_content"] = reasoning

        result: dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage_tokens:
            result["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": usage_tokens,
                "total_tokens": usage_tokens,
            }

        event_logger.proxy_response(request_id, 200, acct_state.account_id, 0)
        return JSONResponse(content=result)

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    def _openai_error_from_upstream(self, status_code: int, body: bytes) -> JSONResponse:
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
