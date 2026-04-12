"""POST /v1/chat/completions — proxy to DashScope with retry, streaming, multi-account."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import AuthManager
from ..config import settings
from ..event_logger import event_logger
from ..headers import build_headers
from ..message_transform import transform_messages
from ..models import (
    clamp_max_tokens,
    is_auth_error,
    is_quota_error,
    is_validation_error,
    make_error_response,
    resolve_model,
    resolve_thinking_params,
)

router = APIRouter()


async def _handle_regular(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_id: str,
    account_id: str,
    start_time: float,
) -> JSONResponse:
    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    latency_ms = int((time.time() - start_time) * 1000)
    data = resp.json()
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    qwen_id = data.get("id")

    if settings.log_requests:
        event_logger.proxy_response(
            request_id=request_id, status_code=resp.status_code,
            account_id=account_id, latency_ms=latency_ms,
            input_tokens=input_tokens, output_tokens=output_tokens, qwen_id=qwen_id,
        )
    return JSONResponse(content=data)


async def _handle_streaming(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    request_id: str,
    account_id: str,
    start_time: float,
) -> StreamingResponse:
    req = client.build_request("POST", url, json=payload, headers=headers)
    resp = await client.send(req, stream=True)
    resp.raise_for_status()
    latency_ms = int((time.time() - start_time) * 1000)
    if settings.log_requests:
        event_logger.proxy_response(
            request_id=request_id, status_code=resp.status_code,
            account_id=account_id, latency_ms=latency_ms,
        )

    async def generate():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )


def _build_payload(body: dict[str, Any], messages: list[dict], model: str, is_streaming: bool,
                    max_tokens: int, session_id: str, turn: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": is_streaming,
        "max_tokens": max_tokens,
    }
    for field in ("temperature", "top_p", "top_k", "repetition_penalty", "tools", "tool_choice"):
        if field in body:
            payload[field] = body[field]
    if is_streaming and payload.get("tools"):
        tools = list(payload["tools"])
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        payload["tools"] = tools
    thinking = resolve_thinking_params(body)
    if thinking:
        payload.update(thinking)
    payload["metadata"] = {"sessionId": session_id, "promptId": f"{session_id}#0#{turn}"}
    if is_streaming:
        payload["stream_options"] = {"include_usage": True}
    return payload


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> JSONResponse | StreamingResponse:
    from ..main import validate_api_key
    validate_api_key(x_api_key, authorization)

    auth: AuthManager = request.app.state.auth
    client: httpx.AsyncClient = request.app.state.http_client
    pool = request.app.state.pool
    request.app.state.request_count += 1

    body: dict[str, Any] = await request.json()
    is_streaming: bool = body.get("stream", False)
    model = resolve_model(body.get("model", settings.default_model))
    max_tokens = clamp_max_tokens(model, body.get("max_tokens", 32000))

    request_id = str(uuid.uuid4())
    start_time = time.time()
    messages = body.get("messages", [])
    token_count = len(str(messages)) // 4

    messages = transform_messages(messages, model, streaming=is_streaming)

    access_token, account_id = await auth.get_valid_token(client)
    acct = pool.get_account(account_id)
    url = f"{auth.get_api_endpoint(acct)}/chat/completions"

    session_id: str = request.app.state.session_id
    turn: int = request.app.state.request_count
    payload = _build_payload(body, messages, model, is_streaming, max_tokens, session_id, turn)
    headers = build_headers(access_token, streaming=is_streaming)

    if settings.log_requests:
        event_logger.proxy_request(
            request_id=request_id, model=model, account_id=account_id,
            token_count=token_count, is_streaming=is_streaming,
        )

    last_error: Exception | None = None
    last_status: int | None = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            if is_streaming:
                return await _handle_streaming(
                    client, url, payload, headers, request_id, account_id, start_time,
                )
            return await _handle_regular(
                client, url, payload, headers, request_id, account_id, start_time,
            )
        except httpx.HTTPStatusError as exc:
            last_error = exc
            last_status = exc.response.status_code
            status = exc.response.status_code
            error_message = str(exc)

            if is_validation_error(error_message):
                log_warning(f"Validation error {status}: {error_message[:100]}")
                if settings.log_requests:
                    event_logger.proxy_error(
                        request_id=request_id, status_code=status,
                        account_id=account_id, error_message=error_message,
                    )
                return JSONResponse(
                    status_code=400, content=make_error_response(error_message, "validation_error", "invalid_request"),
                )

            if status in (500, 429) and attempt < settings.max_retries:
                log_warning(f"Retry {attempt}/{settings.max_retries} (status {status})")
                await asyncio.sleep(settings.retry_delay_s * attempt)
                continue

            if is_auth_error(status, error_message):
                try:
                    log_info(f"Auth error {status} on {account_id}, refreshing...")
                    if acct:
                        ok = await auth.refresh_token(acct, client)
                        if ok:
                            headers = build_headers(acct.access_token, streaming=is_streaming)
                            if is_streaming:
                                return await _handle_streaming(
                                    client, url, payload, headers, request_id, account_id, start_time,
                                )
                            return await _handle_regular(
                                client, url, payload, headers, request_id, account_id, start_time,
                            )
                except Exception as refresh_err:
                    log_error(f"Token refresh on {account_id} failed: {refresh_err}")
                    if settings.log_requests:
                        event_logger.proxy_error(
                            request_id=request_id, status_code=401,
                            account_id=account_id, error_message=str(refresh_err),
                        )
                    if acct:
                        acct.record_error(str(refresh_err))
                    return JSONResponse(
                        status_code=401,
                        content=make_error_response("Authentication failed. Please re-authenticate with Qwen CLI.", "authentication_error", "invalid_token"),
                    )

                # Auth refresh didn't help — try next account
                if acct:
                    acct.record_error(f"Auth error {status}")
                new_token, new_account_id = await _try_next_account(auth, client)
                if new_token:
                    headers = build_headers(new_token, streaming=is_streaming)
                    account_id = new_account_id
                    new_acct = pool.get_account(account_id)
                    url = f"{auth.get_api_endpoint(new_acct)}/chat/completions"
                    if is_streaming:
                        return await _handle_streaming(
                            client, url, payload, headers, request_id, account_id, start_time,
                        )
                    return await _handle_regular(
                        client, url, payload, headers, request_id, account_id, start_time,
                    )
            break

        except Exception as exc:
            last_error = exc
            error_message = str(exc)
            if is_validation_error(error_message):
                log_warning(f"Validation error: {error_message[:100]}")
                if settings.log_requests:
                    event_logger.proxy_error(
                        request_id=request_id, status_code=400,
                        account_id=account_id, error_message=error_message,
                    )
                return JSONResponse(
                    status_code=400, content=make_error_response(error_message, "validation_error", "invalid_request"),
                )
            if attempt < settings.max_retries:
                log_warning(f"Retry {attempt}/{settings.max_retries} (error: {error_message[:50]})")
                await asyncio.sleep(settings.retry_delay_s * attempt)
                continue
            break

    # Final error response
    error_msg = str(last_error) if last_error else "Unknown error"
    if is_validation_error(error_msg):
        if settings.log_requests:
            event_logger.proxy_error(request_id=request_id, status_code=400, account_id=account_id, error_message=error_msg)
        return JSONResponse(status_code=400, content=make_error_response(error_msg, "validation_error", "invalid_request"))
    if is_quota_error(last_status, error_msg):
        if settings.log_requests:
            event_logger.proxy_error(request_id=request_id, status_code=429, account_id=account_id, error_message=error_msg)
        return JSONResponse(status_code=429, content=make_error_response("Rate limit or quota exceeded.", "rate_limit_exceeded", "rate_limit_exceeded"))
    if is_auth_error(last_status, error_msg):
        if settings.log_requests:
            event_logger.proxy_error(request_id=request_id, status_code=401, account_id=account_id, error_message=error_msg)
        return JSONResponse(status_code=401, content=make_error_response("Authentication failed.", "authentication_error", "invalid_token"))
    if settings.log_requests:
        event_logger.proxy_error(request_id=request_id, status_code=500, account_id=account_id, error_message=error_msg)
    return JSONResponse(status_code=500, content=make_error_response(error_msg, "api_error"))


async def _try_next_account(auth: AuthManager, client: httpx.AsyncClient) -> tuple[str, str]:
    """Try refreshing and using the next available account. Returns (token, account_id) or empty."""
    pool = auth.pool
    for _ in range(len(pool.accounts)):
        acct = pool.select_account()
        if not acct:
            break
        if acct.token_valid:
            return acct.access_token, acct.account_id
        pool.check_mtime_and_reload(acct.account_id)
        try:
            ok = await auth.refresh_token(acct, client)
            if ok:
                return acct.access_token, acct.account_id
        except Exception:
            pass
    return "", ""


def log_info(msg: str) -> None:
    import logging
    logging.getLogger("gx_qwen2api").info(msg)


def log_warning(msg: str) -> None:
    import logging
    logging.getLogger("gx_qwen2api").warning(msg)


def log_error(msg: str) -> None:
    import logging
    logging.getLogger("gx_qwen2api").error(msg)
