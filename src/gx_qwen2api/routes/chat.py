"""POST /v1/chat/completions — multi-provider proxy (Qwen, Freebuff, DeepSeek, NVIDIA)."""

from __future__ import annotations

import time
import logging
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import AuthManager
from ..call_logger import call_logger as _call_logger
from ..config import settings
from ..dispatcher import Dispatcher
from ..event_logger import event_logger
from ..message_transform import transform_messages
from ..models import (
    ChatCompletionRequest,
    clamp_max_tokens,
    make_error_response,
    resolve_model,
    resolve_thinking_params,
)

logger = logging.getLogger("gx2api.chat")

router = APIRouter()


# Helper functions for internal use


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
    request.app.state.request_count += 1

    body: dict[str, Any] = await request.json()
    is_streaming: bool = body.get("stream", False)
    model = resolve_model(body.get("model", settings.default_model))
    max_tokens = clamp_max_tokens(model, body.get("max_tokens", 32000))
    freebuff = request.app.state.freebuff

    request_id = str(uuid.uuid4())
    t0 = time.time()

    if freebuff.can_handle_model(model):
        payload = dict(body)
        payload["model"] = model
        payload["max_tokens"] = max_tokens
        chat_req = ChatCompletionRequest(**payload)
        try:
            result = await freebuff.chat_completions(chat_req, request_id)
            if not is_streaming:
                _call_logger.log(request_id=request_id, provider="freebuff", account_id="freebuff",
                               model=model, stream=False, status="success", http_status=200,
                               latency_ms=int((time.time() - t0) * 1000))
            return result
        except Exception as e:
            logger.exception(f"Unhandled error in freebuff chat_completions: {e}")
            _call_logger.log(request_id=request_id, provider="freebuff", account_id="freebuff",
                           model=model, stream=is_streaming, status="error",
                           latency_ms=int((time.time() - t0) * 1000), error_message=str(e)[:200])
            return JSONResponse(status_code=500, content=make_error_response(str(e), "api_error"))

    deepseek = getattr(request.app.state, "deepseek", None)
    if deepseek and deepseek.can_handle_model(model):
        payload = dict(body)
        payload["model"] = model
        payload["max_tokens"] = max_tokens
        chat_req = ChatCompletionRequest(**payload)
        try:
            result = await deepseek.chat_completions(chat_req, request_id)
            if not is_streaming:
                _call_logger.log(request_id=request_id, provider="deepseek", account_id="deepseek",
                               model=model, stream=False, status="success", http_status=200,
                               latency_ms=int((time.time() - t0) * 1000))
            return result
        except Exception as e:
            logger.exception(f"Unhandled error in deepseek chat_completions: {e}")
            _call_logger.log(request_id=request_id, provider="deepseek", account_id="deepseek",
                           model=model, stream=is_streaming, status="error",
                           latency_ms=int((time.time() - t0) * 1000), error_message=str(e)[:200])
            return JSONResponse(status_code=500, content=make_error_response(str(e), "api_error"))

    nvidia = getattr(request.app.state, "nvidia", None)
    if nvidia and nvidia.can_handle_model(model):
        messages = body.get("messages", [])
        try:
            result = await nvidia.chat_completions(
                model=model,
                messages=messages,
                stream=is_streaming,
                max_tokens=max_tokens,
                temperature=body.get("temperature"),
                top_p=body.get("top_p"),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                request_id=request_id,
            )
            return result
        except Exception as e:
            logger.exception(f"Unhandled error in nvidia chat_completions: {e}")
            _call_logger.log(request_id=request_id, provider="nvidia", account_id="nvidia",
                           model=model, stream=is_streaming, status="error",
                           latency_ms=int((time.time() - t0) * 1000), error_message=str(e)[:200])
            return JSONResponse(status_code=500, content=make_error_response(str(e), "api_error"))

    messages = body.get("messages", [])
    messages = transform_messages(messages, model, streaming=is_streaming)

    session_id: str = request.app.state.session_id
    turn: int = request.app.state.request_count

    payload = _build_payload(body, messages, model, is_streaming, max_tokens, session_id, turn)
    chat_req = ChatCompletionRequest(**payload)

    dispatcher = Dispatcher(auth)
    
    try:
        result = await dispatcher.chat_completions_with_failover(client, chat_req, request_id)
        if not is_streaming:
            _call_logger.log(request_id=request_id, provider="qwen", account_id="qwen",
                           model=model, stream=False, status="success", http_status=200,
                           latency_ms=int((time.time() - t0) * 1000))
        return result
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        error_msg = exc.response.text or str(exc)
        _call_logger.log(request_id=request_id, provider="qwen", account_id="qwen",
                       model=model, stream=is_streaming, status="error",
                       http_status=status, latency_ms=int((time.time() - t0) * 1000),
                       error_message=str(error_msg)[:200])
        return JSONResponse(status_code=status, content=make_error_response(error_msg, "api_error"))
    except Exception as e:
        logger.exception(f"Unhandled error in chat_completions: {e}")
        _call_logger.log(request_id=request_id, provider="qwen", account_id="qwen",
                       model=model, stream=is_streaming, status="error",
                       latency_ms=int((time.time() - t0) * 1000), error_message=str(e)[:200])
        return JSONResponse(status_code=500, content=make_error_response(str(e), "api_error"))


# Legacy debug logging


def log_info(msg: str) -> None:
    logger.info(msg)


def log_warning(msg: str) -> None:
    logger.warning(msg)


def log_error(msg: str) -> None:
    logger.error(msg)
