"""DeepSeek authentication and session management.

Aligned with ds-free-api protocol:
- login: POST /users/login with email|mobile + password + device_id + os
- create_session: POST /chat_session/create with empty body, parse data.biz_data.chat_session.id
- delete_session: POST /chat_session/delete with {chat_session_id: ...}
- pow_challenge: POST /chat/create_pow_challenge with {target_path: "/api/v0/chat/completion"}
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import httpx

from .models import DeepseekAccount

logger = logging.getLogger("gx_qwen2api.deepseek.auth")

API_BASE = "https://chat.deepseek.com/api/v0"
DEVICE_ID = "deepseek-web-client"
CLIENT_VERSION = "1.8.0"
CLIENT_PLATFORM = "web"


def _build_headers(token: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/json",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Content-Type": "application/json",
        "Origin": "https://chat.deepseek.com",
        "Referer": "https://chat.deepseek.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/134.0.0.0 Safari/537.36"
        ),
        "X-Client-Version": CLIENT_VERSION,
        "X-Client-Platform": CLIENT_PLATFORM,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


async def _request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    token: str | None = None,
    json_data: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> httpx.Response:
    url = f"{API_BASE.rstrip('/')}{path}"
    headers = _build_headers(token, extra_headers)
    return await client.request(
        method,
        url,
        headers=headers,
        json=json_data,
        timeout=timeout,
        follow_redirects=True,
    )


def _parse_envelope(resp_json: dict[str, Any]) -> Any | None:
    """Parse DeepSeek envelope: {code, msg, data: {biz_code, biz_msg, biz_data}}."""
    if resp_json.get("code") != 0:
        logger.warning("DeepSeek envelope code error: %s %s", resp_json.get("code"), resp_json.get("msg"))
        return None
    data = resp_json.get("data")
    if not isinstance(data, dict):
        return None
    if data.get("biz_code") != 0:
        logger.warning("DeepSeek biz_code error: %s %s", data.get("biz_code"), data.get("biz_msg"))
        return None
    return data.get("biz_data")


async def login(
    client: httpx.AsyncClient,
    email: str = "",
    password: str = "",
    mobile: str = "",
    area_code: str = "",
) -> dict[str, Any] | None:
    """Login with email/password or mobile+area_code/password to /users/login.

    Returns {"token": str, "user_id": str, "email": str|None, "mobile": str|None} or None.
    """
    payload: dict[str, Any] = {
        "password": password,
        "device_id": "",
        "os": "web",
    }
    if email:
        payload["email"] = email
    if mobile:
        payload["mobile"] = mobile
    if area_code:
        payload["area_code"] = area_code

    if not email and not mobile:
        logger.warning("DeepSeek login requires email or mobile")
        return None

    login_resp = await _request(client, "POST", "/users/login", json_data=payload)
    if login_resp.status_code != 200:
        logger.warning("DeepSeek login failed: %s %s", login_resp.status_code, login_resp.text[:200])
        return None

    try:
        login_data = login_resp.json()
    except Exception as exc:
        logger.warning("DeepSeek login response parse error: %s", exc)
        return None

    biz_data = _parse_envelope(login_data)
    if not isinstance(biz_data, dict):
        logger.warning("DeepSeek login response missing biz_data")
        return None

    user = biz_data.get("user")
    if not isinstance(user, dict):
        logger.warning("DeepSeek login response missing user")
        return None

    token = user.get("token", "")
    if not token:
        logger.warning("DeepSeek login response missing token")
        return None

    return {
        "token": token,
        "user_id": user.get("id", ""),
        "email": user.get("email") or email,
        "mobile": user.get("mobile_number"),
    }


async def refresh_token_if_needed(
    client: httpx.AsyncClient,
    account: DeepseekAccount,
    threshold_seconds: float = 300.0,
    force: bool = False,
) -> bool:
    """Check token validity via lightweight call; re-login if needed."""
    if account.access_token and not force:
        # Try a lightweight authenticated call: create_pow_challenge
        try:
            challenge_resp = await _request(
                client, "POST", "/chat/create_pow_challenge",
                token=account.access_token,
                json_data={"target_path": "/api/v0/chat/completion"},
            )
            if challenge_resp.status_code == 200:
                return True
            if challenge_resp.status_code == 401:
                # Token expired, fall through to re-login
                pass
        except Exception:
            pass

    # Fallback: re-login with credentials
    if account.email and account.password:
        result = await login(client, email=account.email, password=account.password)
        if result:
            account.access_token = result["token"]
            account.last_login_at = time.time()
            account.last_error = None
            logger.info("DeepSeek re-login succeeded for %s", account.account_id)
            return True

    if account.mobile and account.password:
        result = await login(client, mobile=account.mobile, password=account.password, area_code=account.area_code)
        if result:
            account.access_token = result["token"]
            account.last_login_at = time.time()
            account.last_error = None
            logger.info("DeepSeek re-login (mobile) succeeded for %s", account.account_id)
            return True

    account.last_error = "Token refresh/re-login failed"
    return False


async def create_chat_session(client: httpx.AsyncClient, token: str) -> str | None:
    """Create a new chat session and return session_id.

    Reference sends empty JSON body and parses data.biz_data.chat_session.id.
    """
    resp = await _request(
        client,
        "POST",
        "/chat_session/create",
        token=token,
        json_data={},
    )
    if resp.status_code != 200:
        logger.warning("DeepSeek create session failed: %s %s", resp.status_code, resp.text[:200])
        return None
    try:
        data = resp.json()
    except Exception as exc:
        logger.warning("DeepSeek create session parse error: %s", exc)
        return None

    biz_data = _parse_envelope(data)
    if not isinstance(biz_data, dict):
        return None

    chat_session = biz_data.get("chat_session")
    if isinstance(chat_session, dict):
        session_id = chat_session.get("id", "")
        if session_id:
            return str(session_id)

    # Fallback: try direct id field
    session_id = biz_data.get("id", "")
    if session_id:
        return str(session_id)
    return None


async def delete_chat_session(client: httpx.AsyncClient, token: str, session_id: str) -> None:
    """Delete a chat session (best-effort).

    Reference uses {chat_session_id: session_id}.
    """
    try:
        await _request(
            client,
            "POST",
            "/chat_session/delete",
            token=token,
            json_data={"chat_session_id": session_id},
            timeout=10.0,
        )
    except Exception:
        logger.debug("DeepSeek delete session failed for %s", session_id, exc_info=True)


async def create_pow_challenge(client: httpx.AsyncClient, token: str) -> dict[str, Any] | None:
    """Request a PoW challenge for chat completion.

    Returns challenge dict with algorithm, challenge, salt, signature, difficulty, expire_after, expire_at, target_path.
    """
    resp = await _request(
        client,
        "POST",
        "/chat/create_pow_challenge",
        token=token,
        json_data={"target_path": "/api/v0/chat/completion"},
    )
    if resp.status_code != 200:
        logger.warning("DeepSeek create_pow_challenge failed: %s %s", resp.status_code, resp.text[:200])
        return None
    try:
        data = resp.json()
    except Exception as exc:
        logger.warning("DeepSeek create_pow_challenge parse error: %s", exc)
        return None

    biz_data = _parse_envelope(data)
    if not isinstance(biz_data, dict):
        return None

    challenge = biz_data.get("challenge")
    if isinstance(challenge, dict):
        return challenge
    return None


async def edit_message(
    client: httpx.AsyncClient,
    token: str,
    pow_header: str,
    payload: dict[str, Any],
) -> httpx.Response:
    """Send edit_message request with PoW header."""
    return await _request(
        client,
        "POST",
        "/chat/edit_message",
        token=token,
        json_data=payload,
        extra_headers={"X-Ds-Pow-Response": pow_header},
    )


async def completion(
    client: httpx.AsyncClient,
    token: str,
    pow_header: str,
    payload: dict[str, Any],
) -> httpx.Response:
    """Send the first message in a chat session with PoW header."""
    return await _request(
        client,
        "POST",
        "/chat/completion",
        token=token,
        json_data=payload,
        extra_headers={"X-Ds-Pow-Response": pow_header},
    )


async def stop_stream(
    client: httpx.AsyncClient,
    token: str,
    session_id: str,
    message_id: int,
) -> None:
    """Stop an ongoing stream (best-effort)."""
    try:
        await _request(
            client,
            "POST",
            "/chat/stop_stream",
            token=token,
            json_data={
                "chat_session_id": session_id,
                "message_id": message_id,
            },
            timeout=10.0,
        )
    except Exception:
        logger.debug("DeepSeek stop_stream failed", exc_info=True)


def build_pow_header(challenge: dict[str, Any], answer: int) -> str:
    """Build base64-encoded X-Ds-Pow-Response header from challenge + answer."""
    payload = {
        "algorithm": challenge.get("algorithm", "DeepSeekHashV1"),
        "challenge": challenge.get("challenge", ""),
        "salt": challenge.get("salt", ""),
        "answer": answer,
        "signature": challenge.get("signature", ""),
        "target_path": challenge.get("target_path", "/api/v0/chat/completion"),
    }
    json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(json_bytes).decode("utf-8")
