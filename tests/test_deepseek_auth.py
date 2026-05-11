from __future__ import annotations

import httpx
import pytest

from gx_qwen2api.providers.deepseek.auth import login


class _LoginTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            return httpx.Response(500, request=request)
        response = self.responses.pop(0)
        return response


@pytest.mark.asyncio
async def test_login_retries_on_202_then_accepts_success() -> None:
    responses = [
        httpx.Response(202, text=""),
        httpx.Response(200, json={"code": 0, "data": {"biz_code": 0, "biz_data": {"user": {"token": "tok-123"}}}}),
    ]
    transport = _LoginTransport(responses)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await login(client, email="a@example.com", password="secret")

    assert result is not None
    assert result["token"] == "tok-123"
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_login_keeps_failing_after_three_pending_responses() -> None:
    responses = [httpx.Response(202, text="")] * 3
    transport = _LoginTransport(responses)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await login(client, email="a@example.com", password="secret")

    assert result is None
    assert len(transport.requests) == 3
