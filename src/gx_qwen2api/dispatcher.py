"""Dispatch logic for multi-account failover and rate-limit resiliency."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Protocol

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .account_pool import AccountState
from .auth import AuthManager
from .config import settings
from .event_logger import event_logger
from .headers import build_headers
from .models import ChatCompletionRequest, is_auth_error, is_quota_error, is_validation_error

logger = logging.getLogger("gx_qwen2api.dispatcher")


class RequestSender(Protocol):
    """Protocol for sending a request to the upstream API."""
    async def __call__(
        self, 
        client: httpx.AsyncClient, 
        account_id: str, 
        token: str, 
        endpoint: str, 
        request: ChatCompletionRequest,
        request_id: str
    ) -> httpx.Response:
        ...


class Dispatcher:
    """Handles multi-account dispatching, 429 failover, and retries."""

    def __init__(self, auth_mgr: AuthManager) -> None:
        self.auth_mgr = auth_mgr

    async def chat_completions_with_failover(
        self,
        client: httpx.AsyncClient,
        request: ChatCompletionRequest,
        request_id: str,
    ) -> httpx.Response | StreamingResponse:
        """Execute chat completion with transparent failover and backoff."""
        
        last_error_msg = "Unknown error"
        last_status = 500
        tried_accounts: set[str] = set()
        
        # We allow up to max_retries total attempts across all accounts
        for attempt in range(1, settings.max_retries + 1):
            account_id = "unknown"
            try:
                # 1. Select a healthy account and get token
                # This automatically skips accounts in cooldown or rate-limited
                token, account_id = await self.auth_mgr.get_valid_token(client, exclude_ids=tried_accounts)
                acct = self.auth_mgr.pool.get_account(account_id)
                if not acct:
                    raise RuntimeError(f"Account {account_id} vanished from pool")

                endpoint = self.auth_mgr.get_api_endpoint(acct)
                
                # PRE-FLIGHT: LOG REQUEST
                event_logger.proxy_request(
                    request_id=request_id,
                    model=request.model,
                    account_id=account_id,
                    token_count=0, # Simplified
                    is_streaming=request.stream
                )

                # 2. Execute request
                # EXTENSION POINT: Future proxy/transport binding can be added here
                # Example: transport = self._get_transport_for_account(account_id)
                
                resp = await self._do_request(client, acct, token, endpoint, request, request_id)
                
                # 3. Handle 429 (Rate Limit)
                if resp.status_code == 429:
                    retry_after = self._parse_retry_after(resp.headers.get("retry-after"))
                    # Default cooldown if no Retry-After, use exponential basis for multiple hits
                    cooldown = retry_after or min(300, 30 * (2 ** (acct.rate_limit_count % 4)))
                    
                    acct.mark_rate_limited(f"Upstream 429", cooldown)
                    event_logger.rate_limit_hit(account_id, retry_after, "Too Many Requests")
                    
                    if attempt < settings.max_retries:
                        # Failover strategy: check if ANY other account is available
                        # We merge tried_accounts with current account to see if there's anywhere else to go
                        alternatives = self.auth_mgr.pool.select_account(exclude_ids=tried_accounts.union({account_id}))
                        
                        if alternatives:
                            logger.warning(f"Attempt {attempt} hit 429 on {account_id}. Failing over...")
                            tried_accounts.add(account_id)
                            continue
                        else:
                            logger.warning(f"Attempt {attempt} hit 429 on {account_id}. No more healthy accounts. Returning 429.")
                    
                    # If we're here, either it's the last attempt or no healthy accounts left.
                    # Stop failover and return the 429 response to the user so they see the real context.
                    return JSONResponse(
                        status_code=429,
                        content=resp.json() if resp.text else {"error": {"message": "Too Many Requests", "type": "rate_limit_error"}},
                        headers={"Retry-After": str(retry_after)} if retry_after else {}
                    )

                # 4. Handle Auth errors (401/403)
                if is_auth_error(resp.status_code, resp.text):
                    logger.warning(f"Auth error (401/403) on {account_id}. Forcing refresh...")
                    await self.auth_mgr.refresh_token(acct, client)
                    if attempt < settings.max_retries:
                        # Optional: should we exclude here? Usually no, because we just refreshed.
                        # But to be safe against infinite refresh loops on same bad account, we could.
                        # For now, we trust refresh_token state.
                        continue
                    else:
                        resp.raise_for_status()

                # 5. Handle Success or non-retryable errors
                resp.raise_for_status()
                
                # SUCCESS
                if request.stream:
                    return self._create_streaming_response(resp, account_id, request_id)
                else:
                    return JSONResponse(content=resp.json())

            except httpx.HTTPStatusError as exc:
                last_status = exc.response.status_code
                last_error_msg = f"HTTP {last_status}: {exc.response.text[:200]}"
                
                # If we've exhausted retries or it's not a retryable error
                if attempt >= settings.max_retries:
                    break
                
                # Exponential backoff for general errors (500, etc.)
                wait_time = settings.retry_delay_s * (2 ** (attempt - 1))
                logger.warning(f"Retry {attempt}/{settings.max_retries} due to {last_status}. Waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                last_error_msg = str(e)
                logger.error(f"Dispatch error on {account_id}: {e}")
                if attempt >= settings.max_retries:
                    break
                await asyncio.sleep(settings.retry_delay_s)

        # If we reach here, all retries failed
        event_logger.proxy_error(request_id, last_status, "system", last_error_msg)
        raise httpx.HTTPStatusError(
            f"All retry attempts failed. Last error: {last_error_msg}",
            request=httpx.Request("POST", settings.qwen_api_base),
            response=httpx.Response(last_status, content=json.dumps({"error": {"message": last_error_msg, "type": "api_error"}}).encode())
        )

    async def _do_request(
        self, 
        client: httpx.AsyncClient, 
        acct: AccountState, 
        token: str, 
        endpoint: str, 
        request: ChatCompletionRequest,
        request_id: str
    ) -> httpx.Response:
        """Low-level request execution."""
        payload = request.model_dump(exclude_none=True)
        headers = build_headers(token, streaming=request.stream)
        headers["X-Request-ID"] = request_id
        
        # EXTENSION POINT: Binding per-account proxies would happen here by modifying 'client' or transport
        
        return await client.post(
            f"{endpoint}/chat/completions",
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def _create_streaming_response(self, resp: httpx.Response, account_id: str, request_id: str) -> StreamingResponse:
        """Wrap the generator to handle mid-stream errors and return error SSE."""
        
        async def generate() -> AsyncGenerator[bytes, None]:
            t0 = time.monotonic()
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                
                # Log success after full stream
                latency_ms = int((time.monotonic() - t0) * 1000)
                event_logger.proxy_response(request_id, 200, account_id, latency_ms)
                
            except Exception as e:
                logger.error(f"Stream error for {request_id} (account: {account_id}): {e}")
                # Send error SSE event
                # OpenAI spec: some clients look for an 'error' block in the JSON data
                err_json = json.dumps({
                    "error": {
                        "message": f"Streaming error: {str(e)}",
                        "type": "api_error",
                        "account_id": account_id
                    }
                })
                yield f"event: error\ndata: {err_json}\n\n".encode("utf-8")
                
            finally:
                await resp.aclose()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
        )

    def _parse_retry_after(self, value: str | None) -> int | None:
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
