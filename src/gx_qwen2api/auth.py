"""Qwen OAuth credential management with detailed refresh logging."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import time
from typing import Any

import httpx

from .account_pool import AccountPool, AccountState
from .config import settings
from .headers import USER_AGENT
from .event_logger import event_logger


class AuthManager:
    """Manages Qwen OAuth credentials across multiple accounts."""

    def __init__(self, pool: AccountPool) -> None:
        self.pool = pool
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, account_id: str) -> asyncio.Lock:
        if account_id not in self._locks:
            self._locks[account_id] = asyncio.Lock()
        return self._locks[account_id]

    async def refresh_token(
        self, acct: AccountState, client: httpx.AsyncClient
    ) -> bool:
        """Refresh a single account's access token. Returns True on success."""
        raw = acct._raw_creds
        if not raw or not raw.get("refresh_token"):
            acct.record_error("No refresh token available")
            acct.mark_refresh_failed("No refresh token available")
            event_logger.refresh_failed(
                account_id=acct.account_id,
                reason="no_refresh_token",
                url=settings.qwen_oauth_token_url,
            )
            return False

        url = settings.qwen_oauth_token_url
        t0 = time.monotonic()
        event_logger.refresh_started(account_id=acct.account_id, url=url)

        try:
            resp = await client.post(
                url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": raw["refresh_token"],
                    "client_id": settings.qwen_oauth_client_id,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
        except TimeoutError:
            elapsed = time.monotonic() - t0
            acct.record_error(f"Timeout after {elapsed:.2f}s")
            acct.mark_refresh_failed(f"Timeout after {elapsed:.2f}s")
            event_logger.refresh_failed(
                account_id=acct.account_id,
                reason="timeout",
                url=url,
                elapsed_ms=elapsed * 1000,
            )
            return False
        except Exception as exc:
            elapsed = time.monotonic() - t0
            acct.record_error(f"Request error: {exc}")
            acct.mark_refresh_failed(f"Request error: {exc}")
            event_logger.refresh_failed(
                account_id=acct.account_id,
                reason="request_error",
                url=url,
                elapsed_ms=elapsed * 1000,
                error=str(exc)[:300],
            )
            return False

        elapsed_ms = (time.monotonic() - t0) * 1000

        if resp.status_code != 200:
            body_preview = resp.text[:500]
            error_msg = (
                f"status={resp.status_code}, "
                f"content_type={resp.headers.get('content-type', '<missing>')}, "
                f"body={body_preview!r}"
            )
            acct.record_error(error_msg)
            acct.mark_refresh_failed(error_msg)
            event_logger.refresh_failed(
                account_id=acct.account_id,
                reason="non_200",
                url=url,
                status_code=resp.status_code,
                content_type=resp.headers.get("content-type", "<missing>"),
                elapsed_ms=elapsed_ms,
                body_preview=body_preview,
            )
            return False

        # Parse response
        try:
            token_data: dict[str, Any] = resp.json()
        except json.JSONDecodeError:
            body_preview = resp.text[:500]
            acct.record_error(
                f"invalid JSON: status={resp.status_code}, "
                f"content_type={resp.headers.get('content-type', '<missing>')}, "
                f"body={body_preview!r}"
            )
            acct.mark_refresh_failed(f"invalid JSON: status={resp.status_code}")
            event_logger.refresh_failed(
                account_id=acct.account_id,
                reason="invalid_json",
                url=url,
                status_code=resp.status_code,
                content_type=resp.headers.get("content-type", "<missing>"),
                elapsed_ms=elapsed_ms,
                body_preview=body_preview,
            )
            return False

        # Persist to disk
        old_url = raw.get("resource_url", "")
        new_url = token_data.get("resource_url", old_url)
        new_expiry = int(time.time() * 1000) + int(token_data.get("expires_in", 0)) * 1000
        new_rt = token_data.get("refresh_token", raw.get("refresh_token", ""))
        
        raw["access_token"] = token_data["access_token"]
        raw["refresh_token"] = new_rt
        raw["expiry_date"] = new_expiry
        raw["token_type"] = token_data.get("token_type", raw.get("token_type", ""))
        raw["resource_url"] = new_url
        
        # Sync in-memory state
        acct.access_token = token_data["access_token"]
        acct.expiry_date = new_expiry
        acct.refresh_token_hash = (
            hashlib.sha256(new_rt.encode()).hexdigest()[:8] if new_rt else ""
        )
        acct._raw_creds = raw.copy() # Use a copy to avoid external mutations
        acct.last_error = ""

        self.pool.save_creds(acct.account_id, raw)

        dt = datetime.datetime.fromtimestamp(
            new_expiry / 1000, tz=datetime.timezone.utc
        )
        acct.last_refresh = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        acct.record_success()

        url_changed = (old_url != new_url)
        logger = logging.getLogger("gx_qwen2api")
        logger.info(
            "Account %s refresh success. URL changed: %s, URL: %s",
            acct.account_id, url_changed, new_url
        )

        event_logger.refresh_succeeded(
            account_id=acct.account_id,
            elapsed_ms=elapsed_ms,
            expires_at=dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            resource_url=new_url,
            url_changed=url_changed
        )
        return True

    async def coordinated_refresh(
        self, acct: AccountState, client: httpx.AsyncClient
    ) -> bool:
        """Refresh with per-account concurrency coordination.
 
        If another coroutine is already refreshing this account, wait for it
        instead of starting a duplicate refresh. Returns True on success.
        """
        aid = acct.account_id
        async with self._get_lock(aid):
            # Re-read state in case another coroutine already refreshed it while we waited
            acct = self.pool.get_account(aid) or acct
            if acct.token_valid:
                return True
                
            return await self.refresh_token(acct, client)

    async def get_valid_token(
        self, client: httpx.AsyncClient, exclude_ids: set[str] | None = None
    ) -> tuple[str, str]:
        """Return (access_token, account_id) for a healthy account.

        Tries selection, refreshes if expired.
        Concurrent callers for the same account wait for the in-flight refresh.
        """
        tried = set(exclude_ids or [])
        for _ in range(len(self.pool.accounts)):
            acct = self.pool.select_account(exclude_ids=tried)
            if not acct:
                break
            
            account_id = acct.account_id
            tried.add(account_id)

            # Check file mtime for hot-reload
            self.pool.check_mtime_and_reload(account_id)
            acct = self.pool.get_account(account_id) or acct

            if acct.token_valid:
                return acct.access_token, account_id

            # Need refresh — use coordinated refresh to avoid races
            ok = await self.coordinated_refresh(acct, client)
            acct = self.pool.get_account(account_id) or acct
            if ok and acct.token_valid:
                return acct.access_token, account_id

        raise RuntimeError(
            "No valid token available. All accounts are expired, in cooldown, or disabled."
        )

    def get_api_endpoint(self, acct: AccountState | None) -> str:
        if acct and acct._raw_creds and acct._raw_creds.get("resource_url"):
            endpoint = acct._raw_creds["resource_url"]
            if not endpoint.startswith("http"):
                endpoint = f"https://{endpoint}"
            if not endpoint.endswith("/v1"):
                endpoint = endpoint.rstrip("/") + "/v1"
            return endpoint
        return settings.qwen_api_base
