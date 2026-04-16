"""Background auto-refresh task for keeping access tokens alive."""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from typing import Any

import httpx

from .account_pool import AccountPool, AccountState
from .auth import AuthManager
from .config import settings
from .event_logger import event_logger

logger = logging.getLogger("gx_qwen2api")


class AutoRefresher:
    """Lightweight background task that periodically refreshes tokens
    before they expire. Runs as an asyncio background task.

    Decision logic per account:
    - skip if not enabled
    - skip if no refresh_token
    - skip if in cooldown
    - skip if token still has plenty of time (> threshold)
    - otherwise, try to refresh
    """

    def __init__(
        self,
        pool: AccountPool,
        auth: AuthManager,
        client: httpx.AsyncClient,
    ) -> None:
        self.pool = pool
        self.auth = auth
        self.client = client
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_cycle_started: float = 0.0
        self._last_cycle_result: str = ""  # "completed", "no_accounts", "error"

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        if not settings.auto_refresh_enabled:
            logger.info("Auto-refresh is disabled (AUTO_REFRESH_ENABLED=false)")
            return
        if self._task is not None:
            return  # Already running
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Auto-refresh started: every %ds, threshold %dmin",
            settings.auto_refresh_interval_seconds,
            settings.auto_refresh_threshold_minutes,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Auto-refresh stopped")

    async def run_once(self) -> str:
        """Run a single auto-refresh cycle manually. Returns result string."""
        return await self._cycle()

    # ── Internal ──────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                result = await self._cycle()
                self._last_cycle_result = result
                logger.debug("Auto-refresh cycle completed: %s", result)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Auto-refresh cycle error")
                self._last_cycle_result = "error"

            # Sleep in small chunks so we can exit quickly on shutdown
            try:
                for _ in range(settings.auto_refresh_interval_seconds * 10):
                    if not self._running:
                        break
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break

    async def _cycle(self) -> str:
        """Execute one auto-refresh cycle. Returns summary."""
        self._last_cycle_started = time.time()
        accounts = self.pool.all_accounts()
        if not accounts:
            return "no_accounts"

        event_logger.auto_refresh_start()
        refreshed = 0
        skipped = 0
        failed = 0
        threshold_ms = settings.auto_refresh_threshold_minutes * 60 * 1000
        now_ms = time.time() * 1000

        for acct in accounts:
            # Determine if this account needs auto-refresh
            decision = self._should_refresh(acct, now_ms, threshold_ms)
            if decision != "refresh":
                skipped += 1
                if decision != "healthy":
                    acct.last_auto_refresh_result = "skipped"
                    acct.last_auto_refresh_at = time.time()
                continue

            try:
                ok = await self.auth.coordinated_refresh(acct, self.client)
                acct = self.pool.get_account(acct.account_id) or acct
                acct.last_auto_refresh_at = time.time()
                if ok and acct.token_valid:
                    refreshed += 1
                    acct.last_auto_refresh_result = "success"
                    acct.last_auto_refresh_error = ""
                    import datetime
                    dt = datetime.datetime.fromtimestamp(
                        acct.expiry_date / 1000, tz=datetime.timezone.utc
                    )
                    
                    raw = acct._raw_creds or {}
                    res_url = raw.get("resource_url", "")
                    
                    event_logger.auto_refresh_ok(
                        account_id=acct.account_id,
                        elapsed_ms=0,  # refresh_token already logs its own time
                        expires_at=dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        resource_url=res_url,
                        url_changed=False # Not easily tracked here, but okay
                    )
                else:
                    failed += 1
                    acct.last_auto_refresh_result = "failed"
                    event_logger.auto_refresh_fail(
                        account_id=acct.account_id,
                        reason="refresh_token_failed",
                        error=acct.last_error or "Unknown error",
                    )
            except Exception as exc:
                failed += 1
                acct.last_auto_refresh_at = time.time()
                acct.last_auto_refresh_result = "failed"
                acct.last_auto_refresh_error = str(exc)[:200]
                acct.record_error(f"Auto-refresh error: {exc}")
                acct.mark_refresh_failed(str(exc)[:200])
                event_logger.auto_refresh_fail(
                    account_id=acct.account_id,
                    reason="exception",
                    error=str(exc)[:200],
                )

        summary = f"refreshed={refreshed} skipped={skipped} failed={failed}"
        logger.info("Auto-refresh cycle: %s", summary)
        return summary

    def _should_refresh(
        self, acct: AccountState, now_ms: float, threshold_ms: float
    ) -> str:
        """Decide if an account needs auto-refresh.

        Returns:
            "refresh" — should refresh now
            "healthy" — token still valid, no action needed
            "skip:<reason>" — skip for a specific reason
        """
        if not acct.enabled:
            return "skip:disabled"

        raw = acct._raw_creds
        if not raw or not raw.get("refresh_token"):
            return "skip:no_refresh_token"

        if acct.is_cooldown:
            return "skip:cooldown"

        # Account has refresh_token but no access_token or expiry_date.
        # This is a recoverable state — we should refresh to bootstrap the
        # account into a usable state. Don't skip it!
        if not acct.access_token or not acct.expiry_date:
            return "refresh"

        # Check if token is close to expiring
        remaining_ms = acct.expiry_date - now_ms
        if remaining_ms > threshold_ms:
            return "healthy"

        # Token is within the refresh threshold — refresh it
        return "refresh"

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_cycle_started(self) -> float | None:
        return self._last_cycle_started or None

    @property
    def last_cycle_result(self) -> str:
        return self._last_cycle_result

    def get_config(self) -> dict[str, Any]:
        """Return current auto-refresh configuration."""
        return {
            "enabled": settings.auto_refresh_enabled,
            "interval_seconds": settings.auto_refresh_interval_seconds,
            "threshold_minutes": settings.auto_refresh_threshold_minutes,
            "is_running": self.is_running,
            "last_cycle_started": self._last_cycle_started or None,
            "last_cycle_result": self._last_cycle_result,
        }
