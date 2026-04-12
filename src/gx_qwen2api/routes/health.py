"""Health check and debug endpoints."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..account_pool import AccountPool
from ..event_logger import event_logger
from .admin import _check_admin

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    pool: AccountPool = request.app.state.pool
    accounts = pool.all_accounts()
    # Use the new health_status field for accurate health counting
    healthy = sum(
        1 for a in accounts
        if a.enabled and a.health_status.value in ("valid", "expiring_soon")
    )
    total = len(accounts)
    persistence_warnings = sum(1 for a in accounts if not a.last_write_persisted)
    
    return {
        "status": "ok",
        "accounts_total": total,
        "accounts_healthy": healthy,
        "persistence_warnings": persistence_warnings,
        "total_requests": request.app.state.request_count,
        "uptime_seconds": round(time.time() - request.app.state.start_time, 1),
    }


@router.get("/healthz")
async def healthz(request: Request) -> str:
    """Minimal liveness probe."""
    return "ok"


@router.get("/debug/auth")
async def debug_auth(request: Request) -> dict[str, Any]:
    """Return detailed state of all accounts."""
    _check_admin(request)
    pool: AccountPool = request.app.state.pool
    return {
        "creds_dir": str(pool.accounts and next(iter(pool.accounts.values())).creds_file.parent if pool.accounts else ""),
        "accounts": [a.to_dict() for a in pool.all_accounts()],
    }


@router.get("/debug/logs")
async def debug_logs(request: Request, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent event log entries."""
    _check_admin(request)
    return event_logger.get_logs(limit=limit)
