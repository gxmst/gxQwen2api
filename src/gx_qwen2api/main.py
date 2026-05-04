"""FastAPI application assembly."""

from __future__ import annotations

import logging
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException

from .account_pool import AccountPool
from .auth import AuthManager
from .auto_refresher import AutoRefresher
from .call_logger import call_logger
from .config import settings
from .event_logger import event_logger
from .providers import DeepseekProvider, FreebuffProvider
from .providers.nvidia import NvidiaProvider
from .routes import chat, health, models as models_router

# ── Basic logging setup ──────────────────────────────────────────

_log_level = logging.DEBUG if settings.log_level == "debug" else logging.INFO
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
    force=True,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings.creds_dir.mkdir(parents=True, exist_ok=True)

    # Build account pool
    pool = AccountPool()
    pool.scan()

    # creds_dir scan already ran above; no extra single-file path needed.

    _app.state.pool = pool
    _app.state.auth = AuthManager(pool)
    _app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300, connect=10),
        headers={"User-Agent": "QwenCode/0.14.0 (linux; x64)"},
    )
    _app.state.request_count = 0
    _app.state.session_id = str(uuid.uuid4())
    _app.state.start_time = time.time()
    _app.state.freebuff = FreebuffProvider(pool, _app.state.http_client)
    await _app.state.freebuff.start()

    _app.state.deepseek = DeepseekProvider(pool, _app.state.http_client)
    await _app.state.deepseek.start()

    _app.state.nvidia = NvidiaProvider(_app.state.http_client)
    await _app.state.nvidia.start()
    _app.state.call_logger = call_logger

    # Start auto-refresher background task
    auto_refresher = AutoRefresher(pool, _app.state.auth, _app.state.http_client)
    _app.state.auto_refresher = auto_refresher
    auto_refresher.start()

    event_logger.server_started(host=settings.address, port=settings.port)

    # Print startup summary
    accounts = pool.all_accounts()
    if accounts:
        logging.getLogger("gx_qwen2api").info(
            "Loaded %d account(s): %s",
            len(accounts),
            ", ".join(f"{a.account_id}({a.token_status})" for a in accounts),
        )
    else:
        logging.getLogger("gx_qwen2api").warning(
            "No credentials found in %s", settings.creds_dir
        )

    # SIGHUP handler for credential reload
    def _sighup_handler(signum, frame):  # type: ignore[no-untyped-def]
        logging.getLogger("gx_qwen2api").info("SIGHUP received, reloading credentials...")
        pool.scan()
        for acct in pool.all_accounts():
            pool.reload_account(acct.account_id)

    try:
        signal.signal(signal.SIGHUP, _sighup_handler)
    except (OSError, ValueError):
        pass  # Windows or non-main thread

    yield

    # Shutdown auto-refresher
    await _app.state.auto_refresher.stop()
    await _app.state.freebuff.stop()
    await _app.state.deepseek.stop()
    await _app.state.nvidia.stop()

    event_logger.shutdown("Server stopping")
    await _app.state.http_client.aclose()


app = FastAPI(title="gx2api", lifespan=lifespan)

app.include_router(chat.router)
app.include_router(models_router.router)
app.include_router(health.router)

if settings.admin_enabled:
    from .routes import admin
    app.include_router(admin.router)


def validate_api_key(
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> None:
    keys = settings.api_keys
    if keys is None:
        return
    key = x_api_key
    if not key and authorization:
        key = (
            authorization.removeprefix("Bearer ").strip()
            if authorization.startswith("Bearer ")
            else authorization.strip()
        )
    if not key or key not in keys:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid or missing API key",
                    "type": "authentication_error",
                }
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.address, port=settings.port)
