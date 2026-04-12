"""Admin routes: reload creds, enable/disable accounts, scan, logs, dashboard."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..account_pool import AccountPool
from ..auth import AuthManager
from ..config import settings
from ..event_logger import event_logger

router = APIRouter(prefix="/admin")

COOKIE_NAME = "qwen_admin"


def _admin_cookie_hash() -> str | None:
    """Return the expected cookie value, or None when no password is set."""
    if not settings.admin_password:
        return None
    return hashlib.sha256(settings.admin_password.encode()).hexdigest()


def _check_admin(request: Request) -> None:
    """Raise 403 when ADMIN_PASSWORD is set and the cookie is missing / wrong."""
    expected = _admin_cookie_hash()
    if expected is None:
        return  # No password configured → open access
    actual = request.cookies.get(COOKIE_NAME)
    if not actual or actual != expected:
        raise HTTPException(status_code=403, detail="Forbidden — login required")


def _set_admin_cookie(response: Response) -> None:
    h = _admin_cookie_hash()
    if h:
        response.set_cookie(
            key=COOKIE_NAME,
            value=h,
            httponly=True,
            samesite="lax",
            max_age=86400,
        )


# ── Login / Page ─────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = "") -> HTMLResponse:
    return HTMLResponse(
        content=f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Admin Login</title>
<style>
body{{background:#0d1117;color:#c9d1d9;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}}
form{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:24px;width:320px}}
input{{width:100%;padding:8px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#c9d1d9;margin:8px 0 12px;box-sizing:border-box}}
button{{width:100%;padding:8px;border-radius:6px;border:none;background:#238636;color:#fff;font-weight:600;cursor:pointer}}
button:hover{{background:#2ea043}}
.err{{color:#f85149;font-size:0.85rem;margin-bottom:8px}}
</style></head><body>
<form method="post" action="/admin/login">
<h2 style="margin:0 0 4px">Admin Login</h2>
{"<div class=err>" + html.escape(error) + "</div>" if error else ""}
<input type="password" name="p" placeholder="Password" autofocus>
<button type="submit">Sign in</button>
</form></body></html>""",
    )


@router.post("/login")
async def do_login(request: Request) -> Response:
    form = await request.form()
    pwd = form.get("p", "")
    if _admin_cookie_hash() == hashlib.sha256(str(pwd).encode()).hexdigest():
        resp: Response = RedirectResponse(url="/admin/", status_code=303)
        _set_admin_cookie(resp)
        return resp
    return RedirectResponse(url="/admin/login?error=Wrong+password", status_code=303)


@router.post("/logout")
async def do_logout() -> Response:
    resp: Response = RedirectResponse(url="/admin/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp


@router.get("/", response_class=HTMLResponse)
async def admin_page(request: Request) -> HTMLResponse:
    _check_admin(request)
    if not settings.admin_enabled:
        raise HTTPException(status_code=404, detail="Admin disabled")
    tpl = Path(__file__).resolve().parent.parent / "static" / "admin.html"
    return HTMLResponse(content=tpl.read_text(encoding="utf-8"))


# ── API endpoints ────────────────────────────────────────────────

@router.get("/api/accounts")
async def api_accounts(request: Request) -> list[dict[str, Any]]:
    _check_admin(request)
    pool: AccountPool = request.app.state.pool
    return [a.to_dict() for a in pool.all_accounts()]


@router.get("/api/logs")
async def api_logs(request: Request, limit: int = 100) -> list[dict[str, Any]]:
    _check_admin(request)
    return event_logger.get_logs(limit=limit)


@router.post("/api/scan")
async def api_scan(request: Request) -> dict[str, str]:
    _check_admin(request)
    pool: AccountPool = request.app.state.pool
    pool.scan()
    return {"status": "ok", "detail": f"Found {len(pool.accounts)} accounts"}


@router.post("/api/reload")
async def api_reload(request: Request, account_id: str) -> dict[str, Any]:
    _check_admin(request)
    pool: AccountPool = request.app.state.pool
    ok = pool.reload_account(account_id)
    return {"status": "ok" if ok else "not_found", "account_id": account_id}


@router.post("/api/reload-all")
async def api_reload_all(request: Request) -> dict[str, str]:
    _check_admin(request)
    pool: AccountPool = request.app.state.pool
    count = 0
    for acct in pool.all_accounts():
        if pool.reload_account(acct.account_id):
            count += 1
    return {"status": "ok", "detail": f"Reloaded {count} accounts"}


@router.post("/api/refresh/{account_id}")
async def api_refresh(request: Request, account_id: str) -> dict[str, Any]:
    _check_admin(request)
    pool: AccountPool = request.app.state.pool
    auth: AuthManager = request.app.state.auth
    client = request.app.state.http_client
    acct = pool.get_account(account_id)
    if not acct:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    if not acct._raw_creds or not acct._raw_creds.get("refresh_token"):
        raise HTTPException(status_code=400, detail="No refresh token for this account")
    ok = await auth.refresh_token(acct, client)
    return {"status": "ok" if ok else "failed", "account_id": account_id}


@router.post("/api/enable/{account_id}")
async def api_enable(request: Request, account_id: str) -> dict[str, Any]:
    _check_admin(request)
    pool: AccountPool = request.app.state.pool
    ok = pool.enable_account(account_id)
    return {"status": "ok" if ok else "not_found", "account_id": account_id}


@router.post("/api/disable/{account_id}")
async def api_disable(request: Request, account_id: str) -> dict[str, Any]:
    _check_admin(request)
    pool: AccountPool = request.app.state.pool
    ok = pool.disable_account(account_id)
    return {"status": "ok" if ok else "not_found", "account_id": account_id}


@router.post("/api/logs/clear")
async def api_clear_logs(request: Request) -> dict[str, str]:
    _check_admin(request)
    event_logger.clear_logs()
    return {"status": "ok"}


# ── Upload credential ────────────────────────────────────────────

_MAX_UPLOAD_BYTES = 64 * 1024  # 64 KB
_REQUIRED_KEYS = {"refresh_token"}  # Must have at minimum this key

# Sanitize: allow alphanumeric, dash, underscore, dot. Strip path separators.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")


@router.post("/api/upload")
async def api_upload(request: Request, file: UploadFile) -> dict[str, Any]:
    _check_admin(request)
    """Validate + save uploaded credential JSON file."""
    # 1. Check filename
    name = file.filename or ""
    # Strip any directory traversal
    name = Path(name).name
    if not name or not _FILENAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename: {html.escape(name)}. Must be a simple .json name (alphanumeric, dashes, underscores, dots).",
        )

    # 2. Read with size limit
    raw = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {_MAX_UPLOAD_BYTES // 1024} KB).",
        )

    # 3. Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Not valid JSON.")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="JSON must be an object.")

    # 4. Validate it looks like Qwen creds
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required keys: {', '.join(sorted(missing))}. Expected a Qwen OAuth credential file.",
        )

    # 5. Ensure sensible defaults
    data.setdefault("access_token", "")
    data.setdefault("token_type", "Bearer")
    data.setdefault("resource_url", "https://portal.qwen.ai/v1")
    if not data.get("expiry_date"):
        import time
        data["expiry_date"] = int(time.time() * 1000) + 7 * 24 * 3600 * 1000  # 7 days

    # 6. Save to creds_dir
    pool: AccountPool = request.app.state.pool
    target = settings.creds_dir / name
    exists = target.exists()
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    account_id = Path(name).stem

    # 7. Register in pool
    state = pool._try_load_account(account_id, target)
    if state:
        pool.accounts[account_id] = state
    else:
        pool.scan()

    logging.getLogger("gx_qwen2api").info(
        "Uploaded credential %s (%s)", account_id, "overwrite" if exists else "new"
    )

    event_logger._emit(
        logging.INFO, "upload",
        {
            "account_id": account_id,
            "detail": f"{'Overwritten' if exists else 'New'} credential: {account_id}",
        },
    )

    return {
        "status": "ok",
        "account_id": account_id,
        "action": "overwritten" if exists else "new",
    }
