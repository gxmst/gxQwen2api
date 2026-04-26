"""Admin routes: reload creds, enable/disable accounts, scan, logs, dashboard."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from ..account_pool import AccountPool, AccountState
from ..auth import AuthManager
from ..auto_refresher import AutoRefresher
from ..config import settings
from ..event_logger import event_logger
from ..models import MODELS

router = APIRouter(prefix="/admin")

COOKIE_NAME = "qwen_admin"


def get_admin_user(request: Request) -> None:
    """Dependency for checking admin credentials."""
    if not settings.admin_enabled:
        raise HTTPException(status_code=404, detail="Admin disabled")
    
    expected = _admin_cookie_hash()
    if expected is None:
        return  # No password configured → open access
        
    actual = request.cookies.get(COOKIE_NAME)
    if not actual or actual != expected:
        raise HTTPException(status_code=403, detail="Forbidden — login required")


def _admin_cookie_hash() -> str | None:
    """Return the expected cookie value, or None when no password is set."""
    if not settings.admin_password:
        return None
    return hashlib.sha256(settings.admin_password.encode()).hexdigest()


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
    # Detect language from Accept-Language header
    accept_lang = request.headers.get("accept-language", "")
    lang = "zh" if "zh" in accept_lang else "en"
    
    titles = {"zh": "管理员登录", "en": "Admin Login"}
    placeholders = {"zh": "密码", "en": "Password"}
    submit_btns = {"zh": "登录", "en": "Sign in"}
    error_msgs = {"zh": "密码错误", "en": "Wrong password"}
    
    title = titles[lang]
    placeholder = placeholders[lang]
    submit_btn = submit_btns[lang]
    error_text = error_msgs[lang] if error else ""
    
    return HTMLResponse(
        content=f"""<!DOCTYPE html>
<html lang="{lang}"><head><meta charset="UTF-8"><title>{title}</title>
<style>
body{{background:#0d1117;color:#c9d1d9;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}}
form{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:24px;width:320px}}
input{{width:100%;padding:8px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#c9d1d9;margin:8px 0 12px;box-sizing:border-box}}
button{{width:100%;padding:8px;border-radius:6px;border:none;background:#238636;color:#fff;font-weight:600;cursor:pointer}}
button:hover{{background:#2ea043}}
.err{{color:#f85149;font-size:0.85rem;margin-bottom:8px}}
</style></head><body>
<form method="post" action="/admin/login">
<h2 style="margin:0 0 4px">{title}</h2>
{"<div class=err>" + html.escape(error_text) + "</div>" if error_text else ""}
<input type="password" name="p" placeholder="{placeholder}" autofocus>
<button type="submit">{submit_btn}</button>
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
async def admin_page(request: Request, _=Depends(get_admin_user)) -> HTMLResponse:
    tpl = Path(__file__).resolve().parent.parent / "static" / "admin.html"
    html_content = tpl.read_text(encoding="utf-8")
    # Inject whether admin password is set so the frontend can show logout UI
    # This avoids needing to read the HttpOnly cookie from JS
    is_auth_required = "true" if settings.admin_password else "false"
    html_content = html_content.replace(
        '/* __ADMIN_AUTH_INJECT__ */',
        f'const ADMIN_AUTH_REQUIRED = {is_auth_required};',
    )
    return HTMLResponse(content=html_content)


# ── API endpoints ────────────────────────────────────────────────

@router.get("/api/accounts")
async def api_accounts(request: Request, _=Depends(get_admin_user)) -> list[dict[str, Any]]:
    pool: AccountPool = request.app.state.pool
    return [a.to_dict() for a in pool.all_accounts()]


@router.get("/api/logs")
async def api_logs(request: Request, limit: int = 100, _=Depends(get_admin_user)) -> list[dict[str, Any]]:
    return event_logger.get_logs(limit=limit)


@router.get("/api/models")
async def api_models(request: Request, _=Depends(get_admin_user)) -> dict[str, Any]:
    freebuff = request.app.state.freebuff
    deepseek = getattr(request.app.state, "deepseek", None)
    qwen_models = [m["id"] for m in MODELS if isinstance(m, dict) and "id" in m]
    providers = [
        {
            "provider": "qwen",
            "label": "Qwen OAuth",
            "models": qwen_models,
            "count": len(qwen_models),
            "enabled_accounts": sum(1 for a in request.app.state.pool.all_accounts() if a.provider == "qwen" and a.enabled),
        }
    ]
    if freebuff and freebuff.has_accounts():
        models = freebuff.registry.models()
        providers.append(
            {
                "provider": "freebuff",
                "label": "Freebuff",
                "models": models,
                "count": len(models),
                "enabled_accounts": sum(1 for a in request.app.state.pool.all_accounts() if a.provider == "freebuff" and a.enabled),
            }
        )
    if deepseek and deepseek.has_accounts():
        providers.append(
            {
                "provider": "deepseek",
                "label": "DeepSeek",
                "models": ["deepseek-flash", "deepseek-pro"],
                "count": 2,
                "enabled_accounts": sum(1 for a in request.app.state.pool.all_accounts() if a.provider == "deepseek" and a.enabled),
            }
        )
    return {"providers": providers}


@router.post("/api/scan")
async def api_scan(request: Request, _=Depends(get_admin_user)) -> dict[str, str]:
    pool: AccountPool = request.app.state.pool
    pool.scan()
    return {"status": "ok", "detail": f"Found {len(pool.accounts)} accounts"}


@router.post("/api/reload")
async def api_reload(request: Request, account_id: str, _=Depends(get_admin_user)) -> dict[str, Any]:
    pool: AccountPool = request.app.state.pool
    ok = pool.reload_account(account_id)
    return {"status": "ok" if ok else "not_found", "account_id": account_id}


@router.post("/api/reload-all")
async def api_reload_all(request: Request, _=Depends(get_admin_user)) -> dict[str, str]:
    pool: AccountPool = request.app.state.pool
    count = 0
    for acct in pool.all_accounts():
        if pool.reload_account(acct.account_id):
            count += 1
    return {"status": "ok", "detail": f"Reloaded {count} accounts"}


@router.post("/api/refresh/{account_id}")
async def api_refresh(request: Request, account_id: str, _=Depends(get_admin_user)) -> dict[str, Any]:
    pool: AccountPool = request.app.state.pool
    auth: AuthManager = request.app.state.auth
    client = request.app.state.http_client
    acct = pool.get_account(account_id)
    if not acct:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    if acct.provider != "qwen":
        raise HTTPException(status_code=400, detail=f"Provider {acct.provider} does not support refresh")
    if not acct._raw_creds or not acct._raw_creds.get("refresh_token"):
        raise HTTPException(status_code=400, detail="No refresh token for this account")
    # Use coordinated_refresh to avoid races with background/auto-refresh
    ok = await auth.coordinated_refresh(acct, client)
    return {"status": "ok" if ok else "failed", "account_id": account_id}


@router.post("/api/enable/{account_id}")
async def api_enable(request: Request, account_id: str, _=Depends(get_admin_user)) -> dict[str, Any]:
    pool: AccountPool = request.app.state.pool
    ok = pool.enable_account(account_id)
    return {"status": "ok" if ok else "not_found", "account_id": account_id}


@router.post("/api/disable/{account_id}")
async def api_disable(request: Request, account_id: str, _=Depends(get_admin_user)) -> dict[str, Any]:
    pool: AccountPool = request.app.state.pool
    ok = pool.disable_account(account_id)
    return {"status": "ok" if ok else "not_found", "account_id": account_id}


@router.post("/api/verify/{account_id}")
async def api_verify(request: Request, account_id: str, _=Depends(get_admin_user)) -> dict[str, Any]:
    """Verify if an account's token is actually valid by checking with the API."""
    pool: AccountPool = request.app.state.pool
    auth: AuthManager = request.app.state.auth
    client = request.app.state.http_client
    freebuff = request.app.state.freebuff

    acct = pool.get_account(account_id)
    if not acct:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    if not acct.access_token:
        return {
            "status": "failed",
            "account_id": account_id,
            "detail": "No access token loaded",
            "valid": False,
            "error_type": "no_token",
        }

    if acct.provider == "freebuff":
        return await freebuff.verify_account(acct)
    if acct.provider == "deepseek":
        deepseek = getattr(request.app.state, "deepseek", None)
        if deepseek:
            return await deepseek.verify_account(acct)
        return {
            "status": "failed",
            "account_id": account_id,
            "detail": "DeepSeek provider not initialized",
            "valid": False,
            "error_type": "runtime_error",
        }
    if acct.provider != "qwen":
        return {
            "status": "unsupported",
            "account_id": account_id,
            "detail": f"Verify is not implemented for provider {acct.provider}",
            "valid": None,
            "error_type": "unsupported",
        }

    # Try a simple API call to verify
    try:
        test_resp = await client.get(
            f"{auth.get_api_endpoint(acct)}/models",
            headers={
                "authorization": f"Bearer {acct.access_token}",
                "content-type": "application/json",
            }
        )

        acct.last_auth_check_at = time.time()

        if test_resp.status_code == 200:
            acct.clear_auth_error()
            acct.last_auth_success_at = time.time()
            acct.update_health()
            return {
                "status": "ok",
                "account_id": account_id,
                "detail": "Token is valid",
                "valid": True,
                "error_type": None,
            }
        elif test_resp.status_code in (401, 403):
            error_body = test_resp.text[:300]
            acct.mark_auth_error(f"Verify failed: HTTP {test_resp.status_code}")
            acct.last_auth_failure_at = time.time()
            return {
                "status": "failed",
                "account_id": account_id,
                "detail": f"Authentication failed (HTTP {test_resp.status_code})",
                "valid": False,
                "error_type": "permission",
                "error": error_body,
            }
        elif test_resp.status_code >= 500:
            return {
                "status": "failed",
                "account_id": account_id,
                "detail": f"API endpoint error (HTTP {test_resp.status_code})",
                "valid": False,
                "error_type": "endpoint",
            }
        else:
            return {
                "status": "unknown",
                "account_id": account_id,
                "detail": f"Unexpected response (HTTP {test_resp.status_code})",
                "valid": None,
                "error_type": None,
            }
    except Exception as e:
        acct.last_auth_check_at = time.time()
        acct.last_auth_failure_at = time.time()
        err_str = str(e).lower()
        if any(x in err_str for x in ["connect", "timeout", "refused", "unreachable", "dns"]):
            error_type = "network"
        else:
            error_type = "network"
        return {
            "status": "error",
            "account_id": account_id,
            "detail": f"Verification error: {str(e)[:200]}",
            "valid": False,
            "error_type": error_type,
        }


@router.post("/api/logs/clear")
async def api_clear_logs(request: Request, _=Depends(get_admin_user)) -> dict[str, str]:
    event_logger.clear_logs()
    return {"status": "ok"}


# ── DeepSeek account management ─────────────────────────────

@router.post("/api/deepseek/add")
async def api_deepseek_add(
    request: Request,
    _=Depends(get_admin_user),
) -> dict[str, Any]:
    """Add a new DeepSeek account (email/password) to a JSON file."""
    body = await request.json()
    email = str(body.get("email", "")).strip()
    password = str(body.get("password", "")).strip()
    mobile = str(body.get("mobile", "")).strip()
    area_code = str(body.get("area_code", "")).strip()
    account_id = str(body.get("account_id", "")).strip()

    if not password:
        raise HTTPException(status_code=400, detail="password is required")
    if not email and not mobile:
        raise HTTPException(status_code=400, detail="email or mobile is required")
    if mobile and not area_code:
        raise HTTPException(status_code=400, detail="area_code is required when using mobile login")
    if not account_id:
        account_id = email.split("@")[0] if email else mobile
    if not re.match(r"^[A-Za-z0-9_-]+$", account_id):
        raise HTTPException(status_code=400, detail="account_id must be alphanumeric with dashes/underscores")

    settings.creds_dir.mkdir(parents=True, exist_ok=True)
    target = settings.creds_dir / f"{account_id}.json"

    data = {
        "email": email,
        "password": password,
        "mobile": mobile,
        "area_code": area_code,
        "access_token": "",
        "refresh_token": "",
    }

    action = "new"
    if target.exists():
        action = "overwrite"
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                data["access_token"] = existing.get("access_token", "")
                data["refresh_token"] = existing.get("refresh_token", "")
        except Exception:
            pass

    try:
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write file: {exc}")

    pool: AccountPool = request.app.state.pool
    pool.scan()

    return {"status": "ok", "account_id": account_id, "action": action}


@router.post("/api/deepseek/delete/{account_id}")
async def api_deepseek_delete(
    request: Request,
    account_id: str,
    _=Depends(get_admin_user),
) -> dict[str, Any]:
    """Delete a DeepSeek account file and remove from pool."""
    pool: AccountPool = request.app.state.pool
    acct = pool.get_account(account_id)
    if not acct:
        raise HTTPException(status_code=404, detail="Account not found")
    if acct.provider != "deepseek":
        raise HTTPException(status_code=400, detail="Not a DeepSeek account")

    # Clean up runtime before removing from pool
    deepseek = getattr(request.app.state, "deepseek", None)
    if deepseek:
        deepseek.remove_account_runtime(account_id)

    try:
        if acct.creds_file.exists():
            acct.creds_file.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {exc}")

    pool.accounts.pop(account_id, None)
    return {"status": "ok", "account_id": account_id}


@router.post("/api/deepseek/update/{account_id}")
async def api_deepseek_update(
    request: Request,
    account_id: str,
    _=Depends(get_admin_user),
) -> dict[str, Any]:
    """Update DeepSeek account password or email."""
    pool: AccountPool = request.app.state.pool
    acct = pool.get_account(account_id)
    if not acct:
        raise HTTPException(status_code=404, detail="Account not found")
    if acct.provider != "deepseek":
        raise HTTPException(status_code=400, detail="Not a DeepSeek account")

    body = await request.json()
    new_email = body.get("email")
    new_password = body.get("password")
    new_mobile = body.get("mobile")
    new_area_code = body.get("area_code")

    if new_email is None and new_password is None and new_mobile is None and new_area_code is None:
        raise HTTPException(status_code=400, detail="Nothing to update")

    try:
        raw = json.loads(acct.creds_file.read_text(encoding="utf-8"))
    except Exception:
        raw = {}

    if isinstance(raw, dict):
        if new_email is not None:
            raw["email"] = str(new_email).strip()
        if new_password is not None:
            raw["password"] = str(new_password).strip()
        if new_mobile is not None:
            raw["mobile"] = str(new_mobile).strip()
        if new_area_code is not None:
            raw["area_code"] = str(new_area_code).strip()
        try:
            acct.creds_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to write file: {exc}")
        pool.reload_account(account_id)
        return {"status": "ok", "account_id": account_id}

    raise HTTPException(status_code=500, detail="Invalid credential file format")


@router.post("/api/deepseek/login/{account_id}")
async def api_deepseek_login(
    request: Request,
    account_id: str,
    _=Depends(get_admin_user),
) -> dict[str, Any]:
    """Trigger immediate login for a DeepSeek account."""
    pool: AccountPool = request.app.state.pool
    acct = pool.get_account(account_id)
    if not acct:
        raise HTTPException(status_code=404, detail="Account not found")
    if acct.provider != "deepseek":
        raise HTTPException(status_code=400, detail="Not a DeepSeek account")

    deepseek = getattr(request.app.state, "deepseek", None)
    if not deepseek:
        raise HTTPException(status_code=500, detail="DeepSeek provider not initialized")

    runtime = await deepseek._get_runtime(account_id)
    if not runtime:
        raise HTTPException(status_code=500, detail="Runtime not found")

    from ..providers.deepseek.auth import login

    result = await login(
        request.app.state.http_client,
        email=runtime.account.email,
        password=runtime.account.password,
        mobile=runtime.account.mobile,
        area_code=runtime.account.area_code,
    )
    if result:
        runtime.account.access_token = result["token"]
        runtime.account.refresh_token = result.get("refresh_token", "")
        runtime.account.last_login_at = time.time()
        runtime.account.last_error = None
        # Sync back to pool
        acct.access_token = runtime.account.access_token
        acct.refresh_token = runtime.account.refresh_token
        # Persist to file
        try:
            raw = json.loads(acct.creds_file.read_text(encoding="utf-8"))
            raw["access_token"] = runtime.account.access_token
            raw["refresh_token"] = runtime.account.refresh_token
            acct.creds_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        except Exception:
            pass
        return {"status": "ok", "account_id": account_id, "detail": "Login succeeded"}
    else:
        return {"status": "failed", "account_id": account_id, "detail": "Login failed"}


# ── Auto-refresh ────────────────────────────────────────────

@router.get("/api/auto-refresh/status")
async def api_auto_refresh_status(request: Request, _=Depends(get_admin_user)) -> dict[str, Any]:
    refresher: AutoRefresher = request.app.state.auto_refresher
    return {
        "config": refresher.get_config(),
    }


@router.post("/api/auto-refresh/run")
async def api_auto_refresh_run(request: Request, _=Depends(get_admin_user)) -> dict[str, str]:
    """Trigger a manual auto-refresh cycle."""
    refresher: AutoRefresher = request.app.state.auto_refresher
    result = await refresher.run_once()
    return {"status": "ok", "detail": result}


# ── Upload credential ────────────────────────────────────────────

_MAX_UPLOAD_BYTES = 64 * 1024  # 64 KB

# Sanitize: allow alphanumeric, dash, underscore, dot. Strip path separators.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")


def _get_unique_path(base_dir: Path, name: str) -> tuple[Path, str]:
    """Return a unique Path by appending _2, _3 etc. if file exists.
    
    Returns (Target_Path, Action_String). Action is 'new' or 'renamed'.
    """
    p = base_dir / name
    if not p.exists():
        return p, "new"
    
    stem = p.stem
    suffix = p.suffix
    counter = 2
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_p = base_dir / new_name
        if not new_p.exists():
            return new_p, "renamed"
        counter += 1


def _detect_credential_provider(data: dict[str, Any]) -> str | None:
    if isinstance(data.get("authToken"), str) and data.get("authToken"):
        return "freebuff"
    if isinstance(data.get("default"), dict) and isinstance(data["default"].get("authToken"), str) and data["default"].get("authToken"):
        return "freebuff"
    if isinstance(data.get("email"), str) and data.get("email") and isinstance(data.get("password"), str) and data.get("password"):
        return "deepseek"
    if data.get("refresh_token") or data.get("access_token"):
        return "qwen"
    return None


@router.post("/api/upload")
async def api_upload(request: Request, file: UploadFile, _=Depends(get_admin_user)) -> dict[str, Any]:
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

    # 4. Detect provider and normalize defaults
    provider = _detect_credential_provider(data)
    if provider is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported credential file. Expected Qwen OAuth "
                "(refresh_token/access_token), Freebuff credentials (authToken), "
                "or DeepSeek credentials (email/password)."
            ),
        )

    # 5. Ensure sensible defaults
    if provider == "qwen":
        data.setdefault("access_token", "")
        data.setdefault("token_type", "Bearer")
        data.setdefault("resource_url", "https://portal.qwen.ai/v1")
        if not data.get("expiry_date"):
            data["expiry_date"] = int(time.time() * 1000) + 7 * 24 * 3600 * 1000  # 7 days
    elif provider == "deepseek":
        data.setdefault("access_token", "")
        data.setdefault("refresh_token", "")

    # 6. Find unique filename
    pool: AccountPool = request.app.state.pool
    settings.creds_dir.mkdir(parents=True, exist_ok=True)
    target, action = _get_unique_path(settings.creds_dir, name)
    write_succeeded = True
    try:
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except PermissionError:
        write_succeeded = False
        logging.getLogger("gx_qwen2api").warning(
            "Upload: cannot write %s (permission denied). "
            "Credential will be registered in-memory only.",
            target,
        )
    except OSError as exc:
        write_succeeded = False
        logging.getLogger("gx_qwen2api").warning(
            "Upload: cannot write %s (%s). "
            "Credential will be registered in-memory only.",
            target, exc,
        )

    account_id = target.stem

    # 7. Register in pool (even if disk write failed)
    if write_succeeded and target.exists():
        state = pool._try_load_account(account_id, target)
    else:
        # Create an in-memory account from the uploaded data
        qwen_rt = data.get("refresh_token", "")
        freebuff_token = str(data.get("authToken") or data.get("default", {}).get("authToken", ""))
        ds_email = data.get("email", "")
        ds_password = data.get("password", "")
        state = AccountState(
            account_id=account_id,
            creds_file=target,
            provider=provider,
            enabled=True,
            access_token=data.get("access_token", "") if provider in ("qwen", "deepseek") else freebuff_token,
            expiry_date=data.get("expiry_date", 0) if provider == "qwen" else 0,
            refresh_token_hash=(
                hashlib.sha256(qwen_rt.encode()).hexdigest()[:8]
                if provider == "qwen" and qwen_rt
                else hashlib.sha256(freebuff_token.encode()).hexdigest()[:8] if freebuff_token else ""
            ),
            email=ds_email,
            password=ds_password,
            last_write_persisted=False,
            last_write_error="Permission denied or write failed during upload",
            _raw_creds=data,
        )
        state.update_health()

    if state:
        pool.accounts[account_id] = state
    else:
        pool.scan()

    if not write_succeeded:
        action = "memory_only"

    logging.getLogger("gx_qwen2api").info(
        "Uploaded credential %s (%s)%s",
        account_id, action,
        "" if write_succeeded else " — in-memory only",
    )

    event_logger._emit(
        logging.INFO, "upload",
        {
            "account_id": account_id,
            "action": action,
            "detail": f"{'New' if action == 'new' else 'Renamed'} credential: {account_id}"
            + ("" if write_succeeded else " (in-memory only)"),
        },
    )

    return {
        "status": "ok",
        "account_id": account_id,
        "saved_filename": target.name,
        "action": action,
        "persisted": write_succeeded,
    }
