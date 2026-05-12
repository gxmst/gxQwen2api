"""Multi-account credential pool with scanning, round-robin, cooldown."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .config import settings


class AccountHealthStatus(str, Enum):
    """Detailed health status for an account.

    Priority order (worst to best):
    disabled > auth_error > refresh_failed > cooldown > expired > expiring_soon > valid
    """
    VALID = "valid"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    COOLDOWN = "cooldown"
    REFRESH_FAILED = "refresh_failed"
    AUTH_ERROR = "auth_error"
    DISABLED = "disabled"
    NO_TOKEN = "no_token"
    UNKNOWN = "unknown"


@dataclass
class AccountState:
    """Runtime state for a single account."""

    account_id: str
    creds_file: Path
    provider: str = "qwen"
    source_slot: str = ""
    enabled: bool = True
    # Token state
    access_token: str = ""
    expiry_date: int = 0
    refresh_token_hash: str = ""
    # DeepSeek credentials
    email: str = ""
    password: str = ""
    mobile: str = ""
    area_code: str = ""
    device_id: str = ""
    # Health
    error_count: int = 0
    cooldown_until: float = 0.0
    last_refresh: str = ""
    last_error: str = ""
    requests_count: int = 0
    # Rate limiting
    rate_limit_count: int = 0
    last_rate_limit_at: float = 0.0
    last_success_at: float = 0.0
    # Detailed health status
    health_status: AccountHealthStatus = AccountHealthStatus.UNKNOWN
    status_reason: str = ""  # i18n key (prefixed with 'reason_')
    status_reason_params: dict[str, Any] = field(default_factory=dict)
    last_auth_error: str = ""
    # None means "refresh never attempted", False = failed, True = succeeded
    last_refresh_success: bool | None = None
    last_auth_check_at: float = 0.0  # timestamp of last explicit verify
    last_auth_success_at: float = 0.0
    last_auth_failure_at: float = 0.0
    # Auto-refresh tracking
    last_auto_refresh_at: float = 0.0
    last_auto_refresh_result: str = ""  # "success", "failed", "skipped", ""
    last_auto_refresh_error: str = ""
    # Persistence tracking
    last_write_persisted: bool = True
    last_write_error: str = ""
    _raw_creds: dict[str, Any] | None = None

    @property
    def is_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def token_valid(self) -> bool:
        if self.provider == "freebuff":
            return bool(self.access_token)
        if self.provider == "deepseek":
            return bool(self.access_token) or ((bool(self.email) or bool(self.mobile)) and bool(self.password))
        if not self.access_token or not self.expiry_date:
            return False
        return (
            time.time() * 1000
            < self.expiry_date - settings.token_refresh_buffer_s * 1000
        )

    @property
    def has_credential(self) -> bool:
        if self.provider == "freebuff":
            return bool(self.access_token)
        return bool(self.refresh_token_hash)

    def compute_health(self) -> AccountHealthStatus:
        """Compute the detailed health status based on multiple factors.

        status_reason is set to an i18n key (prefixed with 'reason_') that
        the dashboard frontend translates. Dynamic values (counts, durations)
        are passed via status_reason_params as a dict.
        """
        if not self.enabled:
            self.status_reason = "reason_disabled"
            self.status_reason_params = {}
            return AccountHealthStatus.DISABLED

        if self.last_auth_error:
            self.status_reason = "reason_auth_error"
            self.status_reason_params = {"error": self.last_auth_error[:100]}
            return AccountHealthStatus.AUTH_ERROR

        # Only mark refresh_failed if refresh was actually attempted and failed
        # Don't penalize accounts that just haven't been refreshed yet
        if self.last_refresh_success is False and not self.token_valid:
            self.status_reason = "reason_refresh_failed"
            self.status_reason_params = {"error": self.last_error[:100]}
            return AccountHealthStatus.REFRESH_FAILED

        if self.is_cooldown:
            remaining = max(0, int(self.cooldown_until - time.time()))
            # If rate_limited or quota_exhausted status is current, use that reason
            if self.health_status == AccountHealthStatus.RATE_LIMITED:
                self.status_reason = "reason_rate_limited"
            elif self.health_status == AccountHealthStatus.QUOTA_EXHAUSTED:
                self.status_reason = "reason_quota_exhausted"
            else:
                self.status_reason = "reason_cooldown"
            
            self.status_reason_params = {"seconds": remaining}
            return self.health_status if self.health_status in (
                AccountHealthStatus.RATE_LIMITED, 
                AccountHealthStatus.QUOTA_EXHAUSTED, 
                AccountHealthStatus.COOLDOWN
            ) else AccountHealthStatus.COOLDOWN

        if self.provider == "freebuff":
            self.status_reason = "reason_valid" if self.access_token else "reason_no_token"
            self.status_reason_params = {}
            return AccountHealthStatus.VALID if self.access_token else AccountHealthStatus.NO_TOKEN

        if self.provider == "deepseek":
            if self.access_token:
                self.status_reason = "reason_valid"
                self.status_reason_params = {}
                return AccountHealthStatus.VALID
            if (self.email or self.mobile) and self.password:
                self.status_reason = "reason_no_token"
                self.status_reason_params = {}
                return AccountHealthStatus.NO_TOKEN
            self.status_reason = "reason_no_credentials"
            self.status_reason_params = {}
            return AccountHealthStatus.NO_TOKEN

        if not self.access_token or not self.expiry_date:
            self.status_reason = "reason_no_token"
            self.status_reason_params = {}
            return AccountHealthStatus.NO_TOKEN

        now_ms = time.time() * 1000
        buffer_ms = settings.token_refresh_buffer_s * 1000

        if now_ms >= self.expiry_date:
            self.status_reason = "reason_expired"
            self.status_reason_params = {}
            return AccountHealthStatus.EXPIRED

        if now_ms >= self.expiry_date - buffer_ms:
            mins_left = max(0, (self.expiry_date - now_ms) / 60000)
            self.status_reason = "reason_expiring_soon"
            self.status_reason_params = {"minutes": int(mins_left)}
            return AccountHealthStatus.EXPIRING_SOON

        self.status_reason = "reason_valid"
        self.status_reason_params = {}
        return AccountHealthStatus.VALID

    def update_health(self) -> None:
        """Update the health_status field based on current state."""
        self.health_status = self.compute_health()

    def mark_auth_error(self, error_message: str) -> None:
        """Mark this account as having an auth error."""
        self.last_auth_error = error_message[:500]
        self.last_auth_failure_at = time.time()
        self.last_auth_check_at = time.time()
        self.health_status = AccountHealthStatus.AUTH_ERROR
        self.status_reason = "reason_auth_error"
        self.status_reason_params = {"error": error_message[:100]}
        self.record_error(f"Auth error: {error_message}")

    def clear_auth_error(self) -> None:
        """Clear auth error state (e.g., after successful refresh)."""
        self.last_auth_error = ""
        if self.health_status == AccountHealthStatus.AUTH_ERROR:
            self.update_health()

    def mark_refresh_failed(self, error_message: str = "") -> None:
        """Mark token refresh as failed."""
        self.last_refresh_success = False
        self.last_auth_failure_at = time.time()
        if not self.last_auth_error:  # Don't override auth errors
            self.health_status = AccountHealthStatus.REFRESH_FAILED
            self.status_reason = "reason_refresh_failed"
            self.status_reason_params = {"error": error_message[:100]}
            if error_message:
                self.last_error = error_message[:500]

    def mark_refresh_success(self) -> None:
        """Mark token refresh as successful."""
        self.last_refresh_success = True
        self.last_auth_success_at = time.time()
        self.clear_auth_error()
        self.update_health()

    def mark_rate_limited(self, error_message: str = "", cooldown_s: int = 60) -> None:
        """Mark account as rate limited."""
        self.rate_limit_count += 1
        self.last_rate_limit_at = time.time()
        self.cooldown_until = time.time() + cooldown_s
        self.health_status = AccountHealthStatus.RATE_LIMITED
        self.status_reason = "reason_rate_limited"
        self.status_reason_params = {"seconds": int(cooldown_s), "error": error_message[:100]}
        if error_message:
            self.last_error = error_message[:500]

    def mark_quota_exhausted(self, error_message: str = "", duration_s: int = 14400) -> None:
        """Mark account as quota exhausted (daily quota reached).
        
        Default duration is 4 hours (14400s), which is conservative.
        """
        self.cooldown_until = time.time() + duration_s
        self.health_status = AccountHealthStatus.QUOTA_EXHAUSTED
        self.status_reason = "reason_quota_exhausted"
        self.status_reason_params = {"seconds": int(duration_s), "error": error_message[:100]}
        if error_message:
            self.last_error = error_message[:500]

    @property
    def token_status(self) -> str:
        """Legacy compatibility - returns health_status as string."""
        return self.health_status.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "creds_file": str(self.creds_file),
            "provider": self.provider,
            "source_slot": self.source_slot or None,
            "enabled": self.enabled,
            "token_status": self.token_status,
            "health_status": self.health_status.value,
            "status_reason": self.status_reason,
            "status_reason_params": self.status_reason_params,
            "expires_in_minutes": max(0, (self.expiry_date - time.time() * 1000) / 60000) if self.expiry_date else None,
            "expiry_date": self.expiry_date or None,
            "access_token_loaded": bool(self.access_token),
            "refresh_token_hash_prefix": self.refresh_token_hash,
            "error_count": self.error_count,
            "in_cooldown": self.is_cooldown,
            "cooldown_until": self.cooldown_until or None,
            "last_refresh": self.last_refresh,
            "last_error": self.last_error,
            "last_auth_error": self.last_auth_error,
            "requests_count": self.requests_count,
            "last_refresh_success": self.last_refresh_success,
            "last_auth_check_at": self.last_auth_check_at or None,
            "last_auth_success_at": self.last_auth_success_at or None,
            "last_auth_failure_at": self.last_auth_failure_at or None,
            # Auto-refresh fields
            "last_auto_refresh_at": self.last_auto_refresh_at or None,
            "last_auto_refresh_result": self.last_auto_refresh_result or None,
            "last_auto_refresh_error": self.last_auto_refresh_error or None,
            "last_write_persisted": self.last_write_persisted,
            "last_write_error": self.last_write_error or None,
            "has_refresh_token": bool(self._raw_creds and self._raw_creds.get("refresh_token")),
            "has_credential": self.has_credential,
            "email": self.email or None,
            "rate_limit_count": self.rate_limit_count,
            "last_rate_limit_at": self.last_rate_limit_at or None,
            "last_success_at": self.last_success_at or None,
            "device_id": self.device_id or None,
        }

    def record_error(self, message: str, cooldown_s: int = 60) -> None:
        self.error_count += 1
        self.last_error = message[:500]
        self.cooldown_until = time.time() + cooldown_s

    def record_success(self) -> None:
        self.error_count = 0
        self.cooldown_until = 0.0
        self.requests_count += 1
        self.last_success_at = time.time()
        self.mark_refresh_success()


class AccountPool:
    """Manages multiple credential files with round-robin selection."""

    def __init__(self) -> None:
        self.accounts: dict[str, AccountState] = {}
        self.last_success_account_id = ""
        self._last_scan_time = 0.0
        self._explicitly_disabled: set[str] = set()

    # ── Scanning ──────────────────────────────────────────────────

    def scan(self) -> None:
        """Scan creds_dir for *.json files and register new accounts."""
        self._last_scan_time = time.time()
        creds_dir = settings.creds_dir
        if not creds_dir.is_dir():
            return

        found_files: set[str] = set()
        for f in creds_dir.glob("*.json"):
            states = self._load_accounts_from_file(f)
            for state in states:
                aid = state.account_id
                found_files.add(aid)
                existing = self.accounts.get(aid)
                if existing:
                    state.enabled = existing.enabled
                    state.error_count = existing.error_count
                    state.cooldown_until = existing.cooldown_until
                    state.requests_count = existing.requests_count
                    state.rate_limit_count = existing.rate_limit_count
                    state.last_rate_limit_at = existing.last_rate_limit_at
                    state.last_error = existing.last_error
                    state.last_refresh_success = existing.last_refresh_success
                    state.last_auth_check_at = existing.last_auth_check_at
                    state.last_auth_success_at = existing.last_auth_success_at
                    state.last_auth_failure_at = existing.last_auth_failure_at
                    state.last_auto_refresh_at = existing.last_auto_refresh_at
                    state.last_auto_refresh_result = existing.last_auto_refresh_result
                    state.last_auto_refresh_error = existing.last_auto_refresh_error
                    state.last_write_persisted = existing.last_write_persisted
                    state.last_write_error = existing.last_write_error
                self.accounts[aid] = state

        # Remove accounts whose files disappeared
        for aid in list(self.accounts):
            if aid not in found_files and aid not in self._explicitly_disabled:
                pass  # Keep in-memory state even if file gone

    def _load_accounts_from_file(self, path: Path) -> list[AccountState]:
        """Parse a credential file into one or more AccountState entries."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        # Freebuff auth-tokens.json format: {"tokens": [{"authToken": ...}, ...]}
        if isinstance(raw.get("tokens"), list):
            states: list[AccountState] = []
            valid_tokens = [
                token for token in raw["tokens"]
                if isinstance(token, dict) and isinstance(token.get("authToken"), str) and token.get("authToken")
            ]
            for idx, token_data in enumerate(valid_tokens, start=1):
                suffix = "" if len(valid_tokens) == 1 else f"_{idx}"
                account_id = f"{path.stem}{suffix}"
                state = self._build_account_state(account_id, path, {"authToken": token_data["authToken"], **token_data}, source_slot=str(idx))
                if state:
                    states.append(state)
            return states

        state = self._build_account_state(path.stem, path, raw)
        return [state] if state else []

    def _try_load_account(self, account_id: str, path: Path) -> AccountState | None:
        """Backward-compatible single-account loader."""
        for state in self._load_accounts_from_file(path):
            if state.account_id == account_id:
                return state
        return None

    def _build_account_state(
        self,
        account_id: str,
        path: Path,
        raw: dict[str, Any],
        *,
        source_slot: str = "",
    ) -> AccountState | None:
        """Build one AccountState from normalized credential data."""

        provider = "qwen"
        access_token = ""
        expiry_date = 0
        refresh_token_hash = ""

        email = ""
        password = ""
        mobile = ""
        area_code = ""

        # Freebuff / Codebuff local credentials
        if isinstance(raw.get("authToken"), str) and raw.get("authToken"):
            provider = "freebuff"
            access_token = raw["authToken"]
            refresh_token_hash = hashlib.sha256(access_token.encode()).hexdigest()[:8]
        elif (
            isinstance(raw.get("default"), dict)
            and isinstance(raw["default"].get("authToken"), str)
            and raw["default"].get("authToken")
        ):
            provider = "freebuff"
            access_token = raw["default"]["authToken"]
            refresh_token_hash = hashlib.sha256(access_token.encode()).hexdigest()[:8]
        elif (isinstance(raw.get("email"), str) and raw.get("email") and isinstance(raw.get("password"), str) and raw.get("password")) or (isinstance(raw.get("mobile"), str) and raw.get("mobile") and isinstance(raw.get("password"), str) and raw.get("password")):
            # DeepSeek email/password or mobile/password credentials
            provider = "deepseek"
            email = raw.get("email", "")
            password = raw.get("password", "")
            mobile = raw.get("mobile", "")
            area_code = raw.get("area_code", "")
            access_token = raw.get("access_token", "")
            refresh_token_hash = hashlib.sha256(password.encode()).hexdigest()[:8]
            # Per-account device_id: load existing or generate a new one
            device_id_val = raw.get("device_id", "")
            if not device_id_val:
                device_id_val = uuid.uuid4().hex
                raw["device_id"] = device_id_val
                try:
                    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    pass
        else:
            # Validate qwen-like credentials
            if not raw.get("refresh_token") and not raw.get("access_token"):
                return None
            rt = raw.get("refresh_token", "")
            access_token = raw.get("access_token", "")
            expiry_date = raw.get("expiry_date", 0)
            refresh_token_hash = hashlib.sha256(rt.encode()).hexdigest()[:8] if rt else ""

        state = AccountState(
            account_id=account_id,
            creds_file=path,
            provider=provider,
            source_slot=source_slot,
            enabled=account_id not in self._explicitly_disabled,
            access_token=access_token,
            expiry_date=expiry_date,
            refresh_token_hash=refresh_token_hash,
            email=email,
            password=password,
            mobile=mobile,
            area_code=area_code,
            device_id=raw.get("device_id", ""),
            _raw_creds=raw,
        )

        if provider == "qwen" and raw.get("expiry_date"):
            dt = datetime.datetime.fromtimestamp(
                raw["expiry_date"] / 1000, tz=datetime.timezone.utc
            )
            state.last_refresh = dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Compute initial health
        state.update_health()

        return state

    # ── Credential (re)loading ────────────────────────────────────

    def reload_account(self, account_id: str) -> bool:
        """Re-read a specific account's creds file from disk."""
        acct = self.accounts.get(account_id)
        if not acct:
            return False
        new_state = None
        for state in self._load_accounts_from_file(acct.creds_file):
            if state.account_id == account_id:
                new_state = state
                break
        if new_state:
            # Preserve runtime counters and auth state
            new_state.error_count = acct.error_count
            new_state.cooldown_until = acct.cooldown_until
            new_state.requests_count = acct.requests_count
            new_state.rate_limit_count = acct.rate_limit_count
            new_state.last_rate_limit_at = acct.last_rate_limit_at
            new_state.last_error = acct.last_error
            new_state.enabled = acct.enabled
            new_state.provider = acct.provider
            # Preserve auth state across reloads (but NOT last_auth_error to allow recovery)
            new_state.last_refresh_success = acct.last_refresh_success
            new_state.last_auth_check_at = acct.last_auth_check_at
            new_state.last_auth_success_at = acct.last_auth_success_at
            new_state.last_auth_failure_at = acct.last_auth_failure_at
            # Preserve auto-refresh state
            new_state.last_auto_refresh_at = acct.last_auto_refresh_at
            new_state.last_auto_refresh_result = acct.last_auto_refresh_result
            new_state.last_auto_refresh_error = acct.last_auto_refresh_error
            # Re-compute health after restoring state (may change if token got updated on disk)
            new_state.update_health()
            self.accounts[account_id] = new_state
            return True
        return False

    def check_mtime_and_reload(self, account_id: str) -> None:
        """If creds file was modified since last read, reload it."""
        acct = self.accounts.get(account_id)
        if not acct:
            return
        try:
            mtime = acct.creds_file.stat().st_mtime
            last_access = self._last_scan_time
            if mtime > last_access:
                self.reload_account(account_id)
        except OSError:
            pass

    # ── Selection (round-robin) ───────────────────────────────────

    def select_account(
        self,
        exclude_ids: set[str] | None = None,
        provider: str | None = None,
    ) -> AccountState | None:
        """Sequential round-robin selection starting from the last success.
        
        Prioritizes the account following the last successful one.
        """
        # 1. Get stable list of IDs
        all_ids = sorted(self.accounts.keys())
        if not all_ids:
            return None
            
        # 2. Find start index (after last success)
        start_idx = 0
        if self.last_success_account_id in self.accounts:
            try:
                # Find current anchor index
                current_idx = all_ids.index(self.last_success_account_id)
                start_idx = (current_idx + 1) % len(all_ids)
            except ValueError:
                pass
                
        # 3. Iterate sequentially from start_idx
        for i in range(len(all_ids)):
            idx = (start_idx + i) % len(all_ids)
            aid = all_ids[idx]
            a = self.accounts[aid]
            
            # Skip logic
            if provider and a.provider != provider: continue
            if not a.enabled: continue
            if a.health_status == AccountHealthStatus.AUTH_ERROR: continue
            if a.is_cooldown: continue
            if not a.has_credential: continue
            if exclude_ids and a.account_id in exclude_ids: continue
            
            return a
            
        return None

    # ── Admin operations ──────────────────────────────────────────

    def enable_account(self, account_id: str) -> bool:
        acct = self.accounts.get(account_id)
        if not acct:
            return False
        acct.enabled = True
        self._explicitly_disabled.discard(account_id)
        acct.update_health()
        return True

    def disable_account(self, account_id: str) -> bool:
        acct = self.accounts.get(account_id)
        if not acct:
            return False
        acct.enabled = False
        self._explicitly_disabled.add(account_id)
        acct.update_health()
        return True

    def get_account(self, account_id: str) -> AccountState | None:
        return self.accounts.get(account_id)

    def all_accounts(self) -> list[AccountState]:
        return list(self.accounts.values())

    def save_creds(self, account_id: str, raw: dict[str, Any]) -> None:
        """Persist updated credentials to disk."""
        acct = self.accounts.get(account_id)
        if not acct:
            return
        
        try:
            acct.creds_file.write_text(json.dumps(raw, indent=2))
            acct.last_write_persisted = True
            acct.last_write_error = ""
        except PermissionError:
            acct.last_write_persisted = False
            acct.last_write_error = "Permission denied"
            logging.getLogger("gx_qwen2api").warning(
                "Cannot write %s — permission denied. "
                "Token updated in-memory only; changes will not persist across restarts.",
                acct.creds_file,
            )
        except OSError as exc:
            acct.last_write_persisted = False
            acct.last_write_error = str(exc)
            logging.getLogger("gx_qwen2api").warning(
                "Cannot write %s — %s. "
                "Token updated in-memory only.",
                acct.creds_file, exc,
            )
