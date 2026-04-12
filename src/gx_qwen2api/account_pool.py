"""Multi-account credential pool with scanning, round-robin, cooldown."""

from __future__ import annotations

import hashlib
import json
import logging
import time
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
    enabled: bool = True
    # Token state
    access_token: str = ""
    expiry_date: int = 0
    refresh_token_hash: str = ""
    # Health
    error_count: int = 0
    cooldown_until: float = 0.0
    last_refresh: str = ""
    last_error: str = ""
    requests_count: int = 0
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
        if not self.access_token or not self.expiry_date:
            return False
        return (
            time.time() * 1000
            < self.expiry_date - settings.token_refresh_buffer_s * 1000
        )

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
            self.status_reason = "reason_cooldown"
            self.status_reason_params = {"seconds": remaining}
            return AccountHealthStatus.COOLDOWN

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

    @property
    def token_status(self) -> str:
        """Legacy compatibility - returns health_status as string."""
        return self.health_status.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "creds_file": str(self.creds_file),
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
        }

    def record_error(self, message: str, cooldown_s: int = 60) -> None:
        self.error_count += 1
        self.last_error = message[:500]
        self.cooldown_until = time.time() + cooldown_s

    def record_success(self) -> None:
        self.error_count = 0
        self.cooldown_until = 0.0
        self.requests_count += 1
        self.mark_refresh_success()


class AccountPool:
    """Manages multiple credential files with round-robin selection."""

    def __init__(self) -> None:
        self.accounts: dict[str, AccountState] = {}
        self._rr_index = 0
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
            aid = f.stem
            found_files.add(aid)
            if aid not in self.accounts:
                state = self._try_load_account(aid, f)
                if state:
                    self.accounts[aid] = state

        # Remove accounts whose files disappeared
        for aid in list(self.accounts):
            if aid not in found_files and aid not in self._explicitly_disabled:
                pass  # Keep in-memory state even if file gone

    def _try_load_account(self, account_id: str, path: Path) -> AccountState | None:
        """Try to parse a creds JSON file into an AccountState."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        # Validate it looks like a qwen creds file
        if not raw.get("refresh_token") and not raw.get("access_token"):
            return None

        rt = raw.get("refresh_token", "")
        rt_hash = hashlib.sha256(rt.encode()).hexdigest()[:8] if rt else ""

        state = AccountState(
            account_id=account_id,
            creds_file=path,
            enabled=account_id not in self._explicitly_disabled,
            access_token=raw.get("access_token", ""),
            expiry_date=raw.get("expiry_date", 0),
            refresh_token_hash=rt_hash,
            _raw_creds=raw,
        )

        if raw.get("expiry_date"):
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
        new_state = self._try_load_account(account_id, acct.creds_file)
        if new_state:
            # Preserve runtime counters and auth state
            new_state.error_count = acct.error_count
            new_state.cooldown_until = acct.cooldown_until
            new_state.requests_count = acct.requests_count
            new_state.last_error = acct.last_error
            new_state.enabled = acct.enabled
            # Preserve auth state across reloads
            new_state.last_auth_error = acct.last_auth_error
            new_state.last_refresh_success = acct.last_refresh_success
            new_state.health_status = acct.health_status
            new_state.status_reason = acct.status_reason
            new_state.status_reason_params = acct.status_reason_params
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

    def select_account(self) -> AccountState | None:
        """Pick the next eligible account via round-robin."""
        eligible = [a for a in self.accounts.values() if a.enabled and not a.is_cooldown and a.refresh_token_hash]
        if not eligible:
            return None

        # Ensure index is within range of currently eligible accounts
        idx = self._rr_index % len(eligible)
        acct = eligible[idx]
        self._rr_index = (idx + 1) % len(eligible)
        return acct

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
