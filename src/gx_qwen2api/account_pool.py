"""Multi-account credential pool with scanning, round-robin, cooldown."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import settings


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

    @property
    def token_status(self) -> str:
        if not self.access_token:
            return "no_token"
        if not self.expiry_date:
            return "unknown"
        minutes_left = (self.expiry_date - time.time() * 1000) / 60000
        if minutes_left < 0:
            return "expired"
        if minutes_left < 30:
            return "expiring_soon"
        return "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "creds_file": str(self.creds_file),
            "enabled": self.enabled,
            "token_status": self.token_status,
            "expires_in_minutes": max(0, (self.expiry_date - time.time() * 1000) / 60000) if self.expiry_date else None,
            "expiry_date": self.expiry_date or None,
            "access_token_loaded": bool(self.access_token),
            "refresh_token_hash_prefix": self.refresh_token_hash,
            "error_count": self.error_count,
            "in_cooldown": self.is_cooldown,
            "cooldown_until": self.cooldown_until or None,
            "last_refresh": self.last_refresh,
            "last_error": self.last_error,
            "requests_count": self.requests_count,
        }

    def record_error(self, message: str, cooldown_s: int = 60) -> None:
        self.error_count += 1
        self.last_error = message[:500]
        self.cooldown_until = time.time() + cooldown_s

    def record_success(self) -> None:
        self.error_count = 0
        self.cooldown_until = 0.0
        self.requests_count += 1


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
            import datetime
            dt = datetime.datetime.fromtimestamp(
                raw["expiry_date"] / 1000, tz=datetime.timezone.utc
            )
            state.last_refresh = dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        return state

    # ── Credential (re)loading ────────────────────────────────────

    def reload_account(self, account_id: str) -> bool:
        """Re-read a specific account's creds file from disk."""
        acct = self.accounts.get(account_id)
        if not acct:
            return False
        new_state = self._try_load_account(account_id, acct.creds_file)
        if new_state:
            # Preserve runtime counters
            new_state.error_count = acct.error_count
            new_state.cooldown_until = acct.cooldown_until
            new_state.requests_count = acct.requests_count
            new_state.last_error = acct.last_error
            new_state.enabled = acct.enabled
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

        for _ in range(len(eligible)):
            self._rr_index %= len(eligible)
            acct = eligible[self._rr_index]
            self._rr_index += 1
            return acct
        return None

    # ── Admin operations ──────────────────────────────────────────

    def enable_account(self, account_id: str) -> bool:
        acct = self.accounts.get(account_id)
        if not acct:
            return False
        acct.enabled = True
        self._explicitly_disabled.discard(account_id)
        return True

    def disable_account(self, account_id: str) -> bool:
        acct = self.accounts.get(account_id)
        if not acct:
            return False
        acct.enabled = False
        self._explicitly_disabled.add(account_id)
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
        acct.creds_file.write_text(json.dumps(raw, indent=2))
