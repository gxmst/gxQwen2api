# Project Code Issues & Potential Bugs Summary

This document summarizes the identified logic-level risks and architectural improvements in the `gxQwen2api` project.

---

## 1. Authentication Error Logic (`models.py`)
- **Issue**: The `is_auth_error` function treats HTTP **400 (Bad Request)** and **504 (Gateway Timeout)** as authentication failures.
- **Impact**: 400 is often a validation error, and 504 is an upstream timeout. Treating them as auth errors triggers unnecessary token refreshes and account cooldowns.

## 2. Streaming Response Reliability (`routes/chat.py`)
- **Issue**: Error handling inside the `generate()` loop of `StreamingResponse` is minimal.
- **Impact**: If the upstream connection drops or returns an error midway, the proxy closes the connection without sending an error JSON, leaving the client in the dark.

## 3. Round-Robin Drift (`account_pool.py`)
- **Issue**: `select_account` filters the `eligible` accounts list on every call while maintaining a global `_rr_index`.
- **Impact**: If the eligible list changes size between requests, the index may skip accounts or hit the same one twice.

## 4. Admin Security & Refactoring (`routes/admin.py`)
- **Issue**: Admin routes use manual `_check_admin` calls instead of FastAPI's `Depends` pattern.
- **Impact**: Code redundancy and harder to maintain/audit.

## 5. Token Refresh Concurrency (`auth.py`)
- **Issue**: Uses a simple dict of booleans for locking.
- **Impact**: Lack of robustness compared to `asyncio.Lock`.

## 6. Code Quality (Imports)
- **Issue**: Redundant nested imports (e.g., `import hashlib` inside functions).
- **Impact**: Slight performance hit and reduced readability.
