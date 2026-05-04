"""Tests for call_logger, NVIDIA provider, and admin API masking."""

import json
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from gx_qwen2api.call_logger import CallLogger, _mask_key, _mask_error

PASS = 0
FAIL = 0


def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {msg}")
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


# ─────────────────────────────────────────
# Test 1: call_logger ring buffer
# ─────────────────────────────────────────
print("=" * 60)
print("Test 1: CallLogger ring buffer")
print("=" * 60)

cl = CallLogger()
check(cl.MAX_ENTRIES == 500, "default max entries is 500")

cl.log("req-001", "qwen", "acct-a", "coder-model", False, "success", http_status=200, input_tokens=100, output_tokens=200, total_tokens=300, latency_ms=150)
cl.log("req-002", "deepseek", "acct-b", "deepseek-flash", True, "success", http_status=200, latency_ms=80)
cl.log("req-003", "nvidia", "nv-key-1", "nvidia-deepseek-v4-flash", False, "auth_error", http_status=401, error_message="Unauthorized")

logs = cl.get_logs(limit=10)
check(len(logs) == 3, f"3 entries logged, got {len(logs)}")
check(logs[0]["provider"] == "nvidia", "most recent first (nvidia)")
check(logs[0]["status"] == "auth_error", "auth_error status")

# Filter by provider
logs_qwen = cl.get_logs(limit=10, provider="qwen")
check(len(logs_qwen) == 1, "filter by provider qwen")
check(logs_qwen[0]["account_id"] == "acct-a", "correct account id")

# Filter by status
logs_err = cl.get_logs(limit=10, status="auth_error")
check(len(logs_err) == 1, "filter by status")

# Stats
stats = cl.get_stats()
check(stats["total"] == 3, "stats total")
check(stats["by_provider"]["qwen"] == 1, "stats by provider")
check(stats["by_status"]["success"] == 2, "stats by status")

cl.clear()
check(len(cl.get_logs()) == 0, "clear works")

# ─────────────────────────────────────────
# Test 2: API key masking
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 2: API key / token masking")
print("=" * 60)

check(_mask_key("nvapi-abcdef1234567890") == "nvap****7890", "long key masked correctly")
check(_mask_key("sk-abc123") == "sk-a****c123", "short key masked")
check(_mask_key("ab") == "ab****", "very short key masked")

check("****" in _mask_error("Error with Bearer nvapi-secret-key-here"), "error message masks bearer token")
check("****" in _mask_error("Unauthorized with token sk-abcdef"), "sk- token masked in error")

# ─────────────────────────────────────────
# Test 3: NVIDIA key state model
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 3: NVIDIA key state model")
print("=" * 60)

from gx_qwen2api.providers.nvidia.models import NvidiaKeyState, NvidiaKeyStatus, get_builtin_models

ks = NvidiaKeyState(key_id="nvidia_1", name="test-key", api_key="nvapi-abcdef1234567890")
check(ks.masked_key == "nvap****7890", "masked_key property")
check(ks.status == NvidiaKeyStatus.UNKNOWN, "initial status unknown")
check(ks.enabled, "enabled by default")

ks.mark_success()
check(ks.status == NvidiaKeyStatus.VALID, "mark_success sets valid")
check(ks.last_success_at > 0, "last_success_at set")

ks.mark_rate_limited(60)
check(ks.status == NvidiaKeyStatus.RATE_LIMITED, "mark_rate_limited sets rate_limited")
check(ks.is_cooldown, "in cooldown")
check(ks.cooldown_remaining > 0, "has remaining cooldown")

ks2 = NvidiaKeyState(key_id="nvidia_2", api_key="bad-key")
ks2.mark_auth_error("Invalid API key")
check(ks2.status == NvidiaKeyStatus.AUTH_ERROR, "mark_auth_error")
check(ks2.error_count == 1, "error_count incremented")

# to_dict
d = ks2.to_dict()
check(d["masked_key"] == "ba****", "to_dict includes masked key")
check(d["status"] == "auth_error", "to_dict includes status")
check("Invalid" in d.get("last_error", ""), "to_dict includes last_error")

# Builtin models
models = get_builtin_models()
check(len(models) >= 3, f"at least 3 builtin models, got {len(models)}")
check(any(m["upstream"] == "deepseek-ai/deepseek-v4-flash" for m in models), "deepseek v4 flash present")

# ─────────────────────────────────────────
# Test 4: NVIDIA provider key management (no API calls)
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 4: NVIDIA provider key management")
print("=" * 60)

from gx_qwen2api.providers.nvidia.provider import _suggest_local_name

check(_suggest_local_name("deepseek-ai/deepseek-v4-flash") == "nvidia-deepseek-ai-deepseek-v4-flash", "local name derivation")

# ─────────────────────────────────────────
# Test 5: Call logger dedup and token fields
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 5: Call logger edge cases")
print("=" * 60)

cl2 = CallLogger()
cl2.log("req-x", "nvidia", "nv-1", "test-model", True, "success", input_tokens=None, output_tokens=None, total_tokens=None, latency_ms=None)
log = cl2.get_logs(limit=1)[0]
check(log["input_tokens"] is None, "None tokens preserved")
check(log["output_tokens"] is None, "None output preserved")
check(log["latency_ms"] is None, "None latency preserved")
check(log["http_status"] is None, "None http_status preserved")

cl2.log("req-y", "qwen", "q-1", "coder-model", False, "error", http_status=502, error_message="Bad Gateway " + "x" * 500)
log2 = cl2.get_logs(limit=1)[0]
check(len(log2["error_message"] or "") <= 200, "error message truncated to 200 chars")
check("nvapi-" not in (log2["error_message"] or ""), "error message masking")

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
