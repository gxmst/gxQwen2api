#!/bin/sh
set -e

# Ensure CREDS_DIR is readable.
# We intentionally avoid recursive chown here because CREDS_DIR may be a
# host bind mount — mutating its ownership would break host-side tooling
# (qwen login, manual edits, etc).
# Instead, we only ensure the directory itself is accessible.
# If token persistence fails due to permissions, the app gracefully
# falls back to in-memory-only mode (logs a warning, continues running).
if [ -n "$CREDS_DIR" ] && [ -d "$CREDS_DIR" ]; then
  # Make sure the directory is at least readable (non-recursive)
  if [ ! -r "$CREDS_DIR" ]; then
    echo "WARNING: CREDS_DIR ($CREDS_DIR) is not readable. Token loading may fail." >&2
  fi
fi

# Drop to non-root user and exec the CMD
exec gosu nonroot "$@"
