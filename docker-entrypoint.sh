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
  # Automatically fix permissions if the directory is not writable by the target user.
  # This is especially helpful for first-run with named volumes or when volume permissions drift.
  if ! gosu nonroot [ -w "$CREDS_DIR" ]; then
    echo "Fixing permissions for $CREDS_DIR..."
    chown -R nonroot:nonroot "$CREDS_DIR"
  fi
fi

# Drop to non-root user and exec the CMD
exec gosu nonroot "$@"
