#!/bin/sh
set -e

# Ensure CREDS_DIR is writable by nonroot user.
# If it's a bind mount, we attempt to fix ownership.
if [ -n "$CREDS_DIR" ] && [ -d "$CREDS_DIR" ]; then
  chown -R nonroot:nonroot "$CREDS_DIR"
fi

# Drop to non-root user and exec the CMD
exec gosu nonroot "$@"
