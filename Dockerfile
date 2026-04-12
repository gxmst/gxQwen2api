# ---
# Stage 1: build
# ---

FROM python:3.12-slim-bookworm AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

RUN pip install uv
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project

COPY src/gx_qwen2api/ ./gx_qwen2api/
RUN uv sync --frozen

# ---
# Stage 2: runtime
# ---

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends gosu && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 999 nonroot \
    && useradd --system --gid 999 --uid 999 --create-home nonroot

COPY --from=builder --chown=nonroot:nonroot /app /app

RUN mkdir -p /app/data/creds && chown -R nonroot:nonroot /app/data/creds

ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE ${PORT:-31998}

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-31998}/healthz')" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "gx_qwen2api.main"]
