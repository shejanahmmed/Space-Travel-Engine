# Multi-Stage Production Dockerfile for Relativistic Space Travel Computational Engine
# Stage 1: Build & Dependency Resolution
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging manifests and install wheels into isolated prefix
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip build wheel && \
    pip install --no-cache-dir --prefix=/install ".[web]"

# Stage 2: Hardened Minimal Runtime
FROM python:3.11-slim AS runtime

# Security: Create non-root system user and group
RUN groupadd -g 10001 science && \
    useradd -u 10001 -g science -s /bin/bash -m science

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy source tree and configuration
COPY --chown=science:science src/ ./src/
COPY --chown=science:science pyproject.toml README.md ./

# Install project package into runtime environment
RUN pip install --no-cache-dir --no-deps -e .

# Prepare runtime ephemeris cache directory with write permissions
RUN mkdir -p /app/data && chown -R science:science /app/data

USER science:science

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

# Automated container healthcheck validating REST API status
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/certificate')" || exit 1

# Default entrypoint starts the FastAPI production server
ENTRYPOINT ["python", "-m", "relativistic_engine.cli", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
