# Multi-stage Dockerfile for 24/7 Cloud Hosting (Render / VPS)
# Stage 1: Build & Dependency Stage
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install \
    pytest \
    torch \
    numpy \
    gymnasium \
    stable-retro \
    Pillow \
    opencv-python-headless

# Stage 2: Final Production Stage
FROM python:3.10-slim AS final

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . /app

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/api/telemetry

CMD ["python3", "scripts/watchdog.py"]
