# Multi-stage build for AutoVoice on Jetson Thor
# Platform: aarch64, CUDA-capable NVIDIA runtime, SM 11.0
#
# NVIDIA's current Thor Docker docs use the generic PyTorch NGC container
# rather than the older l4t-pytorch tags. Keep this overridable for future
# JetPack/container revisions and local testing.
ARG AUTOVOICE_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3@sha256:b70bc3ff73fae58e7ec326849b4726eac04200d9ec1da2b385f94c9f7775971f

# Base stage: CUDA-enabled PyTorch runtime for Jetson Thor
FROM ${AUTOVOICE_BASE_IMAGE} AS base

LABEL maintainer="AutoVoice Team"
LABEL version="0.1.0"
LABEL description="AutoVoice - GPU-accelerated singing voice conversion for Jetson Thor"

WORKDIR /app

# System dependencies (minimal layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    sox \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python dependencies (cached layer)
COPY requirements.txt .
# Keep the NVIDIA base image's bundled torch/torchaudio pair intact. Upgrading
# just one side of that pair breaks binary compatibility on startup. Some
# transitive packages still pull in a stock torchaudio wheel, so remove it and
# rely on the app's soundfile/librosa fallbacks inside the container.
RUN grep -Ev '^(torch|torchaudio)([<>=!~].*)?$' requirements.txt > /tmp/requirements.docker.txt && \
    pip install --no-cache-dir -r /tmp/requirements.docker.txt && \
    pip uninstall -y torchaudio || true

# Frontend build stage
FROM node:20-slim@sha256:3d0f05455dea2c82e2f76e7e2543964c30f6b7d673fc1a83286736d44fe4c41c AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Production stage
FROM base AS production

# Copy built frontend assets
COPY --from=frontend /app/frontend/dist /app/static

# Copy application source
COPY src/ src/
COPY config/ config/
COPY main.py .
COPY setup.py .

# Install package (editable for development flexibility) without re-resolving
# dependencies; the image already installed the curated dependency set above.
RUN pip install --no-cache-dir --no-deps -e .

# Create necessary directories
RUN mkdir -p /app/models/pretrained \
    /app/data/voice_profiles \
    /app/data/uploads \
    /app/data/outputs \
    /app/logs

# Non-root user for security. Use a system account so we do not collide with
# pre-existing UIDs in NVIDIA's base image.
RUN if ! getent group autovoice >/dev/null; then groupadd --system autovoice; fi && \
    if ! id -u autovoice >/dev/null 2>&1; then useradd --system --create-home --gid autovoice autovoice; fi && \
    chown -R autovoice:autovoice /app
USER autovoice

# Environment configuration
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    LOG_LEVEL=info \
    CUDA_HOME=/usr/local/cuda-13.0 \
    TORCH_CUDA_ARCH_LIST="11.0"

# Expose ports
EXPOSE 5000

# Health check (readiness probe)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ready || exit 1

# Graceful shutdown timeout
STOPSIGNAL SIGTERM

# Default command
CMD ["autovoice", "serve", "--host", "0.0.0.0", "--port", "5000"]
