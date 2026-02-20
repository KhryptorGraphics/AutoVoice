# Multi-stage build for AutoVoice on Jetson Thor
# Platform: aarch64, CUDA 13.0, SM 11.0

# Base stage: CUDA 13.0 runtime for Jetson Thor
FROM nvcr.io/nvidia/l4t-pytorch:r38.4.0-pth2.11-py3 AS base

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
RUN pip install --no-cache-dir -r requirements.txt

# Frontend build stage
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --only=production
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

# Install package (editable for development flexibility)
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/models/pretrained \
    /app/data/voice_profiles \
    /app/data/uploads \
    /app/data/outputs \
    /app/logs

# Non-root user for security
RUN useradd -m -u 1000 autovoice && \
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

# Health check (liveness probe)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Graceful shutdown timeout
STOPSIGNAL SIGTERM

# Default command
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "5000"]
