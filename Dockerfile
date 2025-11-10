# Multi-stage Dockerfile for AutoVoice
# Stage 1: Builder - Compile CUDA extensions and create wheel
# Digest retrieved via: docker inspect nvidia/cuda:12.1.0-devel-ubuntu22.04 (2025-01-29)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04@sha256:e3a8f7b933e77ecee74731198a2a5483e965b585cea2660675cf4bb152237e9b AS builder

# Build arguments for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=dev

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="70;80;86;89" \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    cmake \
    ninja-build \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m pip install --upgrade pip virtualenv && \
    python3.10 -m virtualenv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first for better caching
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code
WORKDIR /app
COPY . .

# Build CUDA extensions
RUN python setup.py build_ext --inplace && \
    python setup.py bdist_wheel

# Stage 2: Runtime - Smaller image with only runtime dependencies
# Digest retrieved via: docker inspect nvidia/cuda:12.1.0-runtime-ubuntu22.04 (2025-01-29)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04@sha256:402700b179eb764da6d60d99fe106aa16c36874f7d7fb3e122251ff6aea8b2f7 AS runtime

# Build arguments for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=dev

# Add labels
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="AutoVoice" \
      org.opencontainers.image.description="GPU-accelerated voice synthesis system with real-time processing" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="AutoVoice" \
      org.opencontainers.image.source="https://github.com/autovoice/autovoice"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PYTHONPATH=/app/src \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libsndfile1 \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r autovoice && \
    useradd -r -g autovoice -s /bin/bash -d /app autovoice

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code and built extensions
WORKDIR /app
COPY --from=builder /app /app

# Set proper ownership
RUN chown -R autovoice:autovoice /app && \
    mkdir -p /app/logs /app/data /app/models && \
    chown -R autovoice:autovoice /app/logs /app/data /app/models

# Switch to non-root user
USER autovoice

# Expose ports
EXPOSE 5000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Set working directory
WORKDIR /app

# Default command
CMD ["python3", "main.py"]
