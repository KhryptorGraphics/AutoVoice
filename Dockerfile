FROM nvcr.io/nvidia/l4t-pytorch:r38.4.0-pth2.11-py3 AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY config/ config/
COPY main.py .
COPY setup.py .

# Install package
RUN pip install --no-cache-dir -e .

# Frontend build stage
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Final stage
FROM base AS production
COPY --from=frontend /app/frontend/dist /app/static

# Models directory (mount at runtime)
RUN mkdir -p /app/models/pretrained

EXPOSE 5000

ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=info

HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "5000"]
