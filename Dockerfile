# Use NVIDIA CUDA 12.9.0 development base image
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONPATH=/app/src
ENV TORCH_CUDA_ARCH_LIST="80;86;89"
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    libsndfile1 \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    libportaudio2 \
    libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 support (closest to 12.9)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code and setup
COPY . .

# Build CUDA extensions
RUN python setup.py build_ext --inplace

# Expose ports for web interface
EXPOSE 5000 8080

# Default command to run the application
CMD ["python", "main.py"]