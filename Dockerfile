# Use Ubuntu 24.04 as the base image for modern C++23 support (GCC 13)
FROM ubuntu:24.04

# Set non-interactive installation to avoid stuck builds
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# build-essential: GCC/G++
# cmake: Build system
# libboost-all-dev: Boost libraries
# python3-full: Python 3.12 (default in Ubuntu 24.04) and pip/venv
# unzip, wget, git: Utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libboost-all-dev \
    python3-full \
    python3-pip \
    vim \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# --- project setup ---
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY train/requirements.txt /app/train/requirements.txt

# Create a virtual environment and install dependencies
# Venv is added to PATH
ENV PATH="/app/.venv/bin:$PATH"
RUN python3 -m venv .venv && \
    pip install --upgrade pip && \
    pip install -r train/requirements.txt

# --- LibTorch Configuration ---
# Use the LibTorch included in the python package to ensure architecture compatibility (amd64/arm64)
# Ubuntu 24.04 uses Python 3.12
ENV Torch_DIR=/app/.venv/lib/python3.12/site-packages/torch/share/cmake/Torch
ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/torch/lib:/app/.venv/lib/python3.12/site-packages/torch.libs:$LD_LIBRARY_PATH

# --- Project Build ---
# Copy the rest of the source code
COPY . /app

# Build the C++ project
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j1

# --- Runtime Configuration ---
# Expose TensorBoard port
EXPOSE 6006

# Create a volume mount point for external data/models if needed (optional)
VOLUME ["/app/runs", "/app/models"]

# Default command: Generic interactive shell with venv activated
CMD ["/bin/bash"]
