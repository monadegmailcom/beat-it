# Use a generic Ubuntu base image
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Set non-interactive installation to avoid stuck builds
ENV DEBIAN_FRONTEND=noninteractive

# Install core build tools and dependencies (including g++-13)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && add-apt-repository ppa:savoury1/boost-defaults-1.83 \
    && apt-get update && apt-get install -y \
    build-essential \
    g++-13 \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libboost1.83-dev \
    libboost-json1.83-dev \
    python3-full \
    python3-pip \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Set g++-13 as the default C++ compiler
ENV CXX=g++-13

# --- project setup ---
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY train/requirements.txt /app/train/requirements.txt
COPY train/pytorch_requirements.txt /app/train/pytorch_requirements.txt

# Create a virtual environment and install dependencies
# Venv is added to PATH
ENV PATH="/app/.venv/bin:$PATH"
RUN python3 -m venv .venv && \
    pip install --upgrade pip && \
    pip install -r train/pytorch_requirements.txt --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r train/requirements.txt

# --- LibTorch Configuration ---
# Use the LibTorch included in the python package to ensure architecture compatibility (amd64/arm64)
# Ubuntu 22.04 uses Python 3.10
ENV Torch_DIR=/app/.venv/lib/python3.10/site-packages/torch/share/cmake/Torch
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.10/site-packages/torch/lib:/app/.venv/lib/python3.10/site-packages/torch.libs:${LD_LIBRARY_PATH}"

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
