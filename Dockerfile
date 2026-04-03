# Use a generic Ubuntu base image
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Set non-interactive installation to avoid stuck builds
ENV DEBIAN_FRONTEND=noninteractive

# Install core build tools and dependencies (including g++-13 natively on 24.04)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    g++-13 \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libboost-all-dev \
    libboost-json-dev \
    python3-full \
    python3-pip \
    python3-venv \
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
    if [ "$(uname -m)" = "x86_64" ]; then \
        pip install -r train/pytorch_requirements.txt --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install -r train/pytorch_requirements.txt; \
    fi && \
    pip install -r train/requirements.txt

# --- LibTorch Configuration ---
# Use the LibTorch included in the python package to ensure architecture compatibility (amd64/arm64)
# Ubuntu 24.04 uses Python 3.12
ENV Torch_DIR=/app/.venv/lib/python3.12/site-packages/torch/share/cmake/Torch
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/torch/lib:/app/.venv/lib/python3.12/site-packages/torch.libs:${LD_LIBRARY_PATH}"

# --- Project Build ---
# Copy C++ source files first to aggressively cache compilation
COPY CMakeLists.txt /app/
COPY *.cpp *.h /app/
COPY games /app/games

# Build the C++ project using limited cores to avoid Out-Of-Memory (OOM) kills on constrained VMs
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j1

# Copy the rest of the source code (including Python scripts)
COPY . /app

# Capture the exact Git revision into a file so it is permanently baked into the image
RUN git rev-parse HEAD > /app/git_version.txt 2>/dev/null || echo "N/A" > /app/git_version.txt

# --- Runtime Configuration ---
# Expose TensorBoard port
EXPOSE 6006

# Create a volume mount point for external data/models if needed (optional)
VOLUME ["/app/runs", "/app/models"]

# Ensure entrypoint is executable
RUN chmod +x /app/runpod_entrypoint.sh

# Default command: Generic interactive shell with venv activated
ENTRYPOINT ["/app/runpod_entrypoint.sh"]
