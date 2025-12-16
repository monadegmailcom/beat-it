#!/bin/bash
set -e

# Configuration for Homebrew LLVM (required for C++20/23 features like jthread on macOS)
if [[ "$(uname)" == "Darwin" ]]; then
    export LLVM_PATH="/opt/homebrew/opt/llvm"
    if [ -d "$LLVM_PATH" ]; then
        echo "Using Homebrew LLVM at $LLVM_PATH"
        export CC="$LLVM_PATH/bin/clang"
        export CXX="$LLVM_PATH/bin/clang++"
        export LDFLAGS="-L$LLVM_PATH/lib/c++ -L$LLVM_PATH/lib/unwind -lunwind -Wl,-rpath,$LLVM_PATH/lib/c++"
        export CXXFLAGS="-isystem$LLVM_PATH/include"
    else
        echo "Warning: Homebrew LLVM not found at $LLVM_PATH. Build might fail if system compiler doesn't support C++20/23."
    fi
fi

# Determine the LibTorch path from the current Python environment (venv)
# This ensures binary compatibility between the C++ extension and the Python runtime.
if [ -d ".venv" ]; then
    echo "Using LibTorch from .venv..."
    LIBTORCH_PATH=$(source .venv/bin/activate && python -c "import torch; import os; print(os.path.dirname(torch.__file__))")
else
    # Fallback or error if venv not found. For this user, we expect venv.
    echo "Error: .venv not found! Please compile from within the project root with the venv present."
    exit 1
fi
export CMAKE_PREFIX_PATH="$LIBTORCH_PATH"
echo "CMAKE_PREFIX_PATH set to: $CMAKE_PREFIX_PATH"


# Set build type (default to Release if not provided)
BUILD_TYPE=${1:-Release}
echo "Build Type: $BUILD_TYPE"

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
