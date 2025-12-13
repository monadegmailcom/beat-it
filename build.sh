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

# LibTorch Path (adjust if needed)
LIBTORCH_PATH="/Users/monade/source/libtorch"

mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
