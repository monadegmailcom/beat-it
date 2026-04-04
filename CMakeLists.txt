cmake_minimum_required(VERSION 3.16)

# Suppress warning about FindBoost module removal in newer CMake versions
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# --- Standard Configuration ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Build Type Configuration ---
# Add an option to switch between CUDA and ROCm (HIP)
option(USE_ROCM "Build for AMD GPUs using ROCm/HIP" OFF)

if(USE_ROCM)
    message(STATUS "ROCm build enabled")
    project(beat-it LANGUAGES CXX HIP)
    set(CMAKE_HIP_STANDARD 17)
elseif(APPLE)
    message(STATUS "Apple/MPS build enabled (no CUDA)")
    project(beat-it LANGUAGES CXX)
else()
    message(STATUS "CUDA build enabled")
    project(beat-it LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# Dependencies
find_package(Torch REQUIRED)
find_package(Boost REQUIRED)

# Helper macro to strip standard flags from targets
macro(strip_torch_cxx_flags target_name)
    if(TARGET ${target_name})
        get_target_property(target_options ${target_name} INTERFACE_COMPILE_OPTIONS)
        if(target_options AND NOT "${target_options}" STREQUAL "target_options-NOTFOUND")
            string(REPLACE "-std=c++17" "" target_options "${target_options}")
            string(REPLACE "-std=c++14" "" target_options "${target_options}")
            string(REPLACE "-std=gnu++17" "" target_options "${target_options}")
            string(REPLACE "-std=gnu++14" "" target_options "${target_options}")
            set_target_properties(${target_name} PROPERTIES INTERFACE_COMPILE_OPTIONS "${target_options}")
        endif()
    endif()
endmacro()

# Apply the stripping to known Torch targets
strip_torch_cxx_flags(torch)
strip_torch_cxx_flags(torch_cpu)
strip_torch_cxx_flags(torch_hip)
strip_torch_cxx_flags(torch_cuda)

# Forcefully remove C++17 or C++14 flags that PyTorch might have added
string(REPLACE "-std=c++17" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-std=c++14" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-std=gnu++17" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-std=gnu++14" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Force C++23 globally
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")

# Collect source files
file(GLOB_RECURSE GAME_SOURCES "games/*.cpp")
file(GLOB CORE_SOURCES "*.cpp")

# Filter out main and test files for the library part
list(REMOVE_ITEM CORE_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp"
)

# Common sources for all targets
set(ALL_SOURCES ${CORE_SOURCES} ${GAME_SOURCES})

# --- Shared Library Target ---
add_library(alphazero SHARED ${ALL_SOURCES})
target_link_libraries(alphazero PRIVATE ${TORCH_LIBRARIES} PUBLIC Boost::boost)
target_include_directories(alphazero PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- Main Executable Target ---
add_executable(beat-it main.cpp)
target_link_libraries(beat-it PRIVATE alphazero ${TORCH_LIBRARIES})

# --- Test Executable Target ---
add_executable(beat-it-test test.cpp)
target_link_libraries(beat-it-test PRIVATE alphazero ${TORCH_LIBRARIES})

# --- Compiler Flags ---
if(MSVC)
    target_compile_options(alphazero PRIVATE /W4)
else()
    # Force C++23 standard to override any PyTorch defaults
    target_compile_options(alphazero PRIVATE -Wall -Wextra -pedantic -std=c++23)
    target_compile_options(beat-it PRIVATE -std=c++23)
    target_compile_options(beat-it-test PRIVATE -std=c++23)
endif()

# Ensure standard is respected
target_compile_features(alphazero PUBLIC cxx_std_23)
target_compile_features(beat-it PUBLIC cxx_std_23)
target_compile_features(beat-it-test PUBLIC cxx_std_23)

# Optimized build setup similar to Makefile
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
