cmake_minimum_required(VERSION 3.16)

# Suppress warning about FindBoost module removal in newer CMake versions
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(beat-it LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies
find_package(Torch REQUIRED)
find_package(Boost REQUIRED)

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
    target_compile_options(alphazero PRIVATE -Wall -Wextra -pedantic)
endif()

# Optimized build setup similar to Makefile
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
