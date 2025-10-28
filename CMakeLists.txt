cmake_minimum_required(VERSION 3.15)
project(nnue_interface CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add Stockfish source files
set(STOCKFISH_SOURCES
    src/benchmark.cpp
    src/bitboard.cpp
    src/evaluate.cpp
    src/memory.cpp
    src/misc.cpp
    src/movegen.cpp
    src/movepick.cpp
    src/position.cpp
    src/search.cpp
    src/thread.cpp
    src/timeman.cpp
    src/tt.cpp
    src/uci.cpp
    src/ucioption.cpp
    src/tune.cpp
    src/engine.cpp
    src/score.cpp
    src/syzygy/tbprobe.cpp
    src/nnue/nnue_accumulator.cpp
    src/nnue/nnue_misc.cpp
    src/nnue/features/half_ka_v2_hm.cpp
    src/nnue/network.cpp
)

# Create the Python extension module
pybind11_add_module(nnue_interface
    src/stockfish_nnue_bindings.cpp
    ${STOCKFISH_SOURCES}
)

# Add include directories
target_include_directories(nnue_interface PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nnue
)

# Compiler flags for optimization
if(MSVC)
    # Windows (MSVC)
    target_compile_options(nnue_interface PRIVATE
        /O2 /DNDEBUG /DIS_64BIT
        /DUSE_AVX2 /arch:AVX2
        /DUSE_SSE41 /DUSE_SSSE3 /DUSE_SSE2 /DUSE_POPCNT
    )
else()
    # Linux, macOS, MinGW, etc.
    target_compile_options(nnue_interface PRIVATE
        -O3 -DNDEBUG -DIS_64BIT
        -DUSE_AVX2 -mavx2 -mbmi
        -DUSE_SSE41 -msse4.1
        -DUSE_SSSE3 -mssse3
        -DUSE_SSE2 -msse2
        -DUSE_POPCNT -mpopcnt
        -msse -m64
        -funroll-loops
        -Wall -Wextra -Wshadow
        -fexceptions
    )
    
    # Add pthread for multithreading
    if(UNIX AND NOT APPLE)
        target_link_libraries(nnue_interface PRIVATE pthread)
    endif()
endif()

# Set optimization level
if(NOT MSVC)
    target_compile_options(nnue_interface PRIVATE -march=native)
endif()
