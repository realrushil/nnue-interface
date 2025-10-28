"""
Setup file for building Stockfish NNUE Python bindings
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import platform

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

# Detect if using MSVC
def is_msvc():
    """Check if we're using MSVC compiler"""
    # On Windows, check if MSVC is the compiler
    if platform.system() == 'Windows':
        # setuptools uses MSVC on Windows by default unless MinGW is configured
        # We can check by looking at the environment
        return os.environ.get('DISTUTILS_USE_SDK') != '1'
    return False

# Source files for the extension
sources = [
    'src/stockfish_nnue_bindings.cpp',
    'src/benchmark.cpp',
    'src/bitboard.cpp',
    'src/evaluate.cpp',
    'src/memory.cpp',
    'src/misc.cpp',
    'src/movegen.cpp',
    'src/movepick.cpp',
    'src/position.cpp',
    'src/search.cpp',
    'src/thread.cpp',
    'src/timeman.cpp',
    'src/tt.cpp',
    'src/uci.cpp',
    'src/ucioption.cpp',
    'src/tune.cpp',
    'src/syzygy/tbprobe.cpp',
    'src/nnue/nnue_accumulator.cpp',
    'src/nnue/nnue_misc.cpp',
    'src/nnue/features/half_ka_v2_hm.cpp',
    'src/nnue/network.cpp',
    'src/engine.cpp',
    'src/score.cpp',
]

# Compiler flags
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Windows':
    # Check if MSVC or MinGW/GCC
    if is_msvc():
        # MSVC flags (Visual Studio compiler)
        # Note: Don't use USE_PTHREADS on Windows, it uses native threading
        extra_compile_args = [
            '/std:c++17',
            '/O2',
            '/DNDEBUG',
            '/DIS_64BIT',
            '/DNNUE_EMBEDDING_OFF',  # Don't embed .nnue files
            '/DUSE_AVX2',
            '/arch:AVX2',
            '/DUSE_SSE41',
            '/DUSE_SSSE3',
            '/DUSE_SSE2',
            '/DUSE_POPCNT',
            '/EHsc',  # Enable C++ exceptions
        ]
        extra_link_args = []
    else:
        # GCC/MinGW flags for MSYS2 UCRT64
        extra_compile_args = [
            '-std=c++17',
            '-O3',
            '-DNDEBUG',
            '-DIS_64BIT',
            '-DNNUE_EMBEDDING_OFF',  # Don't embed .nnue files
            '-DUSE_PTHREADS',
            '-DUSE_AVX2',
            '-mavx2',
            '-mbmi',
            '-DUSE_SSE41',
            '-msse4.1',
            '-DUSE_SSSE3',
            '-mssse3',
            '-DUSE_SSE2',
            '-msse2',
            '-DUSE_POPCNT',
            '-msse3',
            '-mpopcnt',
            '-msse',
            '-m64',
            '-funroll-loops',
            '-Wall',
            '-Wcast-qual',
            '-fexceptions',
            '-pedantic',
            '-Wextra',
            '-Wshadow',
            '-Wmissing-declarations',
        ]
        extra_link_args = ['-lpthread']
elif platform.system() == 'Darwin':
    # macOS-specific flags
    # Detect if we're building for ARM64 (Apple Silicon)
    import subprocess
    try:
        # Check if we're targeting ARM64
        arch_output = subprocess.check_output(['uname', '-m'], text=True).strip()
        is_arm64 = 'arm64' in arch_output or 'aarch64' in arch_output
    except:
        is_arm64 = False
    
    # Start with common flags
    extra_compile_args = [
        '-std=c++17',
        '-O3',
        '-DNDEBUG',
        '-DIS_64BIT',
        '-DNNUE_EMBEDDING_OFF',  # Don't embed .nnue files
        '-DUSE_PTHREADS',
        '-funroll-loops',
        '-mmacosx-version-min=10.15',
    ]
    
    # Only add x86 intrinsics for x86_64
    if not is_arm64:
        extra_compile_args.extend([
            '-DUSE_AVX2',
            '-mavx2',
            '-mbmi',
            '-DUSE_SSE41',
            '-msse4.1',
            '-DUSE_SSSE3',
            '-mssse3',
            '-DUSE_SSE2',
            '-msse2',
            '-DUSE_POPCNT',
            '-msse3',
            '-mpopcnt',
            '-msse',
            '-m64',
        ])
    else:
        # ARM64 - disable x86 SIMD to avoid intrinsic header errors
        extra_compile_args.append('-DNO_PREFETCH')
    
    extra_link_args = ['-lpthread', '-mmacosx-version-min=10.15']
else:
    # Linux-specific flags
    extra_compile_args = [
        '-std=c++17',
        '-O3',
        '-DNDEBUG',
        '-DIS_64BIT',
        '-DNNUE_EMBEDDING_OFF',  # Don't embed .nnue files
        '-DUSE_PTHREADS',
        '-DUSE_AVX2',
        '-mavx2',
        '-mbmi',
        '-DUSE_SSE41',
        '-msse4.1',
        '-DUSE_SSSE3',
        '-mssse3',
        '-DUSE_SSE2',
        '-msse2',
        '-DUSE_POPCNT',
        '-msse3',
        '-mpopcnt',
        '-msse',
        '-m64',
        '-funroll-loops',
        '-Wall',
        '-Wcast-qual',
        '-fexceptions',
        '-pedantic',
        '-Wextra',
        '-Wshadow',
        '-Wmissing-declarations',
    ]
    extra_link_args = ['-lpthread']

ext_modules = [
    Extension(
        'stockfish_nnue',
        sources=sources,
        include_dirs=[
            get_pybind_include(),
            'src',
            'src/nnue',
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='nnue-interface',
    version='0.2.6',
    author='Rushil Saraf',
    description='Python bindings for Stockfish NNUE neural network',
    long_description='Extract NNUE activations and evaluations from Stockfish chess engine',
    packages=['nnue_interface'],
    package_dir={'nnue_interface': 'src'},
    ext_modules=ext_modules,
    ext_package='nnue_interface',
    install_requires=['pybind11>=2.6.0', 'numpy>=1.19.0'],
    setup_requires=['pybind11>=2.6.0'],
    python_requires='>=3.8',
    zip_safe=False,
)

