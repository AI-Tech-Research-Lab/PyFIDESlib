cmake_minimum_required(VERSION 3.25.2)
project(fideslib_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# FIDESlib (a.k.a. OurFIDESlib): cloned and built entirely inside this build tree, pinned to an
# exact commit. No system install, no external fideslib_DIR, no path outside this repo -- override
# FIDESLIB_REPOSITORY/FIDESLIB_GIT_TAG to point at a fork or a different pinned commit.
#
# These defaults are deliberately NOT CACHE variables: a cached entry takes precedence over the
# set() below, so bumping the pin here would silently do nothing on an existing build tree (the
# fetched copy stays on the old commit and the wrapper then fails to compile against it). Passing
# -DFIDESLIB_GIT_TAG=... still works and does create a cache entry -- which then shadows this line
# until you clear it with `cmake -B build -U FIDESLIB_GIT_TAG`.
if (NOT DEFINED FIDESLIB_REPOSITORY)
    set(FIDESLIB_REPOSITORY "git@github.com:AI-Tech-Research-Lab/FIDESlib.git")
endif ()
if (NOT DEFINED FIDESLIB_GIT_TAG)
    # multi-GPU fixes: the two NaN fixes lost in the branch resync, the wait direction when a
    # limb buffer returns to the GPU pool, support for devices that own no special primes
    # (#GPUs > K used to abort mid-computation), the memory pool indexed by device rather than
    # by partition index (a device list that was not 0,1,...,n-1 could be served another
    # device's memory, and crashed at teardown), --devices for the gtest suite, and NCCL
    # instead of cudaMemcpyPeerAsync as the default multi-GPU transport, since the peer-copy
    # special-limb exchange races under contention -- which reaches sparse bootstrap, through the
    # Accumulate step, as well as AccumulateSum itself. See docs/multigpu-plan.md.
    set(FIDESLIB_GIT_TAG "0342b2876f5763425ceae2c2ed0781731a25643d")
endif ()

# Library only: skip FIDESlib's own test/benchmark suites (and their gtest/benchmark fetches).
set(FIDESLIB_COMPILE_TESTS OFF CACHE BOOL "" FORCE)
set(FIDESLIB_COMPILE_BENCHMARKS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    fideslib
    GIT_REPOSITORY ${FIDESLIB_REPOSITORY}
    GIT_TAG        ${FIDESLIB_GIT_TAG}
    # No submodule init: upstream history pins an unfetchable openfhe commit (31c9304) that makes
    # recursive fetches fail; FIDESlib's own deps/build.sh inits the submodule itself (--remote).
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(fideslib)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# pybind11 (self-contained fetch, no system install needed)
set(PYBIND11_FINDPYTHON ON)  # use modern FindPython, honors -DPython_EXECUTABLE
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_core src/bindings.cpp)

target_include_directories(_core PRIVATE
    "${fideslib_SOURCE_DIR}/src"
    "${fideslib_SOURCE_DIR}/deps/openfhe-install/include/openfhe"
    "${fideslib_SOURCE_DIR}/deps/openfhe-install/include/openfhe/core"
    "${fideslib_SOURCE_DIR}/deps/openfhe-install/include/openfhe/pke"
    "${fideslib_SOURCE_DIR}/deps/openfhe-install/include/openfhe/binfhe"
    "${fideslib_SOURCE_DIR}/deps/openfhe-install/include/openfhe/third-party/include"
)
target_compile_definitions(_core PRIVATE MULTI_GPU NCCL RCCL)

target_link_libraries(_core PRIVATE fideslib OpenMP::OpenMP_CXX CUDA::cudart)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/fideslib_py"
    BUILD_RPATH "/usr/local/cuda/lib64"
    INSTALL_RPATH "/usr/local/cuda/lib64"
)
