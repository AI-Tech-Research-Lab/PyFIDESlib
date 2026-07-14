cmake_minimum_required(VERSION 3.25.2)
project(fideslib_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# FIDESlib (a.k.a. OurFIDESlib): cloned and built entirely inside this build tree, pinned to an
# exact commit. No system install, no external fideslib_DIR, no path outside this repo -- override
# FIDESLIB_REPOSITORY/FIDESLIB_GIT_TAG to point at a fork or a different pinned commit.
set(FIDESLIB_REPOSITORY "git@github.com:AI-Tech-Research-Lab/FIDESlib.git" CACHE STRING "Git remote for the FIDESlib core library.")
set(FIDESLIB_GIT_TAG "acc16a4f2ec984e040d5d7dee2b27a9023ed3c35" CACHE STRING "Pinned FIDESlib commit (rebased on upstream 2.1.3: their Issue-#27 key-switch fix, our NaN workarounds dropped).")

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

# pybind11 (self-contained fetch, no system install needed)
set(PYBIND11_FINDPYTHON ON)  # use modern FindPython, honors -DPython_EXECUTABLE
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_core src/bindings.cpp)

target_link_libraries(_core PRIVATE fideslib OpenMP::OpenMP_CXX)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/fideslib_py"
    BUILD_RPATH "/usr/local/cuda/lib64"
    INSTALL_RPATH "/usr/local/cuda/lib64"
)
