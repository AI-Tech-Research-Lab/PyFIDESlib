cmake_minimum_required(VERSION 3.25.2)
project(fideslib_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FIDESlib (local install) -- pass -Dfideslib_DIR=$HOME/Fideslib/local/share/fideslib/cmake
find_package(fideslib REQUIRED CONFIG)
find_package(OpenMP REQUIRED)

# pybind11 (self-contained fetch, no system install needed)
set(PYBIND11_FINDPYTHON ON)  # use modern FindPython, honors -DPython_EXECUTABLE
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_core src/bindings.cpp)

target_link_libraries(_core PRIVATE fideslib::fideslib OpenMP::OpenMP_CXX)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/fideslib_py"
    BUILD_RPATH "/usr/local/cuda/lib64"
    INSTALL_RPATH "/usr/local/cuda/lib64"
)
