#!/usr/bin/env bash
# Build fideslib_py: compile the pinned FIDESlib submodule (external/FIDESlib) and the
# Python wrapper against it. No machine-specific FIDESlib path -- the FIDESlib version is
# whatever commit the external/FIDESlib submodule points at.
#
# Usage: [OPENFHE_LOCAL=/path/to/openfhe/install] ./build.sh [/path/to/python]
#
# Prerequisite (once): the patched OpenFHE 1.5.1 that FIDESlib depends on. Build it with
#   (cd external/FIDESlib/deps && ./build.sh "$PWD/openfhe-install")   # ~30 min
# after which it is picked up automatically (default OPENFHE_LOCAL below), or point
# OPENFHE_LOCAL at an existing patched-OpenFHE-1.5.1 install prefix.
set -e
cd "$(dirname "$0")"
ROOT="$PWD"

PY="${1:-python3}"
echo "Building against: $PY ($("$PY" --version))"

# 1. Pinned FIDESlib source (exact commit recorded by the submodule). Non-recursive: the
# OpenFHE source sub-submodule is only needed to *build* OpenFHE (step 2's one-time
# bootstrap fetches it itself), not to link against a prebuilt OpenFHE install.
git submodule update --init external/FIDESlib

# 2. Patched OpenFHE 1.5.1 (FIDESlib dependency; build once -- see header).
OPENFHE_LOCAL="${OPENFHE_LOCAL:-$ROOT/external/FIDESlib/deps/openfhe-install}"
if [ ! -d "$OPENFHE_LOCAL/lib/OpenFHE" ]; then
    echo "error: patched OpenFHE 1.5.1 not found at $OPENFHE_LOCAL" >&2
    echo "Build it once:  (cd external/FIDESlib/deps && ./build.sh \"$OPENFHE_LOCAL\")" >&2
    echo "or re-run with  OPENFHE_LOCAL=<openfhe-1.5.1-install-prefix> $0" >&2
    exit 1
fi

# 3. Build + install the pinned FIDESlib (library only) to a repo-relative prefix.
FIDESLIB_SRC="$ROOT/external/FIDESlib"
FIDESLIB_PREFIX="$FIDESLIB_SRC/install"
cmake -S "$FIDESLIB_SRC" -B "$FIDESLIB_SRC/build" \
    -DFIDESLIB_INSTALL_PREFIX="$FIDESLIB_PREFIX" \
    -DOPENFHE_INSTALL_PREFIX="$OPENFHE_LOCAL" \
    -DFIDESLIB_COMPILE_TESTS=OFF \
    -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$FIDESLIB_SRC/build" -j"$(nproc)"
cmake --install "$FIDESLIB_SRC/build"

# 4. Build the Python wrapper against the freshly-built FIDESlib.
cmake -B build -S . \
    -Dfideslib_DIR="$FIDESLIB_PREFIX/share/fideslib/cmake" \
    -DPython_EXECUTABLE="$PY"
cmake --build build -j"$(nproc)"

echo
echo "Done. Module at: $(ls fideslib_py/_core*.so)"
echo "Test with: $PY -c 'import sys; sys.path.insert(0, \"$PWD\"); import fideslib_py; print(\"ok\")'"
