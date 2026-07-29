#!/usr/bin/env bash
# Build fideslib_py: CMake clones and compiles its own pinned copy of FIDESlib (and FIDESlib's
# own vendored OpenFHE) entirely inside build/, then builds the Python wrapper against it. No
# submodules, no external install prefixes, nothing outside this repo -- see CMakeLists.txt's
# FIDESLIB_REPOSITORY/FIDESLIB_GIT_TAG cache variables to point at a fork or a different commit.
#
# Usage: ./build.sh [/path/to/python]
#
# First run compiles FIDESlib's vendored OpenFHE from scratch (~10-30 min); later runs are fast
# since it's only rebuilt when missing.
set -e
cd "$(dirname "$0")"

PY="${1:-python3}"
echo "Building against: $PY ($("$PY" --version))"

cmake -B build -S . -DPython_EXECUTABLE="$PY" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

# Regenerate type stubs from the freshly built module so editors (Pylance/mypy) stay in sync
# with the bindings. Requires `uv`; skipped with a warning if absent -- the committed
# fideslib_py/_core.pyi then just stays as-is.
if command -v uv >/dev/null 2>&1; then
  echo "Regenerating stubs (fideslib_py/_core.pyi)..."
  PYTHONPATH="$PWD" uv tool run --python 3.12 --with numpy --from pybind11-stubgen \
    pybind11-stubgen fideslib_py._core --numpy-array-use-type-var -o .
else
  echo "WARNING: uv not found -- skipping stub regeneration; fideslib_py/_core.pyi may be stale."
fi

echo
echo "Done. Module at: $(ls fideslib_py/_core*.so)"
echo "Test with: $PY -c 'import sys; sys.path.insert(0, \"$PWD\"); import fideslib_py; print(\"ok\")'"
