#!/usr/bin/env bash
# Build fideslib_py against a chosen Python interpreter (default: the repo venv).
# Usage: [FIDESLIB_LOCAL=/path/to/fideslib/install] ./build.sh [/path/to/python]
#
# FIDESLIB_LOCAL must point at a FIDESlib install prefix (share/fideslib/cmake
# inside it). Default is the dev-machine build at ~/Fideslib/local; on a fresh
# machine build external/FIDESlib first (see external/FIDESlib/VENDORED.md).
set -e
cd "$(dirname "$0")"

PY="${1:-$(cd ../.. && pwd)/.venv/bin/python}"
[ -x "$PY" ] || PY="$HOME/HerMiniRocket/.venv/bin/python"
echo "Building against: $PY ($($PY --version))"

FIDESLIB_LOCAL="${FIDESLIB_LOCAL:-$HOME/Fideslib/local}"
if [ ! -d "$FIDESLIB_LOCAL/share/fideslib/cmake" ]; then
    echo "error: no FIDESlib install at $FIDESLIB_LOCAL" >&2
    echo "Build external/FIDESlib first (see external/FIDESlib/VENDORED.md)," >&2
    echo "then re-run with FIDESLIB_LOCAL=<install-prefix> $0" >&2
    exit 1
fi

cmake -B build -S . \
    -Dfideslib_DIR="$FIDESLIB_LOCAL/share/fideslib/cmake" \
    -DPython_EXECUTABLE="$PY"
cmake --build build -j"$(nproc)"

echo
echo "Done. Module at: $(ls fideslib_py/_core*.so)"
echo "Test with: $PY -c 'import sys; sys.path.insert(0, \"$PWD\"); import fideslib_py; print(\"ok\")'"
