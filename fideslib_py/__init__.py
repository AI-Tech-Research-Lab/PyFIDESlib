"""Python bindings for FIDESlib — CKKS homomorphic encryption on GPU.

FIDESlib (https://github.com/CAPS-UMU/FIDESlib) executes CKKS homomorphic
operations on NVIDIA GPUs while delegating context/key generation, encoding
and encrypt/decrypt to an embedded (patched) OpenFHE 1.5.1.

IMPORTANT: do not import this module together with `openfhe` (openfhe-python)
in the same process — both embed different OpenFHE versions.

See examples/ for usage.
"""

from ._core import *  # noqa: F401,F403
from ._core import __doc__ as _core_doc  # noqa: F401
