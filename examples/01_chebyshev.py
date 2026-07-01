#!/usr/bin/env python3
"""Chebyshev polynomial evaluation on GPU, validated against CPU Clenshaw.

Mirrors the HerMiniRocket Step() building blocks:
  - a degree-31 sign-like stage (tanh(8x), coefficients fitted on the fly),
  - the X4 cleaning polynomial (35x - 35x^3 + 21x^5 - 5x^7)/16 given directly
    as Chebyshev coefficients (with exact zeros at even indices),
  - both composed back-to-back.

Runs in seconds with ~1 GB of GPU memory (toy, NON-SECURE parameters).
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe

# X4 cleaning polynomial in Chebyshev basis (HerMiniRocket
# experiments/minimax_coefficients.py). Exact zeros at even indices.
X4_COEFFS = [0.0, 1.1962890625, 0.0, -0.2392578125,
             0.0, 0.0478515625, 0.0, -0.0048828125]


def clenshaw(coeffs, x):
    """CPU reference: Chebyshev series at x in [-1, 1]."""
    b1 = b2 = 0.0
    for c in reversed(coeffs[1:]):
        b1, b2 = 2.0 * x * b1 - b2 + c, b1
    return x * b1 - b2 + coeffs[0]


# ---------------------------------------------------------------- context
params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_NotSet)
params.SetRingDim(1 << 14)
params.SetMultiplicativeDepth(16)
params.SetScalingModSize(50)
params.SetFirstModSize(60)
params.SetNumLargeDigits(3)
params.SetBatchSize(1 << 13)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetDevices([0])

cc = fhe.GenCryptoContext(params)
for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
    cc.Enable(f)

keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.LoadContext(keys.publicKey)
print(f"Ring dimension: {cc.GetRingDimension()}")

# ---------------------------------------------------------------- input
n = 4096
xs = [-1.0 + 2.0 * i / (n - 1) for i in range(n)]
ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(xs))

# Degree-31 Chebyshev fit of tanh(8x) -- GetChebyshevCoefficients accepts any
# Python callable.
stage = fhe.GetChebyshevCoefficients(lambda v: math.tanh(8.0 * v), -1.0, 1.0, 31)

# ---------------------------------------------------------------- evaluate
t0 = time.perf_counter()
ct_s = cc.EvalChebyshevSeries(ct, stage, -1.0, 1.0)       # deg-31 stage
t1 = time.perf_counter()
ct_out = cc.EvalChebyshevSeries(ct_s, X4_COEFFS, -1.0, 1.0)  # X4 cleaning
cc.Synchronize()
t2 = time.perf_counter()

pt = cc.Decrypt(keys.secretKey, ct_out)
pt.SetLength(n)
he = pt.GetRealPackedValue()

# ---------------------------------------------------------------- check
ref = [clenshaw(X4_COEFFS, clenshaw(stage, x)) for x in xs]
max_err = max(abs(a - b) for a, b in zip(he, ref))

print(f"deg-31 stage: {1e3 * (t1 - t0):8.2f} ms")
print(f"X4 cleaning : {1e3 * (t2 - t1):8.2f} ms")
print(f"max |HE - CPU Clenshaw| = {max_err:.3e}")
assert max_err < 1e-6, "GPU Chebyshev evaluation diverged from CPU reference"
print("PASS")
