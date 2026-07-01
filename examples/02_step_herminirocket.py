#!/usr/bin/env python3
"""Full HerMiniRocket Step() on GPU at production-scale parameters.

Lee alpha=8 composite sign [7,15,15] -> 2x X4 cleaning -> 0.5*x + 0.5,
at logN=17, depth 32, HEStd_128_classic (the real thing, secure parameters).

Needs ~2-4 GB of free GPU memory and takes ~1-2 min total
(dominated by CPU-side key generation at ring dimension 131072).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fideslib_py as fhe
from lee_alpha8_coeffs import LEE_ALPHA8_STAGES

X4_COEFFS = [0.0, 1.1962890625, 0.0, -0.2392578125,
             0.0, 0.0478515625, 0.0, -0.0048828125]
N_X4 = 2


def clenshaw(coeffs, x):
    b1 = b2 = 0.0
    for c in reversed(coeffs[1:]):
        b1, b2 = 2.0 * x * b1 - b2 + c, b1
    return x * b1 - b2 + coeffs[0]


def step_cpu(z):
    for s in LEE_ALPHA8_STAGES:
        z = clenshaw(s, z)
    for _ in range(N_X4):
        z = clenshaw(X4_COEFFS, z)
    return 0.5 * z + 0.5


# ---------------------------------------------------------------- context
params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_128_classic)
params.SetRingDim(1 << 17)
params.SetMultiplicativeDepth(32)
params.SetScalingModSize(50)
params.SetFirstModSize(60)
params.SetNumLargeDigits(4)
params.SetBatchSize(1 << 16)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetDevices([0])

cc = fhe.GenCryptoContext(params)
for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
    cc.Enable(f)
print(f"Ring dimension: {cc.GetRingDimension()}")

t0 = time.perf_counter()
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.LoadContext(keys.publicKey)
print(f"Keygen + GPU load: {time.perf_counter() - t0:.1f} s")

# ---------------------------------------------------------------- input
n = 1 << 16
zs = [-1.0 + 2.0 * i / (n - 1) for i in range(n)]
ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(zs))

# ---------------------------------------------------------------- Step()
t0 = time.perf_counter()
for stage in LEE_ALPHA8_STAGES:
    ct = cc.EvalChebyshevSeries(ct, stage, -1.0, 1.0)
for _ in range(N_X4):
    ct = cc.EvalChebyshevSeries(ct, X4_COEFFS, -1.0, 1.0)
ct = cc.EvalMult(ct, 0.5)
ct = cc.EvalAdd(ct, 0.5)
cc.Synchronize()
step_ms = 1e3 * (time.perf_counter() - t0)

pt = cc.Decrypt(keys.secretKey, ct)
pt.SetLength(n)
he = pt.GetRealPackedValue()

# ---------------------------------------------------------------- check
ref = [step_cpu(z) for z in zs]
max_err = max(abs(a - b) for a, b in zip(he, ref))
band = 2.0 ** -8
max_err_heaviside = max(
    abs(a - (1.0 if z > 0 else 0.0))
    for a, z in zip(he, zs) if abs(z) >= band
)

print(f"FULL STEP ({len(LEE_ALPHA8_STAGES)} stages + {N_X4}x X4 + affine): "
      f"{step_ms:.1f} ms for {n} lanes ({1e3 * step_ms / n:.2f} us/lane)")
print(f"max |HE - CPU composition|            = {max_err:.3e}")
print(f"max |HE - Heaviside| on 2^-8<=|z|<=1  = {max_err_heaviside:.3e}")
assert max_err < 1e-3
print("PASS")
