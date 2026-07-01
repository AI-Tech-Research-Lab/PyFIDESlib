#!/usr/bin/env python3
"""Onboarding: the minimal FIDESlib-from-Python workflow.

Covers: context setup -> key generation -> GPU load -> encrypt ->
homomorphic ops (add, plaintext mult, rotation, slot accumulation) -> decrypt.

Runs in seconds with <1 GB of GPU memory (toy, NON-SECURE parameters).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find fideslib_py
import fideslib_py as fhe

# ---------------------------------------------------------------- parameters
# Toy parameters: ring 2^13, depth 4. HEStd_NotSet disables the security
# check -- fine for functional tests, never for real data.
params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_NotSet)
params.SetRingDim(1 << 13)
params.SetMultiplicativeDepth(4)
params.SetScalingModSize(50)
params.SetFirstModSize(60)
params.SetNumLargeDigits(3)
params.SetBatchSize(1 << 12)  # number of usable slots (<= ringDim / 2)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetDevices([0])  # GPU id(s); [] would run everything on CPU OpenFHE

cc = fhe.GenCryptoContext(params)
cc.Enable(fhe.PKE)
cc.Enable(fhe.KEYSWITCH)
cc.Enable(fhe.LEVELEDSHE)
cc.Enable(fhe.ADVANCEDSHE)
cc.Enable(fhe.FHE)
print(f"Ring dimension: {cc.GetRingDimension()}")

# ---------------------------------------------------------------- keys
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)

# Rotation keys: one key per rotation index you plan to use.
# AccumulateSum(ct, slots, stride) needs the indices below.
N_SUM = 8
rot_indices = [1] + fhe.accumulate_rotation_indices(N_SUM, stride=1)
cc.EvalRotateKeyGen(keys.secretKey, sorted(set(rot_indices)))

# Push context + keys to the GPU. Must happen AFTER all key generation.
cc.LoadContext(keys.publicKey)
print("Context loaded on GPU.")

# ---------------------------------------------------------------- encrypt
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
pt_x = cc.MakeCKKSPackedPlaintext(x)
ct = cc.Encrypt(keys.publicKey, pt_x)

# ---------------------------------------------------------------- operate
ct_twice = cc.EvalAdd(ct, ct)            # ct + ct           -> 2x
ct_plus1 = cc.EvalAdd(ct, 1.0)           # ct + scalar       -> x + 1
pt_w = cc.MakeCKKSPackedPlaintext([0.5] * len(x))
ct_half = cc.EvalMult(ct, pt_w)          # ct * plaintext    -> x / 2
ct_rot = cc.EvalRotate(ct, 1)            # slots shift left  -> x[i+1]
ct_sum = cc.AccumulateSum(ct, N_SUM)     # sum of 8 slots in slot 0 (~EvalSum)

# ---------------------------------------------------------------- decrypt
def dec(c, n=len(x)):
    p = cc.Decrypt(keys.secretKey, c)
    p.SetLength(n)
    return p.GetRealPackedValue()

print(f"x          = {[round(v, 6) for v in dec(ct)]}")
print(f"x + x      = {[round(v, 6) for v in dec(ct_twice)]}")
print(f"x + 1      = {[round(v, 6) for v in dec(ct_plus1)]}")
print(f"x * 0.5    = {[round(v, 6) for v in dec(ct_half)]}")
print(f"rot(x, 1)  = {[round(v, 6) for v in dec(ct_rot)]}")
print(f"sum slot 0 = {dec(ct_sum)[0]:.6f}   (expected {sum(x):.1f})")

assert abs(dec(ct_sum)[0] - sum(x)) < 1e-6
assert all(abs(a - 2 * b) < 1e-6 for a, b in zip(dec(ct_twice), x))
print("\nAll checks passed.")
