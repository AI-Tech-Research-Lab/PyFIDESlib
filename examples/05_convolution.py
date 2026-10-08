"""Three equivalent GPU implementations of a cyclic [0.25, 0.5, 0.25] convolution.

    python examples/05_convolution.py

Uses one GPU by default; respects CUDA_VISIBLE_DEVICES. All rotations are cyclic,
so a non-cyclic convolution needs additional padding or plaintext masks.
"""

import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe

slots = 16
params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_NotSet)
params.SetRingDim(1 << 14)
params.SetMultiplicativeDepth(4)
params.SetScalingModSize(50)
params.SetFirstModSize(60)
params.SetNumLargeDigits(3)
params.SetBatchSize(slots)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetDevices([0])
cc = fhe.GenCryptoContext(params)
for feature in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE):
    cc.Enable(feature)

g_step, b_step, indexes = 2, 2, [-2, -1]
keys = cc.KeyGen()
rotation_keys = fhe.convolution_rotation_indices(g_step, b_step, indexes)
cc.EvalRotateKeyGen(keys.secretKey, rotation_keys)
cc.LoadContext(keys.publicKey)

x = [(i - 8) / 10 for i in range(slots)]
ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
expected = [0.25 * x[(i - 1) % slots] + 0.5 * x[i] + 0.25 * x[(i + 1) % slots]
            for i in range(slots)]

# Shared hoisting for the three rotations, including an independent copy for index 0.
rotations = cc.EvalFastRotation(ct, [-1, 0, 1])
weights = [0.25, 0.5, 0.25]
scalar_result = ct.Clone()
scalar_result.evalLinearWSumMutable(len(rotations), rotations, weights)
cc.RescaleInPlace(scalar_result)

# Plaintext vectors can carry different weights/masks in every slot.
pts = [cc.MakeCKKSPackedPlaintext([w] * slots, level=ct.GetLevel()) for w in weights]
plaintext_result = ct.Clone()
plaintext_result.dotProductPt(rotations, pts)
cc.RescaleInPlace(plaintext_result)

# Packing: R_2(0.5*R_-2(ct) + 0.25*R_-1(ct))
#        + R_1(0.25*R_-2(ct) + 0*R_-1(ct)).
transform_pts = [cc.MakeCKKSPackedPlaintext([w] * slots, level=ct.GetLevel())
                 for w in [0.5, 0.25, 0.25, 0.0]]
transform_result = ct.Clone()
cc.ConvolutionTransformInPlace(transform_result, g_step, b_step, transform_pts, indexes)
cc.RescaleInPlace(transform_result)

for name, result in [("scalar weighted sum", scalar_result),
                     ("plaintext dot product", plaintext_result),
                     ("convolution transform", transform_result)]:
    pt = cc.Decrypt(keys.secretKey, result)
    pt.SetLength(slots)
    actual = list(pt.GetRealPackedValue())
    error = max(abs(a - b) for a, b in zip(actual, expected))
    assert error < 1e-5, (name, error)
    print(f"{name}: max error {error:.3g}; first 4 slots {actual[:4]}")
