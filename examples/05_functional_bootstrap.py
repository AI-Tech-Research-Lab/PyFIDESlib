"""Refresh CKKS and evaluate real functions using Fourier-extension bootstrapping.

    python examples/05_functional_bootstrap.py          # GPU 0
    python examples/05_functional_bootstrap.py --cpu    # OpenFHE CPU backend

Small demonstration parameters (HEStd_NotSet): not suitable for real secrets.
For production, set HEStd_128_classic and let OpenFHE choose the ring dimension.
"""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # FE-FBT computes 2*Re(sum(c[j]*exp(pi*i*j*m))) for m in [-0.5, 0.5).
    # These exact series need no offline approximation generator.
    harmonics = [0, 0.5, 0, 0, 0, 0.125]  # cos(pi*m) + 0.25*cos(5*pi*m)
    sine = [0, -0.5j]                        # sin(pi*m)
    budget, slots = [3, 2], 8
    skd = fhe.SPARSE_ENCAPSULATED
    depth = fhe.CryptoContext.GetFEFBTDepth(budget, harmonics, skd, firstModSize=60) + 6
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_NotSet)
    params.SetRingDim(4096)
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)  # required: FirstModSize == ScalingModSize + 1
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetSecretKeyDist(skd)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(slots)
    params.SetCKKSDataType(fhe.REAL)
    params.SetDevices([] if args.cpu else [0])
    cc = fhe.GenCryptoContext(params)
    for feature in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
        cc.Enable(feature)
    cc.EvalFEFuncBootstrapSetup(levelBudget=budget, dim1=[0, 0], slots=slots)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    if not args.cpu:
        cc.LoadContext(keys.publicKey)

    x = [-0.45, -0.3, -0.15, 0.0, 0.1, 0.2, 0.35, 0.45]
    pt = cc.MakeCKKSPackedPlaintext(x, level=depth - (budget[1] + 1), slots=slots)
    ct = cc.Encrypt(keys.publicKey, pt)

    # One function, including the refresh.
    direct = cc.EvalFEFuncBootstrap(ciphertext=ct, coefficients=harmonics)
    # Several functions sharing one refresh: precompute for the longest series (degree >= 5).
    powers = cc.EvalFEFuncBootstrapPrecompute(ciphertext=ct, coefficients=harmonics)
    shared = cc.EvalFEFuncBootstrapWithPrecomp(powers=powers, coefficients=harmonics)
    sin_ct = cc.EvalFEFuncBootstrapWithPrecomp(powers=powers, coefficients=sine)
    expected_harmonics = [math.cos(math.pi * v) + 0.25 * math.cos(5 * math.pi * v) for v in x]
    for label, output, expected in (("harmonics (direct)", direct, expected_harmonics),
                                    ("harmonics (shared)", shared, expected_harmonics),
                                    ("sin(pi*m) (shared)", sin_ct, [math.sin(math.pi * v) for v in x])):
        result = cc.Decrypt(keys.secretKey, output)
        result.SetLength(len(x))
        values = list(result.GetRealPackedValue())
        error = max(abs(a - b) for a, b in zip(values, expected))
        print(f"{label}: {[round(v, 8) for v in values]} (max error {error:.2e})")


if __name__ == "__main__":
    main()
