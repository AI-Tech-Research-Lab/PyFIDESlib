#!/usr/bin/env python3
"""Bootstrapping: refresh a ciphertext that has run out of levels.

Every multiplication consumes a level; when they are gone the ciphertext is at the
bottom of the modulus chain and can only be decrypted. Bootstrapping evaluates the
decryption circuit homomorphically to hand the levels back, so the computation can
continue on the same encrypted value.

Runs in seconds with ~1 GB of GPU memory (toy, NON-SECURE parameters).

Choose --stc-first for SlotsToCoeffs-first bootstrap, --iterations 2 for error
correction, and --precision for the estimated single-pass precision in bits.

The context depth must cover modular reduction, the two transform budgets and the
levels needed by the application. UNIFORM_TERNARY uses 14 levels for modular reduction
in this recipe. FirstModSize=60 / ScalingModSize=59 follows the OpenFHE bootstrap
recipe. Transform budgets trade depth against rotation work; measure on your workload.
StC-first requires levels for SlotsToCoeffs before the modulus raise, so it starts earlier
than standard bootstrap. These toy parameters are for real values.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find fideslib_py
import fideslib_py as fhe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stc-first", action="store_true")
parser.add_argument("--iterations", type=int, choices=(1, 2), default=1)
parser.add_argument("--precision", type=int, default=12)
args = parser.parse_args()

MOD_DEPTH = 14           # UNIFORM_TERNARY modular reduction
LEVEL_BUDGET = [3, 3]    # levels spent on the two linear transforms
SPARE = 4                # levels left usable after a bootstrap
DEPTH = MOD_DEPTH + sum(LEVEL_BUDGET) + SPARE

RING = 1 << 13
SLOTS = RING // 4        # sparse: half of the full RING/2 slots

# ---------------------------------------------------------------- parameters
params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_NotSet)
params.SetSecretKeyDist(fhe.UNIFORM_TERNARY)  # what MOD_DEPTH = 14 assumes
params.SetRingDim(RING)
params.SetMultiplicativeDepth(DEPTH)
params.SetScalingModSize(59)                  # 60 - 59 = 1
params.SetFirstModSize(60)
params.SetNumLargeDigits(3)
params.SetBatchSize(RING // 2)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)      # bootstrapping requires HYBRID
params.SetDevices([0])

cc = fhe.GenCryptoContext(params)
for feature in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
    cc.Enable(feature)  # ADVANCEDSHE and FHE are both needed to bootstrap

print(f"ring 2^13, depth {DEPTH} = {MOD_DEPTH} (mod reduction) + "
      f"{sum(LEVEL_BUDGET)} (level budget) + {SPARE} (usable), {SLOTS} slots")

# ---------------------------------------------------------------- keys
# Order matters: everything that creates a key has to happen before LoadContext(),
# which is what pushes them to the GPU. All of these throw once the context is loaded.
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalBootstrapSetup(LEVEL_BUDGET, [0, 0], SLOTS, btsfirstboot=args.stc_first)
cc.EvalBootstrapKeyGen(keys.secretKey, SLOTS)        # the rotation keys it needs

t0 = time.time()
cc.LoadContext(keys.publicKey)
print(f"LoadContext (bootstrapping keys included): {time.time() - t0:.1f}s")

# ---------------------------------------------------------------- spend every level
x = [0.25 + 1e-4 * (i % 100) for i in range(SLOTS)]


def dec(c, n=4):
    p = cc.Decrypt(keys.secretKey, c)
    p.SetLength(n)
    return p.GetRealPackedValue()


# Encrypt at the bootstrap input level instead of evaluating a long multiplication chain.
# StC-first needs LEVEL_BUDGET[1] levels for its initial SlotsToCoeffs, plus overhead.
input_level = DEPTH + 1 - (LEVEL_BUDGET[1] + 2) if args.stc_first else DEPTH - 1
pt = cc.MakeCKKSPackedPlaintext(x, 1, input_level, SLOTS)
ct = cc.Encrypt(keys.publicKey, pt)
print(f"\nciphertext is at level {ct.GetLevel()} of {DEPTH}: "
      f"{DEPTH - ct.GetLevel()} level(s) left before refresh")

# ---------------------------------------------------------------- bootstrap
t0 = time.time()
bootstrap = cc.EvalBootstrapStCFirst if args.stc_first else cc.EvalBootstrap
fresh = bootstrap(ct, args.iterations, args.precision)
got = dec(fresh)  # every Eval* only enqueues: the decrypt is what waits for the GPU
print(f"EvalBootstrap (timed with the decrypt that waits on it): {time.time() - t0:.1f}s")
print(f"back at level {fresh.GetLevel()}: {DEPTH - fresh.GetLevel()} level(s) usable again")
assert fresh.GetLevel() < ct.GetLevel(), "no level was restored"

err = max(abs(a - b) for a, b in zip(got, x))
print(f"value survived: {[round(v, 6) for v in got]} vs {x[:4]}  (max error {err:.1e})")
# Bootstrapping is approximate -- it trades precision for levels. ~1e-5 at this ring
# size is normal and has nothing to do with the offload/reload of the keys; do not
# expect the 1e-10 of an ordinary rotation.
assert err < 1e-3, f"bootstrap error {err:.1e} is too large -- check the parameters"

# ---------------------------------------------------------------- use the levels
# The point of the exercise: the refreshed ciphertext can be multiplied again.
squared = cc.EvalMult(fresh, fresh)
fourth = cc.EvalMult(squared, squared)
got = dec(fourth)
expected = [v**4 for v in x[:4]]
print(f"\nx^4 on the refreshed ciphertext: {[round(v, 6) for v in got]} "
      f"vs {[round(v, 6) for v in expected]}")
assert max(abs(a - b) for a, b in zip(got, expected)) < 1e-3, "x^4 after bootstrap is wrong"

# SetRotationKeyCache() bounds rotation-key VRAM and SetBootstrapCache() bounds the
# CoeffsToSlots/SlotsToCoeffs matrices. Set both before LoadContext(); see the README.
print("\nAll checks passed.")
