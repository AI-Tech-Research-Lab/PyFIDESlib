#!/usr/bin/env python3
"""Bootstrapping: refresh a ciphertext that has run out of levels.

Every multiplication consumes a level; when they are gone the ciphertext is at the
bottom of the modulus chain and can only be decrypted. Bootstrapping evaluates the
decryption circuit homomorphically to hand the levels back, so the computation can
continue on the same encrypted value.

Runs in seconds with ~1 GB of GPU memory (toy, NON-SECURE parameters).

  READ THIS BEFORE CHANGING THE PARAMETERS. Bootstrapping is the one part of this API
  that does not tell you when it has been set up wrong -- it segfaults, or it decrypts
  to noise, instead of raising:

  1. The multiplicative depth must cover the bootstrap itself. For a UNIFORM_TERNARY
     secret that is MOD_DEPTH = 14 levels (OpenFHE's degree-44 Chebyshev approximation
     of the modular reduction, plus its 6 double-angle iterations) plus both halves of
     the level budget -- before the levels you actually want to use are counted. Ask for
     less and EvalBootstrapSetup() SEGFAULTS: the leftover level count goes negative in
     unsigned arithmetic.
  2. FirstModSize - ScalingModSize must not exceed the correction factor OpenFHE derives
     from the ring and slot counts (it clamps it to 7..14). The 60/50 pair used by the
     other examples gives 10, which is one too many: the CPU path at least throws
     "Degree [10] must be less than or equal to the correction factor [9]", the GPU path
     just returns noise. 60/59 -- the OpenFHE bootstrapping recipe -- gives 1.
  3. The level budget trades levels against rotation keys and time: [3, 3] here needs
     54 keys, [1, 1] needs 130 and is ~2x slower, [5, 5] needs 28 but costs 4 more
     levels. Note that [2, 2] and [5, 5] decrypt to noise on the GPU at 2048 slots (but
     not at 1024 or 4096, and not on the CPU) -- so re-check the result after changing
     it, and see tests/diag_bootstrap.py.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find fideslib_py
import fideslib_py as fhe

MOD_DEPTH = 14           # see rule 1 above -- UNIFORM_TERNARY
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
params.SetScalingModSize(59)                  # 60 - 59 = 1, see rule 2 above
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
cc.EvalBootstrapSetup(LEVEL_BUDGET, [0, 0], SLOTS)   # precomputation, no keys yet
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


# Encrypting straight at the bottom level is the shortcut for "a long computation just
# finished": DEPTH - 1 multiplications would have got here the slow way.
pt = cc.MakeCKKSPackedPlaintext(x, 1, DEPTH - 1, SLOTS)
ct = cc.Encrypt(keys.publicKey, pt)
print(f"\nciphertext is at level {ct.GetLevel()} of {DEPTH}: "
      f"{DEPTH - ct.GetLevel()} level(s) left, a multiplication chain ends here")

# ---------------------------------------------------------------- bootstrap
t0 = time.time()
fresh = cc.EvalBootstrap(ct)
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

# Bootstrapping keys are ordinary rotation keys and by far the largest VRAM tenant here
# (54 of them, ~688 MB at these parameters). SetRotationKeyCache() bounds them -- see the
# README section on rotation-key VRAM; it must be called before LoadContext().
print("\nAll checks passed.")
