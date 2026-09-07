#!/usr/bin/env python3
"""Diagnostic: an end-to-end bootstrap, one stage at a time, with the parameter rules
that make one work spelled out.

This script used to report `EvalBootstrapSetup()` as a wrapper bug that segfaulted for
"every parameter combination tried". It is not a wrapper bug: every combination tried
happened to break the same rule, a multiplicative depth (11) far below what a bootstrap
costs. Two rules, and breaking either one is silent or fatal rather than an exception:

  depth >= MOD_DEPTH + levelBudget[0] + levelBudget[1]
      MOD_DEPTH = 14 for a UNIFORM_TERNARY secret: OpenFHE's degree-44 Chebyshev
      approximation of the modular reduction plus its 6 double-angle iterations
      (GetModDepthInternal). Below it, the levels left over go negative in unsigned
      arithmetic inside EvalBootstrapSetup() and the process SEGFAULTS. Run
      `BKT_DEPTH=11 tests/diag_bootstrap.py` to watch it happen.

  firstModSize - scalingModSize <= the correction factor
      OpenFHE derives that factor from the ring and slot counts and clamps it to 7..14
      (~9 here). 60/50 gives 10, one too many: the CPU path throws "Degree [10] must be
      less than or equal to the correction factor [9]", the GPU path silently decrypts
      to noise. 60/59 -- the OpenFHE bootstrapping recipe -- gives 1.

Level budgets are not interchangeable either: {2, 2} and {5, 5} decrypt to noise on the
GPU at 2048 slots, while {1, 1}, {3, 3} and {4, 4} are fine, and all of them are fine on
the CPU or at 1024 / 4096 slots. Left as a knob (BKT_LEVEL_BUDGET) because it is the one
parameter here whose failure is still unexplained.

  BKT_DEVICE=0  BKT_RING=16  BKT_SLOTS_DIV=1|2  BKT_BUDGET=none|256MiB
  BKT_LEVEL_BUDGET=3,3  BKT_DEPTH=<override>
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe  # noqa: E402

MOD_DEPTH = 14

DEV = int(os.environ.get("BKT_DEVICE", "0"))
RING = int(os.environ.get("BKT_RING", "16"))
DIV = int(os.environ.get("BKT_SLOTS_DIV", "2"))
bud = os.environ.get("BKT_BUDGET", "256MiB")
LEVEL_BUDGET = [int(x) for x in os.environ.get("BKT_LEVEL_BUDGET", "3,3").split(",")]
DEPTH = int(os.environ.get("BKT_DEPTH", str(MOD_DEPTH + sum(LEVEL_BUDGET) + 2)))
BATCH = 1 << (RING - 1)


def size(s):
    if s == "none":
        return None
    mult = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[s[-3:]]
    return int(s[:-3]) * mult


print(f"ring=2^{RING} batch={BATCH} budget={bud} slots_div={DIV} "
      f"level_budget={LEVEL_BUDGET} depth={DEPTH} "
      f"(needs >= {MOD_DEPTH + sum(LEVEL_BUDGET)})", flush=True)
p = fhe.CCParams()
p.SetSecurityLevel(fhe.HEStd_NotSet)
p.SetRingDim(1 << RING)
p.SetMultiplicativeDepth(DEPTH)
p.SetScalingModSize(59)
p.SetFirstModSize(60)
p.SetNumLargeDigits(3)
p.SetBatchSize(BATCH)
p.SetScalingTechnique(fhe.FLEXIBLEAUTO)
p.SetKeySwitchTechnique(fhe.HYBRID)
p.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
p.SetDevices([DEV])
cc = fhe.GenCryptoContext(p)
for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
    cc.Enable(f)

keys = cc.KeyGen()
print("KeyGen ok", flush=True)
cc.EvalMultKeyGen(keys.secretKey)
print("EvalMultKeyGen ok", flush=True)

budget = size(bud)
if budget is not None:
    cc.SetRotationKeyCache(budget)
    print(f"SetRotationKeyCache({budget:,}) ok", flush=True)

slots = BATCH // DIV
t0 = time.time()
cc.EvalBootstrapSetup(LEVEL_BUDGET, [0, 0], slots)
print(f"EvalBootstrapSetup ok ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
cc.EvalBootstrapKeyGen(keys.secretKey, slots)
print(f"EvalBootstrapKeyGen ok ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
cc.LoadContext(keys.publicKey)
print(f"LoadContext ok ({time.time() - t0:.1f}s); resident={cc.GetRotationKeyCacheResidentBytes():,}", flush=True)

# At the bottom of the modulus chain: bootstrapping a ciphertext that still has all its
# levels proves nothing, and the level it comes back at is the interesting part.
data = [0.25 + 1e-4 * (i % 100) for i in range(slots)]
ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(data, 1, DEPTH - 1, slots))
print(f"Encrypt ok (level {ct.GetLevel()} of {DEPTH})", flush=True)

t0 = time.time()
bt = cc.EvalBootstrap(ct)
print(f"EvalBootstrap #1 ok ({time.time() - t0:.1f}s); level {bt.GetLevel()}; "
      f"resident={cc.GetRotationKeyCacheResidentBytes():,}", flush=True)
pt = cc.Decrypt(keys.secretKey, bt)
pt.SetLength(8)
got = list(pt.GetRealPackedValue())
print(f"result {got[:4]}  expected {data[:4]}  maxerr "
      f"{max(abs(a - b) for a, b in zip(got, data)):.2e}", flush=True)

cc.OffloadRotationKeys()
print(f"OffloadRotationKeys ok; resident={cc.GetRotationKeyCacheResidentBytes():,}", flush=True)
t0 = time.time()
bt2 = cc.EvalBootstrap(ct)
print(f"EvalBootstrap #2 (cold) ok ({time.time() - t0:.1f}s)", flush=True)
pt2 = cc.Decrypt(keys.secretKey, bt2)
pt2.SetLength(8)
got2 = list(pt2.GetRealPackedValue())
print(f"cold result {got2[:4]}; |cold-warm|={max(abs(a - b) for a, b in zip(got, got2)):.2e}", flush=True)
print("ALL OK", flush=True)
