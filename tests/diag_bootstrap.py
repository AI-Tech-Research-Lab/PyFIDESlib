#!/usr/bin/env python3
"""Narrow down the bootstrap crash: is it the rotation-key cache or plain bootstrap?

KNOWN, PRE-EXISTING (not a rotation-cache bug): `EvalBootstrapSetup()` segfaults through this
Python API for every parameter combination tried (ring 2^15/2^16, slots N/2 and N/4,
FLEXIBLEAUTOEXT and FIXEDMANUAL) -- with AND without a rotation-key budget set. The crash is
in the bootstrap setup call itself, before any key is created. Left here to reproduce it.

  BKT_BUDGET=none|256MiB   BKT_RING=15|16   BKT_SLOTS_DIV=1|2
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe  # noqa: E402

DEV = int(os.environ.get("BKT_DEVICE", "0"))
RING = int(os.environ.get("BKT_RING", "16"))
DIV = int(os.environ.get("BKT_SLOTS_DIV", "2"))
bud = os.environ.get("BKT_BUDGET", "256MiB")
BATCH = 1 << (RING - 1)


def size(s):
    if s == "none":
        return None
    mult = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[s[-3:]]
    return int(s[:-3]) * mult


print(f"ring=2^{RING} batch={BATCH} budget={bud} slots_div={DIV}", flush=True)
p = fhe.CCParams()
p.SetSecurityLevel(fhe.HEStd_NotSet)
p.SetRingDim(1 << RING)
p.SetMultiplicativeDepth(11)
p.SetScalingModSize(50)
p.SetFirstModSize(60)
p.SetNumLargeDigits(3)
p.SetBatchSize(BATCH)
p.SetScalingTechnique(fhe.FLEXIBLEAUTOEXT)
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
cc.EvalBootstrapSetup([5, 5], [0, 0], slots)
print(f"EvalBootstrapSetup ok ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
cc.EvalBootstrapKeyGen(keys.secretKey, slots)
print(f"EvalBootstrapKeyGen ok ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
cc.LoadContext(keys.publicKey)
print(f"LoadContext ok ({time.time() - t0:.1f}s); resident={cc.GetRotationKeyCacheResidentBytes():,}", flush=True)

data = [0.25 + 1e-4 * i for i in range(slots)]
ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(data))
print("Encrypt ok", flush=True)

t0 = time.time()
bt = cc.EvalBootstrap(ct)
print(f"EvalBootstrap #1 ok ({time.time() - t0:.1f}s); resident={cc.GetRotationKeyCacheResidentBytes():,}", flush=True)
pt = cc.Decrypt(keys.secretKey, bt)
pt.SetLength(8)
got = list(pt.GetRealPackedValue())
print(f"result {got[:4]}  expected {data[:4]}", flush=True)

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
