#!/usr/bin/env python3
"""Diagnostic: how far off is a rotation after an offload/reload round trip?"""
import faulthandler
import os
import sys
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe  # noqa: E402

DEV = int(os.environ.get("RKTEST_DEVICE", "0"))
RING = 13
BATCH = 1 << (RING - 1)
IDX = [1, 2, 3, 5, 8, 13, 21, 34]


def mk():
    p = fhe.CCParams()
    p.SetSecurityLevel(fhe.HEStd_NotSet)
    p.SetRingDim(1 << RING)
    p.SetMultiplicativeDepth(6)
    p.SetScalingModSize(50)
    p.SetFirstModSize(60)
    p.SetNumLargeDigits(3)
    p.SetBatchSize(BATCH)
    p.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    p.SetKeySwitchTechnique(fhe.HYBRID)
    p.SetDevices([DEV])
    cc = fhe.GenCryptoContext(p)
    for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE):
        cc.Enable(f)
    return cc


def dec(cc, sk, ct, n=64):
    pt = cc.Decrypt(sk, ct)
    pt.SetLength(n)
    return list(pt.GetRealPackedValue())


def d(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


cc = mk()
keys = cc.KeyGen()
cc.SetRotationKeyCache(1 << 40)
cc.EvalRotateKeyGen(keys.secretKey, IDX)
cc.LoadContext(keys.publicKey)
data = [float(i % 37) - 18.0 for i in range(BATCH)]
ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(data))

print("--- A: same key twice, NO offload in between (determinism baseline)")
for i in IDX[:3]:
    a = dec(cc, keys.secretKey, cc.EvalRotate(ct, i))
    b = dec(cc, keys.secretKey, cc.EvalRotate(ct, i))
    print(f"  rotate({i}): maxdiff={d(a, b):.3e}   first: {a[0]!r} vs {b[0]!r}")

print("--- B: offload all, rotate again (reload path)")
for i in IDX[:3]:
    warm = dec(cc, keys.secretKey, cc.EvalRotate(ct, i))
    cc.OffloadRotationKeys([i])
    assert not cc.IsRotationKeyResident(i)
    cold = dec(cc, keys.secretKey, cc.EvalRotate(ct, i))
    print(f"  rotate({i}): maxdiff={d(warm, cold):.3e}   first: {warm[0]!r} vs {cold[0]!r}")

print("--- C: expected values")
for i in IDX[:3]:
    got = dec(cc, keys.secretKey, cc.EvalRotate(ct, i))
    exp = [data[(j + i) % BATCH] for j in range(64)]
    print(f"  rotate({i}): maxerr vs exact={d(got, exp):.3e}")

print("--- D: repeated offload/reload of the same key")
prev = dec(cc, keys.secretKey, cc.EvalRotate(ct, 1))
for k in range(5):
    cc.OffloadRotationKeys()
    got = dec(cc, keys.secretKey, cc.EvalRotate(ct, 1))
    print(f"  cycle {k}: maxdiff vs prev={d(prev, got):.3e} first={got[0]!r}")
    prev = got
