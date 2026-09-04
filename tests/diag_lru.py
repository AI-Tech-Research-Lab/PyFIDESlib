#!/usr/bin/env python3
"""Diagnostic: LRU eviction with a SINGLE context (no second context sharing the GPU ctx).

Also probes whether two CryptoContextImpl with identical params share one GPU context
(FIDESlib GenCryptoContextGPU returns map_param_context[param] when one already exists).
"""
import faulthandler
import os
import sys
from pathlib import Path

faulthandler.enable()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fideslib_py as fhe  # noqa: E402

DEV = int(os.environ.get("RKTEST_DEVICE", "0"))
RING = int(os.environ.get("RKTEST_RING", "13"))
BATCH = 1 << (RING - 1)
IDX = [1, 2, 3, 5, 8, 13, 21, 34]


def mk(mult_depth=6):
    p = fhe.CCParams()
    p.SetSecurityLevel(fhe.HEStd_NotSet)
    p.SetRingDim(1 << RING)
    p.SetMultiplicativeDepth(mult_depth)
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


def load(cc, budget, idxs=IDX):
    keys = cc.KeyGen()
    if budget is not None:
        cc.SetRotationKeyCache(budget)
    cc.EvalRotateKeyGen(keys.secretKey, list(idxs))
    cc.LoadContext(keys.publicKey)
    return keys


def res(cc, idxs=IDX):
    return [i for i in idxs if cc.IsRotationKeyResident(i)]


print("### 1. single context: measure one key's size with a non-binding budget")
cc = mk()
keys = load(cc, 1 << 40)
ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([float(i % 7) for i in range(BATCH)]))
cc.EvalRotate(ct, 1)
K = cc.GetRotationKeyCacheResidentBytes()
print(f"   K (one key) = {K:,}")
for i in IDX:
    cc.EvalRotate(ct, i)
allb = cc.GetRotationKeyCacheResidentBytes()
print(f"   after touching all {len(IDX)} keys: {allb:,}  ({allb / K:.2f} x K), resident={res(cc)}")

print("### 2. same context, tighten to exactly one key, rotate through the indexes")
cc.SetRotationKeyCache(K)
print(f"   tightened -> {cc.GetRotationKeyCacheResidentBytes():,}, resident={res(cc)}")
for i in IDX:
    cc.EvalRotate(ct, i)
    b = cc.GetRotationKeyCacheResidentBytes()
    print(f"   rotate({i:>2}): bytes={b:,} ({b / K:.2f} x K) resident={res(cc)}")

print("### 3. same again (all cold)")
for i in IDX:
    cc.EvalRotate(ct, i)
    b = cc.GetRotationKeyCacheResidentBytes()
    print(f"   rotate({i:>2}): bytes={b:,} ({b / K:.2f} x K) resident={res(cc)}")

print("### 4. does a second CryptoContextImpl with identical params share the GPU context?")
cc2 = mk()
keys2 = load(cc2, None)  # no budget -> its keys are permanently resident
ct2 = cc2.Encrypt(keys2.publicKey, cc2.MakeCKKSPackedPlaintext([float(i % 5) for i in range(BATCH)]))
print(f"   cc (budget {cc.GetRotationKeyCache()}) bytes={cc.GetRotationKeyCacheResidentBytes():,}")
print(f"   cc2 (unlimited)              bytes={cc2.GetRotationKeyCacheResidentBytes():,}")
cc2.EvalRotate(ct2, 3)
print("   after cc2.EvalRotate(3):")
print(f"     cc  bytes={cc.GetRotationKeyCacheResidentBytes():,} resident={res(cc)}")
print(f"     cc2 bytes={cc2.GetRotationKeyCacheResidentBytes():,} resident={res(cc2)}")
print(f"   IsRotationKeyResident(3): cc={cc.IsRotationKeyResident(3)} cc2={cc2.IsRotationKeyResident(3)}")
