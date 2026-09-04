#!/usr/bin/env python3
"""Exercise the rotation-key VRAM cache (SetRotationKeyCache and friends).

Every scenario runs in its own subprocess (`--all`), so a crash pinpoints exactly which
behaviour broke instead of killing the whole run. A crash is reported as `exit=-11`.

    python tests/test_rotation_key_cache.py --all
    python tests/test_rotation_key_cache.py --scenario lru
    RKTEST_BOOTSTRAP=1 python tests/test_rotation_key_cache.py --scenario bootstrap

Env: RKTEST_DEVICE (GPU index), RKTEST_RING (log2 ring dim, default 13).
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fideslib_py as fhe  # noqa: E402

DEVICE = int(os.environ.get("RKTEST_DEVICE", "0"))
RING = int(os.environ.get("RKTEST_RING", "13"))
BATCH = 1 << (RING - 1)
ROT_IDXS = [1, 2, 3, 5, 8, 13, 21, 34]


class Skipped(Exception):
    pass


# ---------------------------------------------------------------- helpers
def make_context(mult_depth: int = 6):
    params = fhe.CCParams()
    params.SetSecurityLevel(fhe.HEStd_NotSet)
    params.SetRingDim(1 << RING)
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    params.SetNumLargeDigits(3)
    params.SetBatchSize(BATCH)
    params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
    params.SetKeySwitchTechnique(fhe.HYBRID)
    params.SetDevices([DEVICE])
    cc = fhe.GenCryptoContext(params)
    for feat in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE):
        cc.Enable(feat)
    return cc


def load(cc, budget, idxs=ROT_IDXS):
    """KeyGen -> SetRotationKeyCache -> EvalRotateKeyGen -> LoadContext.

    The budget MUST be set before LoadContext: only keys built while a finite budget is
    in force keep the host snapshot that offload/reload round-trips through.
    """
    keys = cc.KeyGen()
    if budget is not None:
        cc.SetRotationKeyCache(budget)
    cc.EvalRotateKeyGen(keys.secretKey, list(idxs))
    cc.LoadContext(keys.publicKey)
    return keys


def vec(n=BATCH):
    return [float(i % 37) - 18.0 for i in range(n)]


def decrypt(cc, sk, ct, n=64):
    pt = cc.Decrypt(sk, ct)
    pt.SetLength(n)
    return list(pt.GetRealPackedValue())


def maxdiff(a, b):
    assert len(a) == len(b), f"length mismatch {len(a)} vs {len(b)}"
    return max(abs(x - y) for x, y in zip(a, b))


def noise_floor(cc, sk, ct, index=1, reps=2):
    """FIDESlib rotations are not bit-reproducible run to run (multi-stream reductions),
    so 'bit-exact' can only ever mean 'within the op's own reproducibility floor'.
    Measured here: two identical rotations of the same ciphertext with the same key."""
    ref = decrypt(cc, sk, cc.EvalRotate(ct, index))
    worst = 0.0
    for _ in range(reps):
        worst = max(worst, maxdiff(ref, decrypt(cc, sk, cc.EvalRotate(ct, index))))
    return worst


def resident(cc, idxs=ROT_IDXS):
    return [i for i in idxs if cc.IsRotationKeyResident(i)]


def measure_key_bytes(cc, keys, ct):
    """Bytes of a single rotation key, measured IN THIS context.

    Do not measure in a second context: FIDESlib caches GPU contexts by Parameters
    (GenCryptoContextGPU returns map_param_context[param]), so two CryptoContextImpl
    built from identical params share one ContextData -- its budget, its accounting and
    its LRU would then be shared and the measurement meaningless. See s_params_sharing.
    """
    assert cc.GetRotationKeyCacheResidentBytes() == 0, "measure on a context with no resident keys"
    cc.EvalRotate(ct, ROT_IDXS[0])
    nbytes = cc.GetRotationKeyCacheResidentBytes()
    assert nbytes > 0
    return nbytes


# ---------------------------------------------------------------- scenarios
def s_defaults(scn):
    """No budget: legacy behaviour, every rotation key permanently resident."""
    cc = make_context()
    assert cc.GetRotationKeyCache() is None, "default budget should be unlimited/None"
    keys = load(cc, None)

    b0 = cc.GetRotationKeyCacheResidentBytes()
    scn.log(f"resident bytes after LoadContext: {b0:,}")
    assert b0 > 0, "with no budget the keys must be resident right away"
    assert resident(cc) == list(ROT_IDXS), "with no budget every key must be resident"

    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, 5))
    exp = [vec()[(j + 5) % BATCH] for j in range(64)]
    err = maxdiff(got, exp)
    scn.log(f"rotate(5) error vs exact: {err:.2e}")
    assert err < 1e-6, f"rotation with no budget is wrong ({err:.2e})"
    assert cc.GetRotationKeyCacheResidentBytes() == b0, "no-budget rotation moved bytes"
    scn.ok("legacy (unbounded) behaviour intact")


def s_lazy(scn):
    """Keys built under a budget start VRAM-free and load on first use."""
    cc = make_context()
    big = 1 << 40  # budget that never binds: only laziness is exercised
    keys = load(cc, big)
    assert cc.GetRotationKeyCache() == big

    b0 = cc.GetRotationKeyCacheResidentBytes()
    scn.log(f"resident bytes after LoadContext: {b0:,}")
    assert b0 == 0, f"lazy keys must not occupy VRAM before first use (got {b0:,})"
    assert resident(cc) == [], "lazy keys must start offloaded"

    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, 5))
    exp = [vec()[(j + 5) % BATCH] for j in range(64)]
    assert maxdiff(got, exp) < 1e-6, "lazy rotation is wrong"
    assert resident(cc) == [5], f"used key must be resident, got {resident(cc)}"
    b1 = cc.GetRotationKeyCacheResidentBytes()
    scn.log(f"resident bytes after one rotation: {b1:,}")
    assert 0 < b1 < big, f"one key should be accounted for ({b1:,})"

    cc.EvalRotate(ct, 5)  # re-hit must not re-account
    assert cc.GetRotationKeyCacheResidentBytes() == b1, "re-hit changed resident bytes"
    scn.ok("lazy creation + on-demand load")


def s_exact(scn):
    """Offload/reload round trip costs nothing beyond the op's own noise floor."""
    cc = make_context()
    keys = load(cc, 1 << 40)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    floor = noise_floor(cc, keys.secretKey, ct, index=13)
    scn.log(f"reproducibility floor (no cache involved): {floor:.2e}")

    ref = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, 13))
    for cycle in range(3):
        cc.OffloadRotationKeys()
        assert cc.GetRotationKeyCacheResidentBytes() == 0, "offload-all left bytes accounted"
        assert resident(cc) == [], "offload-all left keys resident"
        got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, 13))
        d = maxdiff(ref, got)
        scn.log(f"cycle {cycle}: |cold - warm| = {d:.2e}")
        assert d <= max(4 * floor, 1e-9), f"reload changed the result: {d:.2e} >> {floor:.2e}"
    scn.ok("3x offload/reload round trip within the noise floor")


def s_lru(scn):
    """A one-key budget forces eviction; results stay correct across misses."""
    cc = make_context()
    keys = load(cc, 1 << 40)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    one_key = measure_key_bytes(cc, keys, ct)
    floor = noise_floor(cc, keys.secretKey, ct, index=1)
    scn.log(f"one rotation key = {one_key:,} bytes")
    cc.SetRotationKeyCache(one_key)  # exactly one key fits
    ref = {}
    for i in ROT_IDXS:
        got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, i))
        exp = [vec()[(j + i) % BATCH] for j in range(64)]
        assert maxdiff(got, exp) < 1e-6, f"rotation {i} wrong under a one-key budget"
        ref[i] = got
        nbytes = cc.GetRotationKeyCacheResidentBytes()
        scn.log(f"after rotate({i}): resident={resident(cc)} bytes={nbytes:,}")
        assert nbytes <= 2 * one_key, f"budget blown: {nbytes:,} > {2 * one_key:,}"
        assert i in resident(cc), f"just-used key {i} is not resident"

    # Second pass = every access misses; must reproduce the first pass.
    for i in ROT_IDXS:
        got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, i))
        d = maxdiff(ref[i], got)
        assert d <= max(4 * floor, 1e-9), f"second-pass rotation {i} differs ({d:.2e})"
    scn.ok("LRU eviction under a one-key budget; misses reproduce")


def s_pin(scn):
    """Pinned keys survive eviction pressure."""
    cc = make_context()
    keys = load(cc, 1 << 40)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    one_key = measure_key_bytes(cc, keys, ct)
    cc.SetRotationKeyCache(one_key)

    # Pin a key that is not resident: pinning alone must not be forced to load it.
    cc.PinRotationKey(2)
    decrypt(cc, keys.secretKey, cc.EvalRotate(ct, 2))  # bring it in
    assert cc.IsRotationKeyResident(2)
    for i in ROT_IDXS:
        decrypt(cc, keys.secretKey, cc.EvalRotate(ct, i))
        assert cc.IsRotationKeyResident(2), f"pinned key 2 evicted after rotate({i})"
        nbytes = cc.GetRotationKeyCacheResidentBytes()
        assert nbytes <= 2 * one_key, f"budget blown with a pin: {nbytes:,}"

    # Unpin -> the LRU pass must be able to evict it again.
    cc.PinRotationKey(2, pin=False)
    for i in ROT_IDXS:
        decrypt(cc, keys.secretKey, cc.EvalRotate(ct, i))
    scn.log(f"after unpin + one full pass: resident={resident(cc)}")
    assert not cc.IsRotationKeyResident(2), "unpinned key was never evicted"

    # A pinned key must still decrypt correctly.
    got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, 2))
    exp = [vec()[(j + 2) % BATCH] for j in range(64)]
    assert maxdiff(got, exp) < 1e-6, "rotation with a pinned key is wrong"
    scn.ok("pin / unpin under eviction pressure")


def s_manual(scn):
    """Manual offload of a subset, unknown indexes, repeated offloads."""
    cc = make_context()
    keys = load(cc, 1 << 40)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    for i in ROT_IDXS:
        decrypt(cc, keys.secretKey, cc.EvalRotate(ct, i))
    assert resident(cc) == list(ROT_IDXS), "not all keys became resident"

    cc.OffloadRotationKeys([ROT_IDXS[0], ROT_IDXS[-1]])
    assert not cc.IsRotationKeyResident(ROT_IDXS[0])
    assert not cc.IsRotationKeyResident(ROT_IDXS[-1])
    assert resident(cc) == list(ROT_IDXS[1:-1]), "offload spilled to other keys"

    # Unknown / negative indexes must be ignored, not crash.
    cc.OffloadRotationKeys([9999, -12345])
    cc.PinRotationKey(9999)
    assert cc.IsRotationKeyResident(9999) is False, "unknown index reported resident"
    cc.OffloadRotationKeys()
    cc.OffloadRotationKeys()  # idempotent
    assert cc.GetRotationKeyCacheResidentBytes() == 0
    got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, ROT_IDXS[0]))
    exp = [vec()[(j + ROT_IDXS[0]) % BATCH] for j in range(64)]
    assert maxdiff(got, exp) < 1e-6, "rotation after manual offload is wrong"
    scn.ok("manual offload / unknown indexes / idempotence")


def s_unloaded(scn):
    """Cache calls on a context that was never loaded: clean errors, no crash."""
    cc = make_context()
    assert cc.GetRotationKeyCacheResidentBytes() == 0
    assert cc.IsRotationKeyResident(1) is False
    for fn, args in ((cc.OffloadRotationKeys, ()), (cc.PinRotationKey, (1,))):
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001
            scn.log(f"{fn.__name__}{args} -> {type(e).__name__}: {str(e)[:70]}")
        else:
            scn.log(f"{fn.__name__}{args} -> no error (accepted)")
    cc.SetRotationKeyCache(1 << 30)
    assert cc.GetRotationKeyCache() == 1 << 30
    scn.ok("cache API on an unloaded context")


def s_accumulate(scn):
    """Hoisted rotation (AccumulateSum fetches many keys before launching anything)."""
    cc = make_context(mult_depth=8)
    slots = 64
    idxs = sorted(set(fhe.accumulate_rotation_indices(slots, stride=1, bstep=4)))
    scn.log(f"{len(idxs)} accumulate rotation indices: {idxs}")
    keys = load(cc, 1 << 40, idxs)
    data = [float(i % 11) + 0.5 for i in range(BATCH)]
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(data))

    warm = decrypt(cc, keys.secretKey, cc.AccumulateSum(ct, slots, 1), n=8)
    expect = sum(data[:slots])
    scn.log(f"warm: slot0={warm[0]:.4f} expected={expect:.4f}")
    assert abs(warm[0] - expect) < 1e-3, "AccumulateSum is wrong (no budget pressure)"

    # Absurdly tight budget -> every key of the hoisted batch misses and reloads.
    cc.SetRotationKeyCache(1)
    cc.OffloadRotationKeys()
    cold = decrypt(cc, keys.secretKey, cc.AccumulateSum(ct, slots, 1), n=8)
    d = maxdiff(warm, cold)
    scn.log(f"cold (1-byte budget): slot0={cold[0]:.4f} |cold-warm|={d:.2e}")
    assert abs(cold[0] - expect) < 1e-3, "AccumulateSum under a tight budget is wrong"
    assert d <= 1e-6, f"hoisted rotation changed the result ({d:.2e})"
    scn.ok("hoisted rotation with a 1-byte budget")


def s_budget_change(scn):
    """Runtime budget changes: tighten (evicts), widen, None (unlimited), zero."""
    cc = make_context()
    keys = load(cc, 1 << 40)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    for i in ROT_IDXS:
        decrypt(cc, keys.secretKey, cc.EvalRotate(ct, i))
    full = cc.GetRotationKeyCacheResidentBytes()
    scn.log(f"all {len(ROT_IDXS)} keys resident: {full:,} bytes")

    one_key = full // len(ROT_IDXS)
    cc.SetRotationKeyCache(one_key)
    b = cc.GetRotationKeyCacheResidentBytes()
    scn.log(f"after tightening to {one_key:,}: {b:,} bytes, resident={resident(cc)}")
    assert b <= 2 * one_key, f"runtime tighten did not evict ({b:,})"

    cc.SetRotationKeyCache(None)
    assert cc.GetRotationKeyCache() is None
    got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, 8))
    exp = [vec()[(j + 8) % BATCH] for j in range(64)]
    assert maxdiff(got, exp) < 1e-6, "rotation after SetRotationKeyCache(None) is wrong"

    cc.SetRotationKeyCache(0)
    decrypt(cc, keys.secretKey, cc.EvalRotate(ct, 21))
    b = cc.GetRotationKeyCacheResidentBytes()
    scn.log(f"zero budget, after a rotation: {b:,} bytes resident={resident(cc)}")
    assert b <= one_key * 2, "zero budget blew up"

    # Back to unlimited: previously-evicted keys must still work.
    cc.SetRotationKeyCache(None)
    for i in ROT_IDXS:
        got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, i))
        exp = [vec()[(j + i) % BATCH] for j in range(64)]
        assert maxdiff(got, exp) < 1e-6, f"rotation {i} after budget churn is wrong"
    scn.ok("runtime budget changes incl. None and 0")


def s_stress(scn):
    """Churn: many rotations over a tiny budget, trim in the middle, keep going."""
    cc = make_context()
    keys = load(cc, 1 << 20)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    floor = noise_floor(cc, keys.secretKey, ct, index=1)
    ref = {}
    t0 = time.time()
    for rep in range(4):
        for i in ROT_IDXS:
            got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, i))
            if rep == 0:
                ref[i] = got
            else:
                d = maxdiff(ref[i], got)
                assert d <= max(4 * floor, 1e-9), f"rep {rep} rotate({i}) diverged ({d:.2e})"
    scn.log(f"{4 * len(ROT_IDXS)} rotations in {time.time() - t0:.1f}s (floor {floor:.1e})")
    cc.TrimGPUMemoryPool()
    got = decrypt(cc, keys.secretKey, cc.EvalRotate(ct, ROT_IDXS[3]))
    assert maxdiff(ref[ROT_IDXS[3]], got) <= max(4 * floor, 1e-9), "diverged after trim"
    scn.ok("churn + TrimGPUMemoryPool")


def s_two_contexts(scn):
    """Two contexts with different budgets keep separate bookkeeping."""
    a = make_context()
    keys_a = load(a, 1 << 40)
    ct_a = a.Encrypt(keys_a.publicKey, a.MakeCKKSPackedPlaintext(vec()))
    a.EvalRotate(ct_a, 5)
    bytes_a = a.GetRotationKeyCacheResidentBytes()

    b = make_context(mult_depth=7)  # different params -> distinct GPU context
    keys_b = load(b, 1 << 20)
    ct_b = b.Encrypt(keys_b.publicKey, b.MakeCKKSPackedPlaintext(vec()))
    b.EvalRotate(ct_b, 5)
    bytes_b = b.GetRotationKeyCacheResidentBytes()
    scn.log(f"context A: {bytes_a:,} bytes, context B: {bytes_b:,} bytes")
    assert bytes_a > 0 and bytes_b > 0

    # Interleave ops on both contexts; bookkeeping must not cross-talk.
    for i in (1, 2, 1, 3):
        a.EvalRotate(ct_a, i)
        b.EvalRotate(ct_b, i)
    ga = decrypt(a, keys_a.secretKey, a.EvalRotate(ct_a, 5))
    gb = decrypt(b, keys_b.secretKey, b.EvalRotate(ct_b, 5))
    exp = [vec()[(j + 5) % BATCH] for j in range(64)]
    assert maxdiff(ga, exp) < 1e-6 and maxdiff(gb, exp) < 1e-6, "cross-context rotation wrong"
    a.OffloadRotationKeys()
    assert a.GetRotationKeyCacheResidentBytes() == 0, "offloading A touched B's accounting"
    assert b.GetRotationKeyCacheResidentBytes() > 0, "offloading A emptied B"
    scn.ok("two contexts with independent budgets")


def s_params_sharing(scn):
    """Hazard check: two CryptoContextImpl with IDENTICAL params share one GPU ContextData.

    FIDESlib's GenCryptoContextGPU() returns the already-registered context for equal
    Parameters, so the second LoadContext() silently replaces the first one's budget and
    both report one shared resident-byte counter. Asserted here as documented behaviour --
    if it ever changes, this test says so instead of a caller being surprised.
    """
    a = make_context()
    keys_a = load(a, 1 << 40)
    ct_a = a.Encrypt(keys_a.publicKey, a.MakeCKKSPackedPlaintext(vec()))
    a.EvalRotate(ct_a, 5)

    b = make_context()  # identical params -> the SAME underlying GPU context
    keys_b = load(b, None)  # unlimited: LoadContext pushes SIZE_MAX onto the shared context
    scn.log(f"A reports {a.GetRotationKeyCache()}, B reports {b.GetRotationKeyCache()} "
            f"(impl-level fields, not the shared context's effective budget)")
    assert a.GetRotationKeyCacheResidentBytes() == b.GetRotationKeyCacheResidentBytes(), \
        "accounting is not shared -> behaviour changed"
    scn.log(f"shared resident bytes: {a.GetRotationKeyCacheResidentBytes():,}")

    # Tighten through A: because the budget lives in the shared ContextData it binds B too.
    ct_b = b.Encrypt(keys_b.publicKey, b.MakeCKKSPackedPlaintext(vec()))
    a.SetRotationKeyCache(1 << 20)  # well under one key -> the shared cache must evict
    scn.log(f"tightened through A to 1 MiB; B sees GetRotationKeyCache()={b.GetRotationKeyCache()}")
    for i in ROT_IDXS:
        b.EvalRotate(ct_b, i)
        a.EvalRotate(ct_a, i)  # must stay correct despite the shared cache
        shared = a.GetRotationKeyCacheResidentBytes()
        assert shared == b.GetRotationKeyCacheResidentBytes(), "accounting diverged"
    scn.log(f"shared resident bytes after interleaved ops: {a.GetRotationKeyCacheResidentBytes():,}")
    assert a.IsRotationKeyResident(3) == b.IsRotationKeyResident(3), "residency views diverged"

    got_a = decrypt(a, keys_a.secretKey, a.EvalRotate(ct_a, 5))
    got_b = decrypt(b, keys_b.secretKey, b.EvalRotate(ct_b, 5))
    exp = [vec()[(j + 5) % BATCH] for j in range(64)]
    assert maxdiff(got_a, exp) < 1e-6 and maxdiff(got_b, exp) < 1e-6, "shared-context rotation wrong"
    scn.log("NOTE: with two key sets in one ContextData, IsRotationKeyResident(i) answers for "
            "the first set that holds i -- it cannot tell which context it belongs to")
    scn.ok("identical params share one GPU context (budget + accounting shared, results correct)")


def s_no_leak(scn):
    """A miss must not leak: host RSS stays flat over repeated offload/reload churn."""
    import resource

    cc = make_context()
    keys = load(cc, 1 << 20)  # far below one key -> every rotation is a miss
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    for _ in range(5):
        for i in ROT_IDXS:
            cc.EvalRotate(ct, i)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    for _ in range(30):
        for i in ROT_IDXS:
            cc.EvalRotate(ct, i)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    scn.log(f"host RSS {rss0} MiB -> {rss1} MiB over 240 rotations (8 key snapshots ~ "
            f"{8 * 3932160 // 1024 // 1024} MiB held on purpose)")
    assert rss1 - rss0 < 128, f"host memory grew {rss1 - rss0} MiB -- reload is leaking"
    scn.ok("no host-memory leak across 240 forced misses")


def s_bootstrap(scn):
    """The real use case: hundreds of bootstrap keys under a byte budget."""
    if os.environ.get("RKTEST_BOOTSTRAP") != "1":
        raise Skipped("set RKTEST_BOOTSTRAP=1 to run")
    cc = make_context(mult_depth=11)
    cc.Enable(fhe.FHE)
    slots = BATCH // 2
    budget = int(os.environ.get("RKTEST_BOOTSTRAP_BUDGET", str(256 * 1024 * 1024)))
    keys = cc.KeyGen()
    cc.SetRotationKeyCache(budget)
    cc.EvalMultKeyGen(keys.secretKey)
    # Both of these must happen before LoadContext (the api throws otherwise).
    cc.EvalBootstrapSetup([5, 5], [0, 0], slots)
    cc.EvalBootstrapKeyGen(keys.secretKey, slots)
    t0 = time.time()
    cc.LoadContext(keys.publicKey)
    scn.log(f"LoadContext with bootstrap keys: {time.time() - t0:.1f}s")
    scn.log(f"resident after load: {cc.GetRotationKeyCacheResidentBytes():,} (budget {budget:,})")

    data = [0.25 + 1e-4 * i for i in range(slots)]
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(data))
    t0 = time.time()
    bt = cc.EvalBootstrap(ct)
    scn.log(f"bootstrap #1: {time.time() - t0:.1f}s")
    got = decrypt(cc, keys.secretKey, bt, n=64)
    scn.log(f"resident after bootstrap #1: {cc.GetRotationKeyCacheResidentBytes():,}")
    err = maxdiff(got, data[:64])
    scn.log(f"bootstrap error: {err:.2e}")
    assert err < 1e-4, f"bootstrap is wrong ({err:.2e})"

    cc.OffloadRotationKeys()
    t0 = time.time()
    bt2 = cc.EvalBootstrap(ct)
    scn.log(f"bootstrap #2 (cold keys): {time.time() - t0:.1f}s")
    got2 = decrypt(cc, keys.secretKey, bt2, n=64)
    d = maxdiff(got, got2)
    scn.log(f"|cold - warm| = {d:.2e}")
    assert d < 1e-6, f"cold-key bootstrap diverges from warm ({d:.2e})"
    scn.ok("bootstrap with a bounded rotation-key cache")


SCENARIOS = {
    "defaults": s_defaults,
    "lazy": s_lazy,
    "exact": s_exact,
    "lru": s_lru,
    "pin": s_pin,
    "manual": s_manual,
    "unloaded": s_unloaded,
    "accumulate": s_accumulate,
    "budget_change": s_budget_change,
    "stress": s_stress,
    "two_contexts": s_two_contexts,
    "params_sharing": s_params_sharing,
    "no_leak": s_no_leak,
    "bootstrap": s_bootstrap,
}


class Scn:
    def __init__(self, name):
        self.name = name

    def log(self, msg):
        print(f"  [{self.name}] {msg}", flush=True)

    def ok(self, msg):
        print(f"PASS [{self.name}] {msg}", flush=True)


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        failed = []
        for name in SCENARIOS:
            print(f"=== {name}", flush=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, __file__, "--scenario", name])
            dt = time.time() - t0
            if r.returncode != 0:
                failed.append((name, r.returncode))
        print("\n===== summary =====")
        for name, code in failed:
            why = f"CRASHED (signal {-code})" if code < 0 else f"failed (exit {code})"
            print(f"  {name}: {why}")
        print(f"{len(SCENARIOS) - len(failed)}/{len(SCENARIOS)} scenarios passed")
        return 1 if failed else 0

    if args.scenario not in SCENARIOS:
        print(f"unknown scenario {args.scenario!r}; choose from {', '.join(SCENARIOS)}")
        return 2
    try:
        SCENARIOS[args.scenario](Scn(args.scenario))
    except Skipped as e:
        print(f"SKIP [{args.scenario}] {e}", flush=True)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
