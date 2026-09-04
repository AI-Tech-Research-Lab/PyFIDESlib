#!/usr/bin/env python3
"""Exercise the plaintext VRAM cache (SetPlaintextCache and friends).

Every scenario runs in its own subprocess (`--all`), so a crash pinpoints exactly which
behaviour broke instead of killing the whole run. A crash is reported as `exit=-11`.

    python tests/test_plaintext_cache.py --all
    python tests/test_plaintext_cache.py --scenario lru

Env: PTTEST_DEVICE (physical GPU index, exported as CUDA_VISIBLE_DEVICES so the process
only ever touches that one GPU), PTTEST_RING (log2 ring dim, default 14).
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

# Confine the process to ONE physical GPU before the module is imported: importing it
# initializes CUDA on every visible device (and FIDESlib then allocates there), which is
# rude on a shared machine and pollutes the VRAM measurements below. Already-set
# CUDA_VISIBLE_DEVICES is left alone, so the `--all` subprocesses inherit this one.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PTTEST_DEVICE", "0"))

import fideslib_py as fhe  # noqa: E402

DEVICE = 0  # index within CUDA_VISIBLE_DEVICES, i.e. the one GPU selected above
RING = int(os.environ.get("PTTEST_RING", "14"))
BATCH = 1 << (RING - 1)
NPT = 8  # plaintexts used by most scenarios


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


def load(cc, budget=None):
    """KeyGen -> LoadContext -> SetPlaintextCache.

    Unlike the rotation-key cache the budget can be set at any time: a plaintext keeps its
    host-side encoding whether or not a budget is in force, so there is no snapshot to
    arrange up front and nothing to prepare before LoadContext().
    """
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.LoadContext(keys.publicKey)
    if budget is not None:
        cc.SetPlaintextCache(budget)
    return keys


def vec(k=0, n=BATCH):
    return [float((i + 3 * k) % 37) - 18.0 for i in range(n)]


def plaintexts(cc, count=NPT):
    return [cc.MakeCKKSPackedPlaintext(vec(k)) for k in range(count)]


def decrypt(cc, sk, ct, n=64):
    pt = cc.Decrypt(sk, ct)
    pt.SetLength(n)
    return list(pt.GetRealPackedValue())


def maxdiff(a, b):
    assert len(a) == len(b), f"length mismatch {len(a)} vs {len(b)}"
    return max(abs(x - y) for x, y in zip(a, b))


def resident_count(cc):
    return cc.GetDeviceObjectCounts()["plaintexts"]


def pool_live_bytes(cc=None):
    """VRAM held by live objects, as FIDESlib's own allocator sees it.

    Neither `cudaMemGetInfo` nor `nvidia-smi` can measure this: freeing a plaintext
    returns its limbs to FIDESlib's pool, not to the driver, and TrimGPUMemoryPool() only
    hands back whole idle slabs -- measured here, the reserved total and the driver's free
    total do not move at all across 32 plaintext loads. The pool's own live-chunk count
    does, and that is what a byte budget is meant to cap.
    """
    if cc is not None:
        cc.Synchronize()
    stats = fhe.GetGPUMemoryPoolStats(DEVICE)
    return sum(b["live_chunks"] * b["chunk_bytes"] for b in stats["buckets"])


def one_plaintext_bytes(cc, pt):
    """Bytes of a single device plaintext, as the cache accounts for them."""
    before = cc.GetPlaintextCacheResidentBytes()
    cc.LoadPlaintext(pt)
    nbytes = cc.GetPlaintextCacheResidentBytes() - before
    assert nbytes > 0, "loading a plaintext accounted for nothing"
    return nbytes


def noise_floor(cc, sk, ct, pt, reps=2):
    """FIDESlib ops are not necessarily bit-reproducible run to run (multi-stream
    reductions), so 'the re-upload changed nothing' can only mean 'within the op's own
    reproducibility floor'. Measured here on the same ct/pt pair, no cache pressure."""
    ref = decrypt(cc, sk, cc.EvalMult(ct, pt))
    worst = 0.0
    for _ in range(reps):
        worst = max(worst, maxdiff(ref, decrypt(cc, sk, cc.EvalMult(ct, pt))))
    return worst


# ---------------------------------------------------------------- scenarios
def s_defaults(scn):
    """No budget: legacy behaviour, every used plaintext stays on the device."""
    cc = make_context()
    assert cc.GetPlaintextCache() is None, "default budget should be unlimited/None"
    keys = load(cc)
    assert cc.GetPlaintextCacheResidentBytes() == 0, "nothing loaded yet"

    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    pts = plaintexts(cc)
    # Autoload is off by default: encoding a plaintext must not touch VRAM.
    assert cc.GetPlaintextCacheResidentBytes() == 0, "MakeCKKSPackedPlaintext loaded to VRAM"

    for k, pt in enumerate(pts):
        cc.EvalMult(ct, pt)
        assert resident_count(cc) == k + 1, f"plaintext {k} was not kept resident"
    full = cc.GetPlaintextCacheResidentBytes()
    scn.log(f"{NPT} plaintexts resident: {full:,} bytes ({full // NPT:,} each)")
    assert full > 0

    got = decrypt(cc, keys.secretKey, cc.EvalMult(ct, pts[0]))
    exp = [a * b for a, b in zip(vec()[:64], vec(0)[:64])]
    err = maxdiff(got, exp)
    scn.log(f"mult error vs exact: {err:.2e}")
    assert err < 1e-3, f"unbounded plaintext multiply is wrong ({err:.2e})"
    assert cc.GetPlaintextCacheResidentBytes() == full, "no-budget multiply moved bytes"
    scn.ok("legacy (unbounded) behaviour intact")


def s_accounting(scn):
    """GetPlaintextCacheResidentBytes matches the VRAM the driver actually hands out."""
    cc = make_context()
    keys = load(cc)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    cc.EvalMult(ct, cc.MakeCKKSPackedPlaintext(vec()))  # warm the pool / one-off allocations

    count = 32
    pts = plaintexts(cc, count)
    live0 = pool_live_bytes(cc)
    accounted0 = cc.GetPlaintextCacheResidentBytes()
    for k in range(count):
        cc.LoadPlaintext(pts[k])  # by index: a `for pt in pts` loop variable would outlive
    live1 = pool_live_bytes(cc)   # the list and keep the last plaintext from being dropped
    accounted = cc.GetPlaintextCacheResidentBytes() - accounted0
    measured = live1 - live0
    scn.log(f"{count} plaintexts: accounted {accounted:,} bytes, allocator {measured:,} "
            f"({accounted / count:,.0f} vs {measured / count:,.0f} per plaintext)")
    assert accounted > 0 and measured > 0
    ratio = accounted / measured
    assert 0.85 <= ratio <= 1.2, f"accounting is off by {ratio:.2f}x from the real allocation"

    # And the same VRAM comes back when the plaintexts are dropped.
    del pts
    live2 = pool_live_bytes(cc)
    scn.log(f"after dropping them: {cc.GetPlaintextCacheResidentBytes():,} accounted, "
            f"{(live1 - live2):,} bytes released by the allocator")
    assert cc.GetPlaintextCacheResidentBytes() == accounted0, "dropping left bytes accounted"
    assert live1 - live2 > 0.85 * measured, "the VRAM was not released"
    scn.ok("byte accounting tracks the real allocation")


def s_lru(scn):
    """A one-plaintext budget forces eviction; results stay correct across misses."""
    cc = make_context()
    keys = load(cc)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    pts = plaintexts(cc)
    one = one_plaintext_bytes(cc, pts[0])
    floor = noise_floor(cc, keys.secretKey, ct, pts[0])
    scn.log(f"one plaintext = {one:,} bytes, reproducibility floor {floor:.2e}")

    cc.SetPlaintextCache(one)  # exactly one plaintext fits
    ref = {}
    for k, pt in enumerate(pts):
        got = decrypt(cc, keys.secretKey, cc.EvalMult(ct, pt))
        exp = [a * b for a, b in zip(vec()[:64], vec(k)[:64])]
        assert maxdiff(got, exp) < 1e-3, f"multiply by plaintext {k} wrong under a 1-pt budget"
        ref[k] = got
        nbytes = cc.GetPlaintextCacheResidentBytes()
        scn.log(f"after mult({k}): resident={resident_count(cc)} bytes={nbytes:,}")
        assert nbytes <= 2 * one, f"budget blown: {nbytes:,} > {2 * one:,}"
        assert pt.IsLoadedOnDevice(), f"just-used plaintext {k} is not resident"

    # Second pass = every access misses and re-uploads; must reproduce the first pass.
    for k, pt in enumerate(pts):
        got = decrypt(cc, keys.secretKey, cc.EvalMult(ct, pt))
        d = maxdiff(ref[k], got)
        assert d <= max(4 * floor, 1e-9), f"second-pass mult({k}) differs ({d:.2e})"
    scn.log(f"second pass (all misses) reproduced the first within {max(4 * floor, 1e-9):.1e}")
    scn.ok("LRU eviction under a one-plaintext budget; misses reproduce")


def s_lru_order(scn):
    """Eviction takes the least recently USED plaintext, not the oldest loaded."""
    cc = make_context()
    keys = load(cc)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    pts = plaintexts(cc, 4)
    one = one_plaintext_bytes(cc, pts[0])
    for pt in pts:
        cc.LoadPlaintext(pt)
    assert resident_count(cc) == 4

    # Touch 0 so it becomes the most recent, then squeeze down to three plaintexts.
    cc.EvalMult(ct, pts[0])
    cc.SetPlaintextCache(3 * one)
    live = [k for k, pt in enumerate(pts) if pt.IsLoadedOnDevice()]
    scn.log(f"budget = 3 plaintexts, resident after touching 0: {live}")
    assert 0 in live, "the most recently used plaintext was evicted"
    assert live == [0, 2, 3], f"eviction did not follow use order (got {live})"

    # Squeeze to one: only the most recently used survives.
    cc.EvalMult(ct, pts[2])
    cc.SetPlaintextCache(one)
    live = [k for k, pt in enumerate(pts) if pt.IsLoadedOnDevice()]
    scn.log(f"budget = 1 plaintext, resident after touching 2: {live}")
    assert live == [2], f"expected only the most recently used to survive (got {live})"
    scn.ok("eviction follows recency of use")


def s_pin(scn):
    """Pinned plaintexts survive eviction pressure."""
    cc = make_context()
    keys = load(cc)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    pts = plaintexts(cc)
    mask = cc.MakeCKKSPackedPlaintext(vec(99))
    one = one_plaintext_bytes(cc, pts[0])

    cc.PinPlaintext(mask)  # pinning loads it
    assert mask.IsLoadedOnDevice(), "PinPlaintext did not load the plaintext"
    cc.SetPlaintextCache(2 * one)
    for k, pt in enumerate(pts):
        cc.EvalMult(ct, pt)
        assert mask.IsLoadedOnDevice(), f"pinned plaintext evicted after mult({k})"
        nbytes = cc.GetPlaintextCacheResidentBytes()
        assert nbytes <= 3 * one, f"budget blown with a pin: {nbytes:,}"
    scn.log(f"pinned plaintext survived {NPT} mults under a 2-plaintext budget")

    got = decrypt(cc, keys.secretKey, cc.EvalMult(ct, mask))
    exp = [a * b for a, b in zip(vec()[:64], vec(99)[:64])]
    assert maxdiff(got, exp) < 1e-3, "multiply by a pinned plaintext is wrong"

    cc.PinPlaintext(mask, pin=False)  # unpinning must let the budget bind again
    for pt in pts:
        cc.EvalMult(ct, pt)
    scn.log(f"after unpin + one full pass: mask resident={mask.IsLoadedOnDevice()}")
    assert not mask.IsLoadedOnDevice(), "unpinned plaintext was never evicted"
    got = decrypt(cc, keys.secretKey, cc.EvalMult(ct, mask))
    assert maxdiff(got, exp) < 1e-3, "multiply after unpin/evict is wrong"
    scn.ok("pin / unpin under eviction pressure")


def s_manual(scn):
    """OffloadPlaintexts(), per-plaintext UnloadFromDevice(), idempotence."""
    cc = make_context()
    keys = load(cc)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    pts = plaintexts(cc)
    for pt in pts:
        cc.EvalMult(ct, pt)
    assert resident_count(cc) == NPT

    pts[3].UnloadFromDevice()
    assert not pts[3].IsLoadedOnDevice()
    assert resident_count(cc) == NPT - 1, "single unload spilled to other plaintexts"
    pts[3].UnloadFromDevice()  # idempotent
    assert resident_count(cc) == NPT - 1

    cc.OffloadPlaintexts()
    assert cc.GetPlaintextCacheResidentBytes() == 0, "offload-all left bytes accounted"
    assert resident_count(cc) == 0, "offload-all left plaintexts resident"
    assert not any(pt.IsLoadedOnDevice() for pt in pts)
    cc.OffloadPlaintexts()  # idempotent

    # A pinned plaintext is skipped by OffloadPlaintexts().
    cc.PinPlaintext(pts[1])
    cc.OffloadPlaintexts()
    assert pts[1].IsLoadedOnDevice(), "OffloadPlaintexts() evicted a pinned plaintext"
    scn.log("OffloadPlaintexts() skipped the pinned plaintext")

    got = decrypt(cc, keys.secretKey, cc.EvalMult(ct, pts[0]))
    exp = [a * b for a, b in zip(vec()[:64], vec(0)[:64])]
    assert maxdiff(got, exp) < 1e-3, "multiply after manual offload is wrong"
    scn.ok("manual offload / per-plaintext unload / idempotence")


def s_budget_change(scn):
    """Runtime budget changes: tighten (evicts immediately), widen, None, zero."""
    cc = make_context()
    keys = load(cc)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    pts = plaintexts(cc)
    for pt in pts:
        cc.EvalMult(ct, pt)
    full = cc.GetPlaintextCacheResidentBytes()
    one = full // NPT
    scn.log(f"all {NPT} plaintexts resident: {full:,} bytes")

    cc.SetPlaintextCache(3 * one)
    b = cc.GetPlaintextCacheResidentBytes()
    scn.log(f"after tightening to {3 * one:,}: {b:,} bytes, {resident_count(cc)} resident")
    assert b <= 3 * one, f"runtime tighten did not evict ({b:,})"
    assert cc.GetPlaintextCache() == 3 * one

    cc.SetPlaintextCache(None)
    assert cc.GetPlaintextCache() is None
    for pt in pts:
        cc.EvalMult(ct, pt)
    assert cc.GetPlaintextCacheResidentBytes() == full, "unlimited budget did not refill"

    cc.SetPlaintextCache(0)
    assert cc.GetPlaintextCacheResidentBytes() == 0, "zero budget kept plaintexts resident"
    got = decrypt(cc, keys.secretKey, cc.EvalMult(ct, pts[5]))
    exp = [a * b for a, b in zip(vec()[:64], vec(5)[:64])]
    assert maxdiff(got, exp) < 1e-3, "multiply under a zero budget is wrong"
    b = cc.GetPlaintextCacheResidentBytes()
    scn.log(f"zero budget, after a multiply: {b:,} bytes ({resident_count(cc)} resident)")
    assert b <= one, "a zero budget must keep at most the plaintext in use"

    cc.SetPlaintextCache(None)
    for k, pt in enumerate(pts):
        got = decrypt(cc, keys.secretKey, cc.EvalMult(ct, pt))
        exp = [a * b for a, b in zip(vec()[:64], vec(k)[:64])]
        assert maxdiff(got, exp) < 1e-3, f"multiply {k} after budget churn is wrong"
    scn.ok("runtime budget changes incl. None and 0")


def s_lifetime(scn):
    """Destroying a Plaintext, and Decrypt reusing one, keep the accounting straight."""
    cc = make_context()
    keys = load(cc)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    pts = plaintexts(cc)
    for k in range(NPT):
        cc.EvalMult(ct, pts[k])  # by index: a loop variable would outlive the list below
    full = cc.GetPlaintextCacheResidentBytes()
    one = full // NPT

    dropped = pts.pop()
    del dropped
    b = cc.GetPlaintextCacheResidentBytes()
    scn.log(f"dropped one plaintext: {full:,} -> {b:,} bytes, {resident_count(cc)} resident")
    assert resident_count(cc) == NPT - 1, "destroying a Plaintext left it registered"
    assert abs((full - b) - one) <= one // 8, "destroying a Plaintext left bytes accounted"

    # Decrypt into a fresh plaintext, then keep using the cache: the Decrypt path evicts
    # the plaintext it writes into, which must not corrupt the accounting.
    for _ in range(3):
        got = decrypt(cc, keys.secretKey, cc.EvalMult(ct, pts[0]))
    exp = [a * b for a, b in zip(vec()[:64], vec(0)[:64])]  # pts[0] is still vec(0)
    assert maxdiff(got, exp) < 1e-3
    b2 = cc.GetPlaintextCacheResidentBytes()
    scn.log(f"after 3 decrypts: {b2:,} bytes, {resident_count(cc)} resident")
    assert b2 == b, "decrypting moved the plaintext accounting"

    # All plaintexts gone -> nothing accounted, and the budget survives.
    cc.SetPlaintextCache(4 * one)
    pts.clear()
    assert cc.GetPlaintextCacheResidentBytes() == 0, f"leftover accounting: {cc.GetPlaintextCacheResidentBytes()}"
    assert resident_count(cc) == 0
    assert cc.GetPlaintextCache() == 4 * one
    scn.ok("plaintext lifetime vs cache bookkeeping")


def s_unloaded(scn):
    """Cache calls before LoadContext: no crash, sane answers."""
    cc = make_context()
    assert cc.GetPlaintextCache() is None
    assert cc.GetPlaintextCacheResidentBytes() == 0
    cc.SetPlaintextCache(1 << 30)
    assert cc.GetPlaintextCache() == 1 << 30
    cc.OffloadPlaintexts()  # nothing to do, must not throw
    assert cc.GetPlaintextCacheResidentBytes() == 0

    keys = cc.KeyGen()
    pt = cc.MakeCKKSPackedPlaintext(vec())
    assert not pt.IsLoadedOnDevice()
    try:
        cc.PinPlaintext(pt)  # needs a loaded context to upload anything
    except Exception as e:  # noqa: BLE001
        scn.log(f"PinPlaintext before LoadContext -> {type(e).__name__}: {str(e)[:70]}")
    else:
        scn.log("PinPlaintext before LoadContext -> no error (accepted)")

    cc.LoadContext(keys.publicKey)
    cc.LoadPlaintext(pt)
    assert pt.IsLoadedOnDevice() and cc.GetPlaintextCacheResidentBytes() > 0
    scn.ok("cache API on an unloaded context")


def s_per_context(scn):
    """Two contexts with identical params keep INDEPENDENT plaintext caches.

    Unlike the rotation-key cache -- which lives in the GPU ContextData that FIDESlib
    shares between every CryptoContextImpl built from equal Parameters -- this cache lives
    in the api-level context, so budgets and accounting never cross-talk.
    """
    a = make_context()
    keys_a = load(a)
    ct_a = a.Encrypt(keys_a.publicKey, a.MakeCKKSPackedPlaintext(vec()))
    pts_a = plaintexts(a, 4)
    for pt in pts_a:
        a.EvalMult(ct_a, pt)
    one = a.GetPlaintextCacheResidentBytes() // 4

    b = make_context()  # identical params -> the SAME underlying GPU context
    keys_b = load(b)
    ct_b = b.Encrypt(keys_b.publicKey, b.MakeCKKSPackedPlaintext(vec()))
    pts_b = plaintexts(b, 4)
    b.SetPlaintextCache(one)
    for pt in pts_b:
        b.EvalMult(ct_b, pt)
    scn.log(f"A (no budget): {a.GetPlaintextCacheResidentBytes():,} bytes / "
            f"{resident_count(a)} resident; B (1-pt budget): "
            f"{b.GetPlaintextCacheResidentBytes():,} bytes / {resident_count(b)} resident")
    assert a.GetPlaintextCache() is None and b.GetPlaintextCache() == one
    assert resident_count(a) == 4, "B's budget evicted A's plaintexts"
    assert resident_count(b) <= 2, "B's budget did not bind"

    b.OffloadPlaintexts()
    assert resident_count(a) == 4, "offloading B touched A"
    got_a = decrypt(a, keys_a.secretKey, a.EvalMult(ct_a, pts_a[1]))
    got_b = decrypt(b, keys_b.secretKey, b.EvalMult(ct_b, pts_b[1]))
    exp = [x * y for x, y in zip(vec()[:64], vec(1)[:64])]
    assert maxdiff(got_a, exp) < 1e-3 and maxdiff(got_b, exp) < 1e-3, "shared-GPU-context mult wrong"
    scn.ok("independent per-context budgets (unlike the rotation-key cache)")


def s_vram_reclaim(scn):
    """A bound cache really does cap VRAM: 128 plaintexts, budget of four."""
    cc = make_context()
    keys = load(cc)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    pts = plaintexts(cc, 4)
    one = one_plaintext_bytes(cc, pts[0])
    cc.OffloadPlaintexts()

    count = 128
    base = pool_live_bytes(cc)
    cc.SetPlaintextCache(4 * one)
    kept = [cc.MakeCKKSPackedPlaintext(vec(k)) for k in range(count)]
    for pt in kept:
        cc.EvalMult(ct, pt)
    bounded_live = pool_live_bytes(cc) - base
    bounded_acct = cc.GetPlaintextCacheResidentBytes()
    scn.log(f"{count} plaintexts under a {4 * one:,}-byte budget: {bounded_acct:,} accounted, "
            f"{resident_count(cc)} resident, {bounded_live:,} bytes of live VRAM")
    assert bounded_acct <= 5 * one, f"budget blown: {bounded_acct:,}"
    assert bounded_live < 12 * one, \
        f"VRAM grew past the budget: {bounded_live:,} bytes for {count} plaintexts"

    # Same run with no budget must be visibly more expensive -- proof the cap is what did it.
    cc.SetPlaintextCache(None)
    for pt in kept:
        cc.EvalMult(ct, pt)
    unbounded_live = pool_live_bytes(cc) - base
    scn.log(f"same {count} plaintexts unbounded: {cc.GetPlaintextCacheResidentBytes():,} "
            f"accounted, {unbounded_live:,} bytes of live VRAM")
    assert cc.GetPlaintextCacheResidentBytes() > 100 * one, "unbounded run did not keep them"
    assert unbounded_live > 10 * bounded_live, \
        "bounded and unbounded runs used comparable VRAM -- the cap did nothing"
    scn.ok("the budget caps real VRAM use, not just the counter")


def s_no_leak(scn):
    """Forced misses must not leak: host RSS and VRAM stay flat over long churn."""
    import resource

    cc = make_context()
    keys = load(cc)
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec()))
    pts = plaintexts(cc)
    one = one_plaintext_bytes(cc, pts[0])
    cc.SetPlaintextCache(one)  # every multiply but the repeat misses

    for _ in range(3):
        for pt in pts:
            cc.EvalMult(ct, pt)
    cc.Synchronize()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    live0 = pool_live_bytes(cc)
    reps = 30
    for _ in range(reps):
        for pt in pts:
            cc.EvalMult(ct, pt)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    live1 = pool_live_bytes(cc)
    scn.log(f"host RSS {rss0} MiB -> {rss1} MiB, live VRAM {live0:,} -> {live1:,} "
            f"over {reps * NPT} forced misses")
    assert rss1 - rss0 < 128, f"host memory grew {rss1 - rss0} MiB -- re-upload is leaking"
    assert live1 - live0 < 4 * one, f"live VRAM grew {(live1 - live0):,} bytes -- eviction is leaking"
    scn.ok(f"no leak across {reps * NPT} forced misses")


def s_ops(scn):
    """Every plaintext-taking operation goes through the cache and stays correct."""
    cc = make_context()
    keys = load(cc)
    a = vec()
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(a))
    pts = plaintexts(cc, 4)
    one = one_plaintext_bytes(cc, pts[0])
    cc.SetPlaintextCache(one)  # one plaintext fits: every op below misses

    b = vec(1)
    checks = [
        ("EvalAdd", lambda: cc.EvalAdd(ct, pts[1]), [x + y for x, y in zip(a[:64], b[:64])]),
        ("EvalSub", lambda: cc.EvalSub(ct, pts[1]), [x - y for x, y in zip(a[:64], b[:64])]),
        ("EvalMult", lambda: cc.EvalMult(ct, pts[1]), [x * y for x, y in zip(a[:64], b[:64])]),
    ]
    for name, op, exp in checks:
        got = decrypt(cc, keys.secretKey, op())
        err = maxdiff(got, exp)
        nbytes = cc.GetPlaintextCacheResidentBytes()
        scn.log(f"{name}: error {err:.2e}, {nbytes:,} bytes resident")
        assert err < 1e-3, f"{name} under a one-plaintext budget is wrong ({err:.2e})"
        assert nbytes <= 2 * one, f"{name} blew the budget ({nbytes:,})"

    # In-place variants mutate the ciphertext, so run them on clones.
    for name, op, exp in (
        ("EvalAddInPlace", cc.EvalAddInPlace, [x + y for x, y in zip(a[:64], b[:64])]),
        ("EvalMultInPlace", cc.EvalMultInPlace, [x * y for x, y in zip(a[:64], b[:64])]),
    ):
        target = ct.Clone()
        op(target, pts[1])
        got = decrypt(cc, keys.secretKey, target)
        err = maxdiff(got, exp)
        scn.log(f"{name}: error {err:.2e}, {cc.GetPlaintextCacheResidentBytes():,} bytes resident")
        assert err < 1e-3, f"{name} under a one-plaintext budget is wrong ({err:.2e})"
    scn.ok("add / sub / mult (incl. in-place) under a one-plaintext budget")


SCENARIOS = {
    "defaults": s_defaults,
    "accounting": s_accounting,
    "lru": s_lru,
    "lru_order": s_lru_order,
    "pin": s_pin,
    "manual": s_manual,
    "budget_change": s_budget_change,
    "lifetime": s_lifetime,
    "unloaded": s_unloaded,
    "per_context": s_per_context,
    "vram_reclaim": s_vram_reclaim,
    "no_leak": s_no_leak,
    "ops": s_ops,
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
            r = subprocess.run([sys.executable, __file__, "--scenario", name])
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
