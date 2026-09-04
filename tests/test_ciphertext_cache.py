#!/usr/bin/env python3
"""Exercise the ciphertext VRAM cache (SetCiphertextCache and friends).

Every scenario runs in its own subprocess (`--all`), so a crash pinpoints exactly which
behaviour broke instead of killing the whole run. A crash is reported as `exit=-11`.

    python tests/test_ciphertext_cache.py --all
    python tests/test_ciphertext_cache.py --scenario lru

Env: CTTEST_DEVICE (physical GPU index, exported as CUDA_VISIBLE_DEVICES so the process
only ever touches that one GPU), CTTEST_RING (log2 ring dim, default 14).
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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CTTEST_DEVICE", "0"))

import fideslib_py as fhe  # noqa: E402

DEVICE = 0  # index within CUDA_VISIBLE_DEVICES, i.e. the one GPU selected above
RING = int(os.environ.get("CTTEST_RING", "14"))
BATCH = 1 << (RING - 1)
NCT = 8  # ciphertexts used by most scenarios


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


def load(cc, budget=None, rot_idxs=(1, 2, 3)):
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, list(rot_idxs))
    cc.LoadContext(keys.publicKey)
    if budget is not None:
        cc.SetCiphertextCache(budget)
    return keys


def vec(k=0, n=BATCH):
    return [float((i + 3 * k) % 37) - 18.0 for i in range(n)]


def encrypt(cc, keys, k=0):
    return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vec(k)))


def ciphertexts(cc, keys, count=NCT):
    return [encrypt(cc, keys, k) for k in range(count)]


def decrypt(cc, sk, ct, n=64):
    pt = cc.Decrypt(sk, ct)
    pt.SetLength(n)
    return list(pt.GetRealPackedValue())


def maxdiff(a, b):
    assert len(a) == len(b), f"length mismatch {len(a)} vs {len(b)}"
    return max(abs(x - y) for x, y in zip(a, b))


def resident_count(cc, cts):
    return sum(0 if ct.IsOffloaded() else 1 for ct in cts)


def pool_live_bytes(cc=None):
    """VRAM held by live objects, as FIDESlib's own allocator sees it.

    Neither `cudaMemGetInfo` nor `nvidia-smi` can measure this: an offload returns the limbs
    to FIDESlib's pool, not to the driver, and TrimGPUMemoryPool() only hands back whole
    idle slabs. The pool's own live-chunk count does move, and that is what a byte budget is
    meant to cap.

    Passing `cc` first drains FIDESlib's auxiliary-polynomial cache, which otherwise sits
    between the two: destroying a ciphertext hands its polynomials to that cache instead of
    to the allocator, so they stay *live* chunks and a measurement taken without draining it
    shows nothing being released at all (measured: 0 bytes back after dropping 24
    ciphertexts). See FIDESLIB_AUX_POLY_CACHE_LIMIT in the README.
    """
    if cc is not None:
        cc.Synchronize()
        cc.ClearAuxiliaryPolyPool()
    stats = fhe.GetGPUMemoryPoolStats(DEVICE)
    return sum(b["live_chunks"] * b["chunk_bytes"] for b in stats["buckets"])


def one_ct_bytes(cc, keys):
    """Bytes of a single resident ciphertext, as the cache accounts for them.

    The probe ciphertext is dropped again before returning, so it does not linger in the
    accounting for the rest of the scenario.
    """
    before = cc.GetCiphertextCacheResidentBytes()
    ct = encrypt(cc, keys, 0)
    nbytes = cc.GetCiphertextCacheResidentBytes() - before
    del ct
    assert nbytes > 0, "encrypting accounted for nothing"
    assert cc.GetCiphertextCacheResidentBytes() == before, "the probe ciphertext leaked"
    return nbytes


def noise_floor(cc, sk, a, b, reps=2):
    """FIDESlib ops are not necessarily bit-reproducible run to run (multi-stream
    reductions), so 'the round trip changed nothing' can only mean 'within the op's own
    reproducibility floor'. Measured here on the same pair, with no cache pressure."""
    ref = decrypt(cc, sk, cc.EvalAdd(a, b))
    worst = 0.0
    for _ in range(reps):
        worst = max(worst, maxdiff(ref, decrypt(cc, sk, cc.EvalAdd(a, b))))
    return worst


# ---------------------------------------------------------------- scenarios
def s_defaults(scn):
    """No budget: legacy behaviour, every ciphertext stays on the device."""
    cc = make_context()
    assert cc.GetCiphertextCache() is None, "default budget should be unlimited/None"
    keys = load(cc)
    base = cc.GetCiphertextCacheResidentBytes()

    cts = ciphertexts(cc, keys)
    full = cc.GetCiphertextCacheResidentBytes() - base
    scn.log(f"{NCT} ciphertexts resident: {full:,} bytes ({full // NCT:,} each)")
    assert full > 0
    assert resident_count(cc, cts) == NCT, "a ciphertext was offloaded without a budget"

    got = decrypt(cc, keys.secretKey, cc.EvalAdd(cts[0], cts[1]))
    exp = [a + b for a, b in zip(vec(0)[:64], vec(1)[:64])]
    err = maxdiff(got, exp)
    scn.log(f"add error vs exact: {err:.2e}")
    assert err < 1e-3, f"unbounded add is wrong ({err:.2e})"
    scn.ok("legacy (unbounded) behaviour intact")


def s_accounting(scn):
    """Resident bytes match what the allocator hands out, and come back on destruction."""
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    # 2 components x (L+1) limbs x N x 8 bytes, TIMES TWO: unlike a plaintext's (constant)
    # limbs, a ciphertext limb also owns an equally sized auxiliary vector for the NTTs. So a
    # ciphertext costs 4x a plaintext of the same level, not 2x.
    expected = 4 * 7 * (1 << RING) * 8
    scn.log(f"one ciphertext = {one:,} bytes (4 x (L+1) x N x 8 = {expected:,}, i.e. payload "
            f"+ NTT scratch for both components)")
    assert one == expected, f"unexpected ciphertext size (expected {expected:,})"

    # Descending levels does NOT make a ciphertext cheaper: FIDESlib keeps the limbs (the
    # release in RNSPoly::dropToLevel is disabled upstream, `if (0 && ...)`). Asserted here
    # because it is the opposite of what one would assume when sizing a budget -- a deep
    # ciphertext costs exactly as much VRAM as a fresh one.
    a, b = encrypt(cc, keys, 0), encrypt(cc, keys, 1)
    before = cc.GetCiphertextCacheResidentBytes()
    deep = cc.Rescale(cc.EvalMult(a, b))
    after = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"level {a.GetLevel()} -> {deep.GetLevel()}: that ciphertext accounts for "
            f"{after - before:,} bytes (a fresh one is {one:,})")
    assert deep.GetLevel() > a.GetLevel(), "no rescale happened"
    assert after - before == one, \
        f"a rescaled ciphertext no longer costs a full one ({after - before:,} vs {one:,})"
    del a, b, deep

    # Nothing else may run between the two measurements below: any op in between allocates
    # (and FIDESlib's auxiliary-polynomial cache retains what it frees), which would show up
    # as live VRAM and swamp the comparison.
    count = 24
    live0 = pool_live_bytes(cc)
    acct0 = cc.GetCiphertextCacheResidentBytes()
    cts = ciphertexts(cc, keys, count)
    live1 = pool_live_bytes(cc)
    accounted = cc.GetCiphertextCacheResidentBytes() - acct0
    measured = live1 - live0
    scn.log(f"{count} ciphertexts: accounted {accounted:,} bytes, allocator {measured:,} "
            f"({accounted / count:,.0f} vs {measured / count:,.0f} each)")
    assert accounted > 0 and measured > 0
    ratio = accounted / measured
    assert 0.85 <= ratio <= 1.2, f"accounting is off by {ratio:.2f}x from the real allocation"

    del cts
    live2 = pool_live_bytes(cc)
    scn.log(f"after dropping them: {cc.GetCiphertextCacheResidentBytes() - acct0:,} accounted, "
            f"{(live1 - live2):,} bytes released by the allocator")
    assert cc.GetCiphertextCacheResidentBytes() == acct0, "dropping left bytes accounted"
    assert live1 - live2 > 0.85 * measured, "the VRAM was not released"
    scn.ok("byte accounting tracks the real allocation; levels do not shrink a ciphertext")


def s_lru(scn):
    """A budget of two ciphertexts forces eviction; results stay correct across misses."""
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    cts = ciphertexts(cc, keys)
    floor = noise_floor(cc, keys.secretKey, cts[0], cts[1])
    scn.log(f"one ciphertext = {one:,} bytes, reproducibility floor {floor:.2e}")

    ref = {}
    for k in range(NCT):
        ref[k] = decrypt(cc, keys.secretKey, cc.EvalAdd(cts[k], cts[(k + 1) % NCT]))

    # An op needs both operands plus its result: three ciphertexts, so budget for three and
    # expect the overshoot of one op's working set on top.
    cc.SetCiphertextCache(3 * one)
    for k in range(NCT):
        got = decrypt(cc, keys.secretKey, cc.EvalAdd(cts[k], cts[(k + 1) % NCT]))
        d = maxdiff(ref[k], got)
        nbytes = cc.GetCiphertextCacheResidentBytes()
        scn.log(f"add({k},{(k + 1) % NCT}): |bounded - unbounded| = {d:.2e}, "
                f"{nbytes:,} bytes resident, {resident_count(cc, cts)}/{NCT} of my cts on GPU")
        assert d <= max(4 * floor, 1e-9), f"add({k}) changed under a budget ({d:.2e})"
        assert nbytes <= 8 * one, f"budget blown: {nbytes:,} > {8 * one:,}"
    assert resident_count(cc, cts) < NCT, "the budget never evicted anything"
    scn.ok("LRU eviction under a three-ciphertext budget; misses reproduce")


def s_lru_order(scn):
    """Eviction takes the least recently USED ciphertext, not the oldest created."""
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    cts = ciphertexts(cc, keys, 4)
    assert resident_count(cc, cts) == 4

    # Touch 0 so it becomes the most recent, then squeeze. The op that does the touching
    # also creates a result, so give the budget room for that plus the two survivors.
    cc.EvalAdd(cts[0], cts[0])
    cc.SetCiphertextCache(2 * one)
    live = [k for k, ct in enumerate(cts) if not ct.IsOffloaded()]
    scn.log(f"budget = 2 ciphertexts, resident after touching 0: {live}")
    assert 0 in live, "the most recently used ciphertext was evicted"
    assert 1 not in live and 2 not in live, f"eviction did not follow use order ({live})"

    got = decrypt(cc, keys.secretKey, cts[1])  # an evicted ciphertext still decrypts
    assert maxdiff(got, vec(1)[:64]) < 1e-3, "an offloaded ciphertext decrypted wrong"
    scn.ok("eviction follows recency of use")


def s_pin(scn):
    """A pinned ciphertext survives eviction pressure."""
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    acc = encrypt(cc, keys, 99)
    cts = ciphertexts(cc, keys)

    cc.PinCiphertext(acc)
    assert not acc.IsOffloaded(), "PinCiphertext left the ciphertext offloaded"
    cc.SetCiphertextCache(3 * one)
    for k in range(NCT):
        cc.EvalAdd(cts[k], cts[(k + 1) % NCT])
        assert not acc.IsOffloaded(), f"pinned ciphertext evicted after add({k})"
    scn.log(f"pinned ciphertext survived {NCT} adds under a 3-ciphertext budget")

    got = decrypt(cc, keys.secretKey, cc.EvalAdd(acc, acc))
    exp = [2 * x for x in vec(99)[:64]]
    assert maxdiff(got, exp) < 1e-3, "add on a pinned ciphertext is wrong"

    cc.PinCiphertext(acc, pin=False)
    for _ in range(3):
        for k in range(NCT):
            cc.EvalAdd(cts[k], cts[(k + 1) % NCT])
    scn.log(f"after unpin + 3 passes: acc offloaded={acc.IsOffloaded()}")
    assert acc.IsOffloaded(), "unpinned ciphertext was never evicted"
    got = decrypt(cc, keys.secretKey, cc.EvalAdd(acc, acc))
    assert maxdiff(got, exp) < 1e-3, "add after unpin/evict is wrong"
    scn.ok("pin / unpin under eviction pressure")


def s_manual(scn):
    """Explicit Offload()/Reload() and OffloadCiphertexts() keep the accounting straight."""
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    cts = ciphertexts(cc, keys)
    full = cc.GetCiphertextCacheResidentBytes()

    cts[3].Offload()
    assert cts[3].IsOffloaded()
    b = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"one explicit Offload(): {full:,} -> {b:,} bytes")
    assert abs((full - b) - one) <= one // 8, "explicit offload did not update the accounting"
    cts[3].Offload()  # idempotent
    assert cc.GetCiphertextCacheResidentBytes() == b

    cts[3].Reload()
    assert not cts[3].IsOffloaded()
    assert cc.GetCiphertextCacheResidentBytes() == full, "reload did not restore the accounting"
    cts[3].Reload()  # idempotent
    assert cc.GetCiphertextCacheResidentBytes() == full

    cc.OffloadCiphertexts()
    b = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"OffloadCiphertexts(): {b:,} bytes resident, "
            f"{resident_count(cc, cts)}/{NCT} of my cts on GPU")
    assert resident_count(cc, cts) == 0, "offload-all left ciphertexts resident"
    cc.OffloadCiphertexts()  # idempotent

    cc.PinCiphertext(cts[1])
    cc.OffloadCiphertexts()
    assert not cts[1].IsOffloaded(), "OffloadCiphertexts() evicted a pinned ciphertext"

    got = decrypt(cc, keys.secretKey, cc.EvalAdd(cts[0], cts[2]))
    exp = [a + b for a, b in zip(vec(0)[:64], vec(2)[:64])]
    assert maxdiff(got, exp) < 1e-3, "add after a manual offload is wrong"
    scn.ok("manual offload / reload / offload-all / idempotence")


def s_budget_change(scn):
    """Runtime budget changes: tighten (evicts immediately), widen, None, zero."""
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    cts = ciphertexts(cc, keys)
    full = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"all {NCT} ciphertexts resident: {full:,} bytes")

    cc.SetCiphertextCache(3 * one)
    b = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"after tightening to {3 * one:,}: {b:,} bytes resident")
    assert b <= 3 * one, f"runtime tighten did not evict ({b:,})"
    assert cc.GetCiphertextCache() == 3 * one

    cc.SetCiphertextCache(None)
    assert cc.GetCiphertextCache() is None
    for ct in cts:
        ct.Reload()
    assert cc.GetCiphertextCacheResidentBytes() == full, \
        f"reloading everything did not restore {full:,} bytes"

    cc.SetCiphertextCache(0)
    assert cc.GetCiphertextCacheResidentBytes() == 0, "zero budget kept ciphertexts resident"
    got = decrypt(cc, keys.secretKey, cc.EvalAdd(cts[5], cts[6]))
    exp = [a + b for a, b in zip(vec(5)[:64], vec(6)[:64])]
    assert maxdiff(got, exp) < 1e-3, "add under a zero budget is wrong"
    b = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"zero budget, after an add: {b:,} bytes (one op's working set)")
    assert b <= 4 * one, f"zero budget kept more than an op's working set ({b:,})"

    cc.SetCiphertextCache(None)
    for k in range(NCT):
        got = decrypt(cc, keys.secretKey, cc.EvalAdd(cts[k], cts[(k + 1) % NCT]))
        exp = [a + b for a, b in zip(vec(k)[:64], vec((k + 1) % NCT)[:64])]
        assert maxdiff(got, exp) < 1e-3, f"add({k}) after budget churn is wrong"
    scn.ok("runtime budget changes incl. None and 0")


def s_overshoot(scn):
    """The documented caveat: eviction happens at op boundaries, so an op overshoots.

    Ciphertexts created *during* an op (its result, a hoisted batch) are accounted for but
    cannot be evicted while the op holds raw pointers to them; the next op's LoadCiphertext
    brings the budget back. Asserted here so a change in that contract shows up as a test
    failure rather than as a surprise.
    """
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    a, b = encrypt(cc, keys, 0), encrypt(cc, keys, 1)
    cc.SetCiphertextCache(1)  # one byte: nothing may stay resident between ops

    r = cc.EvalAdd(a, b)
    after_op = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"1-byte budget, right after one add: {after_op:,} bytes "
            f"({after_op / one:.1f} ciphertexts of working set)")
    assert after_op > 0, "an op cannot have run with nothing resident"
    assert after_op <= 4 * one, f"overshoot is larger than one op's working set ({after_op:,})"

    cc.EvalAdd(a, b)  # the next op's Load must shrink it back
    scn.log(f"after the next op started: {cc.GetCiphertextCacheResidentBytes():,} bytes")
    assert cc.GetCiphertextCacheResidentBytes() <= after_op, "the budget never re-shrank"

    got = decrypt(cc, keys.secretKey, r)
    exp = [x + y for x, y in zip(vec(0)[:64], vec(1)[:64])]
    assert maxdiff(got, exp) < 1e-3, "the result of the overshooting op is wrong"
    scn.ok("overshoot bounded by one op's working set, re-shrinks on the next op")


def s_lifetime(scn):
    """Destroying a ciphertext, offloaded or not, keeps the accounting straight."""
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    cts = ciphertexts(cc, keys)
    full = cc.GetCiphertextCacheResidentBytes()

    dropped = cts.pop()
    del dropped
    b = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"dropped a resident ciphertext: {full:,} -> {b:,} bytes")
    assert abs((full - b) - one) <= one // 8, "destroying a ciphertext left bytes accounted"

    # Now drop an OFFLOADED one: its bytes are already off the books, so the total must not move.
    cts[0].Offload()
    b2 = cc.GetCiphertextCacheResidentBytes()
    dropped = cts.pop(0)
    del dropped
    b3 = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"dropped an offloaded ciphertext: {b2:,} -> {b3:,} bytes (must not move)")
    assert b3 == b2, "destroying an offloaded ciphertext double-counted"

    cts.clear()
    assert cc.GetCiphertextCacheResidentBytes() == 0, \
        f"leftover accounting: {cc.GetCiphertextCacheResidentBytes():,}"
    scn.ok("ciphertext lifetime vs cache bookkeeping")


def s_unloaded(scn):
    """Cache calls before LoadContext: no crash, sane answers."""
    cc = make_context()
    assert cc.GetCiphertextCache() is None
    assert cc.GetCiphertextCacheResidentBytes() == 0
    cc.SetCiphertextCache(1 << 30)
    assert cc.GetCiphertextCache() == 1 << 30
    cc.OffloadCiphertexts()  # nothing to do, must not throw
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.LoadContext(keys.publicKey)
    ct = encrypt(cc, keys, 0)
    assert cc.GetCiphertextCacheResidentBytes() > 0
    cc.PinCiphertext(ct)
    cc.PinCiphertext(ct, pin=False)
    scn.ok("cache API on an unloaded context")


def s_per_context(scn):
    """Two contexts with identical params keep INDEPENDENT ciphertext caches."""
    a = make_context()
    keys_a = load(a)
    cts_a = ciphertexts(a, keys_a, 4)
    one = a.GetCiphertextCacheResidentBytes() // 4

    b = make_context()  # identical params -> the SAME underlying GPU context
    keys_b = load(b)
    cts_b = ciphertexts(b, keys_b, 4)
    b.SetCiphertextCache(one)
    for k in range(4):
        b.EvalAdd(cts_b[k], cts_b[(k + 1) % 4])
    scn.log(f"A (no budget): {a.GetCiphertextCacheResidentBytes():,} bytes, "
            f"{resident_count(a, cts_a)}/4 resident; B (1-ct budget): "
            f"{b.GetCiphertextCacheResidentBytes():,} bytes, {resident_count(b, cts_b)}/4 resident")
    assert a.GetCiphertextCache() is None and b.GetCiphertextCache() == one
    assert resident_count(a, cts_a) == 4, "B's budget evicted A's ciphertexts"
    assert resident_count(b, cts_b) < 4, "B's budget did not bind"

    b.OffloadCiphertexts()
    assert resident_count(a, cts_a) == 4, "offloading B touched A"
    got_a = decrypt(a, keys_a.secretKey, a.EvalAdd(cts_a[0], cts_a[1]))
    got_b = decrypt(b, keys_b.secretKey, b.EvalAdd(cts_b[0], cts_b[1]))
    exp = [x + y for x, y in zip(vec(0)[:64], vec(1)[:64])]
    assert maxdiff(got_a, exp) < 1e-3 and maxdiff(got_b, exp) < 1e-3, "shared-GPU-context add wrong"
    scn.ok("independent per-context budgets")


def s_vram_reclaim(scn):
    """A bound cache really does cap VRAM: 64 ciphertexts, budget of four."""
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    cc.OffloadCiphertexts()

    count = 64
    base = pool_live_bytes(cc)
    cc.SetCiphertextCache(4 * one)
    kept = []
    for k in range(count):
        ct = encrypt(cc, keys, k)
        cc.EvalAdd(ct, ct)  # an op per ciphertext, so the budget gets a boundary to act on
        kept.append(ct)
    bounded_live = pool_live_bytes(cc) - base
    bounded_acct = cc.GetCiphertextCacheResidentBytes()
    scn.log(f"{count} ciphertexts under a {4 * one:,}-byte budget: {bounded_acct:,} accounted, "
            f"{resident_count(cc, kept)} of mine resident, {bounded_live:,} bytes of live VRAM")
    assert bounded_acct <= 8 * one, f"budget blown: {bounded_acct:,}"
    assert bounded_live < 16 * one, \
        f"VRAM grew past the budget: {bounded_live:,} bytes for {count} ciphertexts"

    cc.SetCiphertextCache(None)
    for ct in kept:
        ct.Reload()
    unbounded_live = pool_live_bytes(cc) - base
    scn.log(f"same {count} ciphertexts reloaded, unbounded: "
            f"{cc.GetCiphertextCacheResidentBytes():,} accounted, {unbounded_live:,} bytes live")
    assert cc.GetCiphertextCacheResidentBytes() > 50 * one, "reload did not bring them back"
    assert unbounded_live > 4 * bounded_live, \
        "bounded and unbounded runs used comparable VRAM -- the cap did nothing"

    got = decrypt(cc, keys.secretKey, kept[7])
    assert maxdiff(got, vec(7)[:64]) < 1e-3, "a ciphertext did not survive the churn"
    scn.ok("the budget caps real VRAM use, not just the counter")


def s_no_leak(scn):
    """Steady-state churn: host RSS and live VRAM flat over 240 forced round trips.

    Offloaded ciphertexts hold host RAM on purpose (that is where the snapshot lives), so
    what must stay flat is the steady state, not the absolute footprint.
    """
    import resource

    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    cts = ciphertexts(cc, keys)
    reps = 30

    def churn():
        for _ in range(reps):
            for k in range(NCT):
                cc.EvalAdd(cts[k], cts[(k + 1) % NCT])

    t0 = time.time()
    churn()  # unbounded baseline, for the price of the round trips below
    unbounded = time.time() - t0

    cc.SetCiphertextCache(3 * one)  # an op needs 3 ciphertexts, so every op has to reload
    churn()  # warm up the steady state before measuring
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    live0 = pool_live_bytes(cc)
    t0 = time.time()
    churn()
    bounded = time.time() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    live1 = pool_live_bytes(cc)
    offloaded = NCT - resident_count(cc, cts)
    scn.log(f"host RSS {rss0} MiB -> {rss1} MiB, live VRAM {live0:,} -> {live1:,} "
            f"over {reps * NCT} ops, {offloaded}/{NCT} ciphertexts parked in host RAM")
    scn.log(f"cost of the round trips: {1000 * bounded / (reps * NCT):.2f} ms/op bounded vs "
            f"{1000 * unbounded / (reps * NCT):.2f} ms/op unbounded ({bounded / unbounded:.1f}x)")
    assert offloaded > 0, "the budget never evicted anything -- this scenario tests nothing"
    assert rss1 - rss0 < 128, f"host memory grew {rss1 - rss0} MiB -- the snapshots leak"
    assert live1 - live0 < 4 * one, f"live VRAM grew {(live1 - live0):,} bytes -- eviction leaks"
    scn.ok(f"no leak across {reps * NCT} ops with forced round trips")


def s_ops(scn):
    """Every ciphertext-taking operation stays correct under a tight budget."""
    cc = make_context()
    keys = load(cc)
    one = one_ct_bytes(cc, keys)
    a, b = encrypt(cc, keys, 0), encrypt(cc, keys, 1)
    va, vb = vec(0)[:64], vec(1)[:64]
    pt = cc.MakeCKKSPackedPlaintext(vec(2))

    ref = {
        "EvalAdd": decrypt(cc, keys.secretKey, cc.EvalAdd(a, b)),
        "EvalSub": decrypt(cc, keys.secretKey, cc.EvalSub(a, b)),
        "EvalMult": decrypt(cc, keys.secretKey, cc.EvalMult(a, b)),
        "EvalMultPt": decrypt(cc, keys.secretKey, cc.EvalMult(a, pt)),
        "EvalSquare": decrypt(cc, keys.secretKey, cc.EvalSquare(a)),
        "EvalRotate": decrypt(cc, keys.secretKey, cc.EvalRotate(a, 2)),
        "Rescale": decrypt(cc, keys.secretKey, cc.Rescale(cc.EvalMult(a, b))),
        "AccumulateSum": decrypt(cc, keys.secretKey, cc.AccumulateSum(a, 4, 1), n=8),
    }
    cc.SetCiphertextCache(one)  # one ciphertext: every op has to reload something
    got = {
        "EvalAdd": decrypt(cc, keys.secretKey, cc.EvalAdd(a, b)),
        "EvalSub": decrypt(cc, keys.secretKey, cc.EvalSub(a, b)),
        "EvalMult": decrypt(cc, keys.secretKey, cc.EvalMult(a, b)),
        "EvalMultPt": decrypt(cc, keys.secretKey, cc.EvalMult(a, pt)),
        "EvalSquare": decrypt(cc, keys.secretKey, cc.EvalSquare(a)),
        "EvalRotate": decrypt(cc, keys.secretKey, cc.EvalRotate(a, 2)),
        "Rescale": decrypt(cc, keys.secretKey, cc.Rescale(cc.EvalMult(a, b))),
        "AccumulateSum": decrypt(cc, keys.secretKey, cc.AccumulateSum(a, 4, 1), n=8),
    }
    for name in ref:
        d = maxdiff(ref[name], got[name])
        scn.log(f"{name}: |bounded - unbounded| = {d:.2e}, "
                f"{cc.GetCiphertextCacheResidentBytes():,} bytes resident")
        assert d <= 1e-6, f"{name} changed under a one-ciphertext budget ({d:.2e})"
    # And spot-check one against the exact answer, not just against itself.
    assert maxdiff(got["EvalAdd"], [x + y for x, y in zip(va, vb)]) < 1e-3, "add is wrong"
    assert maxdiff(got["EvalRotate"], [vec(0)[(j + 2) % BATCH] for j in range(64)]) < 1e-3, \
        "rotate is wrong"
    scn.ok("add/sub/mult/square/rotate/rescale/accumulate under a one-ciphertext budget")


def s_with_other_caches(scn):
    """All three caches bound at once (rotation keys + plaintexts + ciphertexts)."""
    cc = make_context()
    keys = cc.KeyGen()
    cc.SetRotationKeyCache(1 << 20)  # must be set before LoadContext
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, [1, 2, 3])
    cc.LoadContext(keys.publicKey)

    a = encrypt(cc, keys, 0)
    pt = cc.MakeCKKSPackedPlaintext(vec(1))
    ref = decrypt(cc, keys.secretKey, cc.EvalRotate(cc.EvalMult(a, pt), 2))

    one_ct = cc.GetCiphertextCacheResidentBytes()
    cc.SetPlaintextCache(1)
    cc.SetCiphertextCache(one_ct)
    got = decrypt(cc, keys.secretKey, cc.EvalRotate(cc.EvalMult(a, pt), 2))
    d = maxdiff(ref, got)
    scn.log(f"all three caches bound: keys {cc.GetRotationKeyCacheResidentBytes():,} B, "
            f"plaintexts {cc.GetPlaintextCacheResidentBytes():,} B, "
            f"ciphertexts {cc.GetCiphertextCacheResidentBytes():,} B, |diff| = {d:.2e}")
    assert d <= 1e-6, f"mult+rotate under all three budgets changed ({d:.2e})"
    scn.ok("rotation-key, plaintext and ciphertext budgets active together")


SCENARIOS = {
    "defaults": s_defaults,
    "accounting": s_accounting,
    "lru": s_lru,
    "lru_order": s_lru_order,
    "pin": s_pin,
    "manual": s_manual,
    "budget_change": s_budget_change,
    "overshoot": s_overshoot,
    "lifetime": s_lifetime,
    "unloaded": s_unloaded,
    "per_context": s_per_context,
    "vram_reclaim": s_vram_reclaim,
    "no_leak": s_no_leak,
    "ops": s_ops,
    "with_other_caches": s_with_other_caches,
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
