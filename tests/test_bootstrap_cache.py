#!/usr/bin/env python3
"""Exercise the bootstrap-precomputation VRAM cache (SetBootstrapCache and friends).

Every scenario runs in its own subprocess (`--all`), so a crash pinpoints exactly which
behaviour broke instead of killing the whole run. A crash is reported as `exit=-11`.

    python tests/test_bootstrap_cache.py --all
    python tests/test_bootstrap_cache.py --scenario policy

Env: BCTEST_DEVICES (physical GPU indexes, comma-separated, exported as
CUDA_VISIBLE_DEVICES; default "0"). With two or more the context is multi-GPU, which the
cache accounts across all of them.

Uses the parameters of tests/test_bootstrap_memory.py (small and non-secure, sparse
encapsulated secret, full packing at logN=13): with level budget [4, 4] the precomputation
is 8 transform stages, the cache's unit, and `lt` covers the [1, 1] budget, which keeps its
matrices in a single non-collapsed linear transform instead.
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

# Confine the process to the selected GPUs before the module is imported: importing it
# initializes CUDA on every visible device. Already-set CUDA_VISIBLE_DEVICES is left alone,
# so the `--all` subprocesses inherit it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("BCTEST_DEVICES", "0"))

import fideslib_py as fhe  # noqa: E402

DEVICES = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
SLOTS = 4096
DEPTH = 30
BOOT_TOL = 1e-5  # measured ~1e-6 for these parameters (tests/test_bootstrap_memory.py)
# EvalBootstrap(ct, ITERATIONS, 8) is the iterative bootstrap the training code runs: it walks
# the transform stages ITERATIONS times, so per-bootstrap reload volumes scale with it.
ITERATIONS = 2


class Skipped(Exception):
    pass


# ---------------------------------------------------------------- helpers
class Ctx:
    """A loaded bootstrappable context, its keys and a bottom-level ciphertext."""

    def __init__(self, budget, *, level_budget=(4, 4), slots=SLOTS, rotation_budget=None,
                 set_after_load=False):
        self.slots = slots
        p = fhe.CCParams()
        p.SetSecurityLevel(fhe.HEStd_NotSet)
        p.SetRingDim(8192)
        p.SetMultiplicativeDepth(DEPTH)
        p.SetScalingModSize(59)
        p.SetFirstModSize(60)
        p.SetNumLargeDigits(3)
        p.SetBatchSize(slots)
        p.SetScalingTechnique(fhe.FLEXIBLEAUTO)
        p.SetKeySwitchTechnique(fhe.HYBRID)
        p.SetSecretKeyDist(fhe.SPARSE_ENCAPSULATED)
        p.SetDevices(DEVICES)
        self.cc = cc = fhe.GenCryptoContext(p)
        for feature in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
            cc.Enable(feature)
        self.keys = cc.KeyGen()
        cc.EvalMultKeyGen(self.keys.secretKey)
        cc.EvalBootstrapSetup(list(level_budget), [0, 0], slots)
        cc.EvalBootstrapKeyGen(self.keys.secretKey, slots)
        if rotation_budget is not None:
            cc.SetRotationKeyCache(rotation_budget)
        if not set_after_load:
            cc.SetBootstrapCache(budget)
        cc.LoadContext(self.keys.publicKey)
        if set_after_load:
            cc.SetBootstrapCache(budget)
        self.values = [0.01 + 0.0001 * (i % 17) for i in range(slots)]
        self.ct = cc.Encrypt(self.keys.publicKey,
                             cc.MakeCKKSPackedPlaintext(self.values, 1, DEPTH - 1, slots))

    def bootstrap(self):
        """Bootstrap the ciphertext, check the result, return (decrypted slots, bytes reloaded)."""
        before = self.cc.GetBootstrapCacheLoadedBytes()
        out = self.cc.EvalBootstrap(self.ct, ITERATIONS, 8)
        pt = self.cc.Decrypt(self.keys.secretKey, out)
        pt.SetLength(self.slots)
        got = list(pt.GetRealPackedValue())
        err = maxdiff(got, self.values)
        assert err < BOOT_TOL, f"bootstrap is wrong ({err:.2e})"
        return got, self.cc.GetBootstrapCacheLoadedBytes() - before

    def resident(self):
        return self.cc.GetBootstrapCacheResidentBytes()


def maxdiff(a, b):
    assert len(a) == len(b), f"length mismatch {len(a)} vs {len(b)}"
    return max(abs(x - y) for x, y in zip(a, b))


def live_pool_bytes():
    """The allocator's own view of the live limbs, summed over the context's devices."""
    total = 0
    for device in DEVICES:
        stats = fhe.GetGPUMemoryPoolStats(device)
        total += sum(b["live_chunks"] * b["chunk_bytes"] for b in stats["buckets"])
    return total


def settle(cc):
    cc.Synchronize()
    cc.ClearAuxiliaryPolyPool()


MiB = 1 << 20


# ---------------------------------------------------------------- scenarios
def s_defaults(scn):
    """No budget: legacy behaviour, the whole precomputation resident for good."""
    c = Ctx(None)
    assert c.cc.GetBootstrapCache() is None, "default budget should be unlimited/None"
    total = c.resident()
    scn.log(f"precomputation: {total / MiB:.1f} MiB")
    assert total > 0, "with no budget the precomputation must be resident right away"
    c.bootstrap()
    _, loaded = c.bootstrap()
    assert loaded == 0 and c.resident() == total, "no-budget bootstrap moved the precomputation"
    c.cc.OffloadBootstrapPrecomputation()
    assert c.resident() == total, "matrices built without a budget have no snapshot to offload to"
    scn.ok("legacy (unbounded) behaviour intact")


def s_lazy(scn):
    """Built under a budget, the precomputation costs no VRAM until the first bootstrap."""
    c = Ctx(1 << 40)  # never binds: only laziness is exercised
    assert c.cc.GetBootstrapCache() == 1 << 40
    assert c.resident() == 0, f"lazy matrices hold {c.resident():,} bytes before first use"
    _, loaded = c.bootstrap()
    total = c.resident()
    scn.log(f"first bootstrap loaded {loaded / MiB:.1f} MiB, resident {total / MiB:.1f} MiB")
    assert total > 0 and loaded == total, "the first bootstrap must load every stage, once"
    _, loaded = c.bootstrap()
    assert loaded == 0 and c.resident() == total, "a budget that fits must not reload anything"
    scn.ok("lazy creation + on-demand load")


def s_accounting(scn):
    """The resident-byte counter matches what the allocator actually frees."""
    c = Ctx(1 << 40)
    c.bootstrap()
    settle(c.cc)
    resident, live = c.resident(), live_pool_bytes()
    c.cc.OffloadBootstrapPrecomputation()
    settle(c.cc)
    freed = live - live_pool_bytes()
    scn.log(f"counter said {resident:,} bytes, the allocator freed {freed:,}")
    assert c.resident() == 0, "offload-all left bytes accounted"
    assert freed == resident, f"counter ({resident:,}) != freed VRAM ({freed:,})"
    scn.ok("byte accounting exact against the allocator")


def s_exact(scn):
    """Reloading the matrices does not change the result beyond the bootstrap's own spread."""
    c = Ctx(1 << 40)
    ref, _ = c.bootstrap()
    floor = max(maxdiff(ref, c.bootstrap()[0]) for _ in range(2))
    scn.log(f"warm-to-warm reproducibility floor: {floor:.2e}")
    for cycle in range(3):
        c.cc.OffloadBootstrapPrecomputation()
        got, loaded = c.bootstrap()
        d = maxdiff(ref, got)
        scn.log(f"cycle {cycle}: reloaded {loaded / MiB:.1f} MiB, |cold - warm| = {d:.2e}")
        assert loaded > 0, "an offloaded precomputation must be reloaded"
        assert d <= max(4 * floor, 1e-9), f"reload changed the result: {d:.2e} >> {floor:.2e}"
    scn.ok("3x offload/reload round trip within the noise floor")


def measured(**kwargs):
    """A context built under a budget that never binds, and its total precomputation size.

    The scenarios below then re-tune the budget at runtime on this same context rather than
    building another one: FIDESlib caches GPU contexts by parameters, so a second Ctx in the
    same process would share this one's precomputation, budget and counters anyway."""
    c = Ctx(1 << 40, **kwargs)
    c.bootstrap()
    return c, c.resident()


def s_bounded(scn):
    """Under a budget the resident bytes stay within it, plus at most the stage in use."""
    c, total = measured()
    for frac in (0.0, 0.25, 0.5, 0.75):
        budget = int(total * frac)
        c.cc.SetBootstrapCache(budget)
        for i in range(3):
            c.bootstrap()
            r = c.resident()
            # Soft by one stage: with 8 stages, a quarter of the total is a generous bound.
            assert r <= budget + total // 4, f"budget {budget:,}: {r:,} resident after bootstrap {i}"
        scn.log(f"budget {frac:.2f} x total: {c.resident() / MiB:.1f} of {total / MiB:.1f} MiB resident")
    scn.ok("the budget binds, soft by one stage")


def s_policy(scn):
    """A bootstrap walks its stages cyclically, so the cache evicts the most recently used one:
    only what does not fit is reloaded on each pass over the stages. (LRU would reload all of
    them on every pass.)"""
    c, total = measured()
    for frac in (0.5, 0.75, 0.9):
        c.cc.SetBootstrapCache(int(total * frac))
        c.bootstrap()
        steady = [c.bootstrap()[1] / ITERATIONS for _ in range(3)]
        scn.log(f"budget {frac:.2f} x total: reloads {[f'{s / MiB:.1f}' for s in steady]} MiB "
                f"per pass over the stages, of {total / MiB:.1f}")
        # The ideal is total - budget per pass; one stage of slack on top, plus rounding.
        worst = max(steady)
        assert worst < total, f"budget {frac:.2f}: every stage reloaded ({worst:,.0f} of {total:,})"
        assert worst <= total * (1 - frac) + total // 4, f"budget {frac:.2f}: reloaded {worst:,.0f}"
    scn.ok("the cache streams only what exceeds the budget")


def s_budget_change(scn):
    """Re-tuning the budget at runtime evicts immediately, and None lets everything back in."""
    c = Ctx(1 << 40)
    c.bootstrap()
    total = c.resident()
    c.cc.SetBootstrapCache(total // 3)
    assert c.resident() <= total // 3, f"tightening did not evict ({c.resident():,} > {total // 3:,})"
    c.bootstrap()
    c.cc.SetBootstrapCache(None)
    assert c.cc.GetBootstrapCache() is None
    c.bootstrap()  # reloads what was evicted, then keeps it
    assert c.resident() == total, "unlimited again but not everything came back"
    _, loaded = c.bootstrap()
    assert loaded == 0, "unlimited budget still reloading"
    scn.ok("runtime budget changes")


def s_after_load(scn):
    """A budget set only after LoadContext finds no snapshot and evicts nothing."""
    c = Ctx(0, set_after_load=True)
    total = c.resident()
    assert total > 0, "the matrices were built without a budget, so they must be resident"
    c.bootstrap()
    assert c.resident() == total, "evicted matrices that have no snapshot to reload from"
    scn.ok("matrices built without a budget stay resident (documented)")


def s_lt(scn):
    """Level budget [1, 1]: the matrices live in one non-collapsed transform, LT.A / LT.invA.
    One plaintext per diagonal, so it is kept sparse: at 4096 slots it would be 18 GiB."""
    c = Ctx(1 << 40, level_budget=(1, 1), slots=256)
    ref, loaded = c.bootstrap()
    total = c.resident()
    assert total > 0 and loaded == total, "the LT matrices were not cached"
    c.cc.SetBootstrapCache(0)
    got, loaded = c.bootstrap()
    scn.log(f"LT precomputation {total / MiB:.1f} MiB; budget 0 reloaded {loaded / MiB:.1f} MiB "
            f"and kept {c.resident() / MiB:.1f} MiB")
    assert loaded > 0 and c.resident() < total, "budget 0 did not evict the LT matrices"
    scn.ok("non-collapsed linear transform cached too")


def s_with_rotation_cache(scn):
    """Both budgets binding at once: rotation keys and precomputation streamed together."""
    c, total = measured(rotation_budget=1)
    c.cc.SetBootstrapCache(total // 2)
    for _ in range(3):
        c.bootstrap()
    keys = c.cc.GetRotationKeyCacheResidentBytes()
    scn.log(f"resident: {keys / MiB:.1f} MiB of keys, {c.resident() / MiB:.1f} of "
            f"{total / MiB:.1f} MiB of precomputation")
    assert c.resident() <= total // 2 + total // 4, "precomputation budget not binding"
    scn.ok("composes with the rotation-key cache")


def s_unloaded(scn):
    """Before LoadContext: the budget is stored, offloading throws, nothing is resident."""
    p = fhe.CCParams()
    p.SetRingDim(8192)
    p.SetMultiplicativeDepth(4)
    p.SetSecurityLevel(fhe.HEStd_NotSet)
    p.SetDevices(DEVICES)
    cc = fhe.GenCryptoContext(p)
    cc.SetBootstrapCache(123)
    assert cc.GetBootstrapCache() == 123
    assert cc.GetBootstrapCacheResidentBytes() == 0 and cc.GetBootstrapCacheLoadedBytes() == 0
    try:
        cc.OffloadBootstrapPrecomputation()
    except Exception as e:  # noqa: BLE001 -- OpenFHE raises a generic exception type
        scn.log(f"OffloadBootstrapPrecomputation before LoadContext: {type(e).__name__}")
    else:
        raise AssertionError("OffloadBootstrapPrecomputation must throw before LoadContext")
    scn.ok("pre-load behaviour")


def s_no_leak(scn):
    """Every bootstrap under budget 0 reloads every stage: neither VRAM nor host RAM may grow."""
    import resource

    c = Ctx(0)
    for _ in range(3):
        c.bootstrap()
    settle(c.cc)
    live0, rss0 = live_pool_bytes(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    loaded = 0
    for _ in range(20):
        loaded += c.bootstrap()[1]
    settle(c.cc)
    live1, rss1 = live_pool_bytes(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    scn.log(f"20 bootstraps reloaded {loaded / MiB:.0f} MiB; live VRAM {live0:,} -> {live1:,}, "
            f"host RSS {rss0} -> {rss1} MiB")
    assert live1 == live0, f"live VRAM grew by {live1 - live0:,} bytes"
    assert rss1 - rss0 < 128, f"host memory grew {rss1 - rss0} MiB -- reload is leaking"
    scn.ok("no leak across 20 fully streamed bootstraps")


SCENARIOS = {
    "defaults": s_defaults,
    "lazy": s_lazy,
    "accounting": s_accounting,
    "exact": s_exact,
    "bounded": s_bounded,
    "policy": s_policy,
    "budget_change": s_budget_change,
    "after_load": s_after_load,
    "lt": s_lt,
    "with_rotation_cache": s_with_rotation_cache,
    "unloaded": s_unloaded,
    "no_leak": s_no_leak,
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
            r = subprocess.run([sys.executable, __file__, "--scenario", name],
                               stdout=subprocess.PIPE, text=True)
            print(r.stdout, end="", flush=True)
            print(f"    ({time.time() - t0:.0f}s)", flush=True)
            # Native CUDA teardown can turn a Python failure into exit status 0 (see
            # tests/test_bootstrap_memory.py), so a pass also needs the scenario's own marker.
            passed = any(line.startswith((f"PASS [{name}]", f"SKIP [{name}]"))
                         for line in r.stdout.splitlines())
            if r.returncode != 0 or not passed:
                failed.append((name, r.returncode))
        print("\n===== summary =====")
        for name, code in failed:
            why = f"CRASHED (signal {-code})" if code < 0 else f"failed (exit {code})"
            if code == 0:
                why = "failed (no PASS marker, exit 0)"
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
