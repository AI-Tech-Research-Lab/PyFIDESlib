# fideslib_py — Python bindings for FIDESlib (CKKS on GPU)

Minimal pybind11 wrapper around [FIDESlib](https://github.com/CAPS-UMU/FIDESlib)
v2.1.2, the CKKS GPU library interoperable with OpenFHE. Exposes the subset of
the API needed for encrypted inference pipelines (HerMiniRocket/PolyMiniRocket):
context setup, key generation, encoding, encrypt/decrypt, leveled arithmetic,
rotations, `EvalChebyshevSeries`, `AccumulateSum` and bootstrapping.

The compiled module **statically embeds FIDESlib and a patched OpenFHE 1.5.1**.
CMake clones and compiles its own pinned copy of FIDESlib (and FIDESlib's own vendored
OpenFHE) entirely inside `build/` -- no git submodule, no system install, no path outside
this repo. The pin lives in `CMakeLists.txt` (`FIDESLIB_REPOSITORY`/`FIDESLIB_GIT_TAG`; edit
the defaults there, or pass them with `-D` to define cache entries that shadow the defaults).
Its only runtime dependencies are the CUDA runtime (RPATH'd to
`/usr/local/cuda/lib64`) and an NVIDIA GPU.

## Build

```bash
./build.sh                          # clones + builds FIDESlib (and its OpenFHE) + the wrapper
./build.sh /usr/bin/python3         # against a specific interpreter
```

First run compiles the vendored OpenFHE from scratch (~10-30 min); later runs are fast since
it's only rebuilt when missing. To pin a different FIDESlib commit or fork, pass
`-DFIDESLIB_REPOSITORY=... -DFIDESLIB_GIT_TAG=...` to the `cmake -B build` step in `build.sh`,
or edit the defaults in `CMakeLists.txt`.

Prereqs: CUDA toolkit ≥ 12.4, gcc ≥ 11, CMake ≥ 3.25, network access (FIDESlib/OpenFHE/pybind11
fetches), and the Python dev headers for the chosen interpreter.

The module is bound to the Python **minor version** it was built against
(e.g. `_core.cpython-312-x86_64-linux-gnu.so` ⇒ Python 3.12). Rebuild to switch.

## Use

```bash
export PYTHONPATH=/path/to/PyFIDESlib          # or sys.path.insert / pip install -e
python examples/00_onboarding.py
```

```python
import fideslib_py as fhe

params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_128_classic)
params.SetMultiplicativeDepth(32)
params.SetScalingModSize(50)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetDevices([0])                        # GPU id(s)

cc = fhe.GenCryptoContext(params)
for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
    cc.Enable(f)

keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalRotateKeyGen(keys.secretKey, [1, 2, 3, 4])
cc.LoadContext(keys.publicKey)                # pushes keys to GPU; AFTER keygen

ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([1.0, 2.0, 3.0]))
ct = cc.EvalChebyshevSeries(ct, coeffs, -1.0, 1.0)   # GPU
pt = cc.Decrypt(keys.secretKey, ct)
pt.SetLength(3)
print(pt.GetRealPackedValue())
```

## Offloading ciphertexts / reclaiming VRAM

When you hold more ciphertexts than fit in VRAM, or want to free GPU memory before other
work, evict GPU-resident ciphertexts to host RAM and bring them back on demand:

```python
ct.Offload()          # limbs -> host RAM, VRAM freed into FIDESlib's pool
ct.IsOffloaded()      # -> True
ct.Reload()           # limbs -> GPU (also happens automatically on first use)
cc.TrimGPUMemoryPool()# return the freed VRAM to the OS
```

`Offload()`/`Reload()` are a bit-exact round trip (no decrypt/rescale/NTT). `Offload()` on
its own only pools the memory for cheap reuse by later FIDESlib ops — it does **not** shrink
the process's VRAM footprint (`nvidia-smi` shows no drop). Call `cc.TrimGPUMemoryPool()` once,
after offloading, to actually hand that memory back to the system; skip it if you only intend
to reuse the memory for more FIDESlib work. See `examples/03_offload.py`.

FIDESlib also has a higher-level cache of destroyed ciphertext polynomials. It can help
regular create/destroy workloads, but its upstream-unbounded behavior may strand many GiB
that plaintext and other allocation paths cannot reuse. Set its maximum retained polynomial
count before starting Python; `0` disables only this upper cache while the lower limb
allocator continues to reuse GPU buffers:

```bash
export FIDESLIB_AUX_POLY_CACHE_LIMIT=0
```

If the variable is absent or invalid, the original unbounded behavior is retained.

### Bounding rotation-key VRAM

Rotation keys are usually the largest permanent VRAM tenant — measured with `dnum=3`: 3.8 MiB
per key at logN=13/L=6, 24 MiB at logN=15/L=11, 132 MiB at logN=17/L=15 — and a bootstrappable
context needs hundreds of them — 300 keys at logN=17 alone is ~39 GiB. Give them a byte budget
and only the most
recently used ones stay on the GPU; the rest are parked in host RAM and reloaded on demand. A
miss costs one host-to-device copy.

```python
cc = fhe.GenCryptoContext(params)
...
keys = cc.KeyGen()
cc.EvalRotateKeyGen(keys.secretKey, [1, 2, 3, 4])
cc.SetRotationKeyCache(4 * 1024**3)   # keep rotation keys under 4 GiB -- BEFORE LoadContext
cc.LoadContext(keys.publicKey)        # keys are now created VRAM-free, loading on first use

cc.GetRotationKeyCacheResidentBytes()  # VRAM the resident keys actually occupy
cc.GetRotationKeyCache()               # the budget (None = unlimited, the default)
cc.IsRotationKeyResident(1)            # is the key for rotation index 1 on the GPU?
cc.PinRotationKey(1)                   # never evict a hot key (pin=False to unpin)
cc.OffloadRotationKeys([2, 3])         # evict now instead of waiting for the budget
cc.OffloadRotationKeys()               # ... all of them
cc.SetRotationKeyCache(2 * 1024**3)   # retighten at runtime: evicts down immediately
cc.SetRotationKeyCache(None)           # back to unlimited (every key permanently resident)
```

**Set the budget before `LoadContext()`.** Only keys *created* while a finite budget is in
force keep the host-RAM snapshot that offload/reload round-trips through — and they allocate
no VRAM at all until first used. Keys built with an unlimited budget have no snapshot, so the
cache treats them as pinned forever. There is no way to add rotation keys after the load
either: `EvalRotateKeyGen`, `EvalMultKeyGen`, `EvalBootstrapSetup`, `EvalBootstrapKeyGen` and
the `Deserialize*Key` calls all throw once the context is loaded. Calling `SetRotationKeyCache`
afterwards therefore does not make anything offloadable; it only re-tunes the live budget (and
thus evicts) for keys that already have snapshots.

The budget is **soft**. Ops that need several keys at once — hoisted rotation, the bootstrap
linear transforms — fetch them all before launching anything (an eviction in between would
free a key whose kernels have not been enqueued yet), so they transiently overshoot and the
cache shrinks back on the next load. Measured: plain `EvalRotate` under a one-key budget stays
at exactly one resident key (evict-before-load), while `AccumulateSum(ct, 64)` holds its whole
hoisted batch — 9 keys — whatever the budget says. Budget it to comfortably exceed your largest
such working set, or rotation-heavy code will thrash.

Two things to know:

- **Reload is lossless; rotations are not bit-reproducible anyway.** Offload/reload restores
  the key limbs verbatim, but FIDESlib rotations differ by ~1e-10 run to run at logN=13 even
  with no cache involved (multi-stream reductions), so validate cold-vs-warm results against
  that floor rather than `==`.
- **The budget belongs to the GPU context, not to the `CryptoContext` object.** FIDESlib caches
  GPU contexts by parameters, so two contexts built from *identical* params share one —
  including its budget, its LRU list and its resident-byte counter, and a second
  `LoadContext()` silently replaces the first one's budget. Give concurrently loaded contexts
  different parameters if you want independent budgets.

`OffloadRotationKeys`/`PinRotationKey` throw if the context is not loaded; unknown rotation
indexes are ignored. `tests/test_rotation_key_cache.py` (`--all`, one subprocess per case)
exercises all of the above end to end.

### Bounding plaintext VRAM

A plaintext uploads to the GPU the first time an op uses it and then stays there until its
`Plaintext` object is destroyed, so a host-side plaintext pool — masks, weights, convolution
kernels — pins one device copy per entry. Measured: `(L+1) x N x 8` bytes each (896 KiB at
logN=14/L=6, 6 MiB at logN=16/L=11), so a few thousand of them is tens of GiB of VRAM held
for the whole run. Same idea as above, with a byte budget:

```python
cc.SetPlaintextCache(2 * 1024**3)    # keep plaintexts under 2 GiB -- any time, before or after LoadContext
cc.GetPlaintextCache()               # the budget (None = unlimited, the default)
cc.GetPlaintextCacheResidentBytes()  # VRAM the resident plaintexts actually occupy
pt.IsLoadedOnDevice()                # does this plaintext have a GPU copy right now?
cc.PinPlaintext(pt)                  # never evict a hot plaintext (pin=False to unpin)
pt.UnloadFromDevice()                # drop this one now
cc.OffloadPlaintexts()               # drop every unpinned one now
cc.SetPlaintextCache(None)           # back to unlimited
```

Everything keeps working after an eviction: the next op that needs the plaintext re-uploads it
from the encoding its `Plaintext` carries anyway. That is the reason this cache is much cheaper
than the rotation-key one — a plaintext is **read-only in every op that takes one**, so eviction
is a plain free (no host snapshot, no device sync, no extra host RAM) and a miss is one
host-to-device copy of unchanged data. Measured on 128 plaintexts at logN=14: under a budget of
four, 7.4 MB of live VRAM instead of 122 MB, with identical results.

Two more differences from the rotation-key cache:

- **The budget can be set at any time**, and tightening it evicts immediately — there is nothing
  to arrange before `LoadContext()`.
- **The budget belongs to the `CryptoContext` object**, not to the shared GPU context, so two
  contexts built from identical parameters keep independent budgets and independent accounting.

The budget is **soft** in three ways: a plaintext's size is only known once it is built, so a
load overshoots by that one plaintext before the cache re-shrinks; an op that holds several
plaintexts at once (the convolution transforms fetch a whole batch before launching anything)
keeps all of them until it returns; and a budget smaller than a single plaintext still keeps the
one in use. Pinned plaintexts are never evicted, and only the plaintexts you create are
cached — the ones inside the bootstrapping precomputation belong to the GPU context and are
untouched.

Note that the freed VRAM goes back to FIDESlib's allocator pool, not to the driver, so
`nvidia-smi` and `cudaMemGetInfo` will not show a drop — and `TrimGPUMemoryPool()` only returns
whole idle slabs (measured: it did not move the reserved total at all after 32 plaintext loads).
Use `GetPlaintextCacheResidentBytes()`, or the pool's own live-chunk view, to see the cache
working:

```python
stats = fhe.GetGPUMemoryPoolStats(0)
live = sum(b["live_chunks"] * b["chunk_bytes"] for b in stats["buckets"])
```

`tests/test_plaintext_cache.py` (`--all`, one subprocess per case) covers the byte accounting
against that allocator view, LRU order, pinning, manual offload, runtime budget changes,
plaintext lifetime vs the bookkeeping, per-context independence, the real VRAM cap and a
no-leak run over 240 forced misses.

### Bounding ciphertext VRAM

Same idea for ciphertexts, on top of the `Offload()`/`Reload()` round trip above: give them a
byte budget and the least recently used ones get parked in host RAM instead of filling the GPU.

```python
cc.SetCiphertextCache(8 * 1024**3)    # keep ciphertexts under 8 GiB -- any time
cc.GetCiphertextCache()               # the budget (None = unlimited, the default)
cc.GetCiphertextCacheResidentBytes()  # VRAM the resident (non-offloaded) ciphertexts occupy
cc.PinCiphertext(ct)                  # never evict a hot one, e.g. an accumulator
cc.OffloadCiphertexts()               # park every unpinned one now
cc.SetCiphertextCache(None)           # back to unlimited
```

**This is the expensive one of the three caches — reach for it to survive a working set that
does not fit in VRAM, not to go faster.** An eviction synchronizes the device and copies the
limbs to the host; a miss copies them back; and the host snapshot costs as much RAM as the
ciphertext did VRAM, so you are trading VRAM for host RAM, not saving memory outright. Measured
at logN=14 on a churn of `EvalAdd`s tight enough to force a round trip per op: **0.80 ms/op
against 0.03 ms/op unbounded, ~24x**. Where the dataflow is known, offload explicitly with
`ct.Offload()` (see `examples/03_offload.py`); use a budget when it is not.

What it costs in VRAM per ciphertext is `4 x (L+1) x N x 8` bytes — 3.5 MiB at logN=14/L=6 —
i.e. **twice** what the naive "two polynomials" count suggests, because each ciphertext limb
also owns an equally sized auxiliary vector for the NTTs (a plaintext's constant limbs do not).
And a ciphertext does **not** get cheaper as it descends the levels: FIDESlib keeps the limbs
(the release in `RNSPoly::dropToLevel` is disabled upstream), so size the budget from the top
level. Measured: 64 ciphertexts under a budget of four hold 15.7 MB of live VRAM instead of
236 MB, with identical results.

The one caveat worth knowing: **the budget is enforced at operation boundaries.** The only
automatic eviction point is the `LoadCiphertext()` every operation performs on its operands
before it takes a single GPU pointer — which is what makes eviction safe, since operations hold
raw pointers to ciphertexts across fetches (a hoisted rotation builds a whole batch of them).
Ciphertexts created *during* an operation are therefore counted but not evicted until the next
one starts, so expect an overshoot of one operation's working set — measured at 2 ciphertexts
for `EvalAdd` even under a 1-byte budget, shrinking back on the next call. A ciphertext caught
in the extended basis mid-key-switch cannot be snapshotted and is skipped as well, and pinned
ones are never evicted.

The three caches are independent and compose (`tests/test_ciphertext_cache.py --scenario
with_other_caches` runs a mult+rotate with all three budgets binding at once). Note that
`GetCiphertextCacheResidentBytes()` is again the metric to watch rather than `nvidia-smi`, and
that FIDESlib's auxiliary-polynomial cache sits between a destroyed ciphertext and the
allocator — it retains the polynomials, so dropping ciphertexts frees nothing visible until you
call `cc.ClearAuxiliaryPolyPool()` (or cap it with `FIDESLIB_AUX_POLY_CACHE_LIMIT`).

`tests/test_ciphertext_cache.py` (`--all`, one subprocess per case) covers all of the above:
accounting against the allocator, LRU order, pinning, manual offload/reload, runtime budget
changes, the operation-boundary overshoot, ciphertext lifetime (resident and offloaded),
per-context independence, the VRAM cap, a no-leak run over 240 forced round trips, and
add/sub/mult/square/rotate/rescale/accumulate under a one-ciphertext budget.

## Examples (in suggested order)

| Script | What it shows | Needs |
|---|---|---|
| `examples/00_onboarding.py` | context → keys → encrypt → add/mult/rotate/sum → decrypt | <1 GB VRAM, seconds |
| `examples/01_chebyshev.py` | deg-31 polynomial + X4 cleaning vs CPU Clenshaw reference | ~1 GB VRAM, seconds |
| `examples/02_step_herminirocket.py` | full Step() (Lee α=8 + 2×X4) at logN=17, **secure** params | ~2–4 GB VRAM, ~1–2 min |
| `examples/03_offload.py` | offload/reload ciphertexts to host RAM, reclaim VRAM with `TrimGPUMemoryPool` | <1 GB VRAM, seconds |

## Coming from openfhe-python

| openfhe-python | fideslib_py |
|---|---|
| `cc.EvalSum(ct, n)` | `cc.AccumulateSum(ct, n, stride=1)` |
| `cc.EvalSumKeyGen(sk)` | `cc.EvalRotateKeyGen(sk, fhe.accumulate_rotation_indices(n, stride))` |
| — | `cc.LoadContext(keys.publicKey)` — required once, after all keygen |
| `cc.EvalChebyshevSeries(ct, coeffs, a, b)` | same |
| `GetSchemeSwitchingData`, FHEW comparisons, BFV/BGV | not available (CKKS only) |

All `Eval*` calls release the GIL, so a Python timing/monitoring thread stays
responsive. A single dispatch thread is enough — the GPU serializes the work.

## Caveats

- **Never `import openfhe` and `import fideslib_py` in the same process** —
  they embed different OpenFHE versions (1.5.0 vs patched 1.5.1). Run CPU/GPU
  comparisons as separate processes.
- Rotation keys must exist for every index used by `EvalRotate` /
  `AccumulateSum` (helper: `fhe.accumulate_rotation_indices`).
- GPU out-of-memory aborts the process (FIDESlib behavior) — check
  `nvidia-smi` before logN=17 runs on a shared GPU.
- `numpy` arrays are accepted wherever a list of floats is (converted on the
  way in); returned values are Python lists.
