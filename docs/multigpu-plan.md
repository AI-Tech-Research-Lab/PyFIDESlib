# FIDESlib multi-GPU: findings and remediation plan

Investigation date: 2026-09-04.
Code under test: `AI-Tech-Research-Lab/FIDESlib` @ `a84aabe` (the commit PyFIDESlib pins in
`CMakeLists.txt`), exercised through `fideslib_py`.
Hardware: 4x NVIDIA H100 NVL (93.6 GiB each). NVLink (NV12) between GPU0-GPU1 and GPU2-GPU3;
across those pairs only PCIe + cross-NUMA interconnect (`SYS`). Note GPU1 was occupied
throughout by an unrelated job (~85 GB, 99% utilisation) -- this turned out to matter.

Written in English to match the rest of the repo and because parts of it are meant to become
upstream issues on `CAPS-UMU/FIDESlib`.

---

## 0. What was done (2026-09-05)

Phases 0-4 and 6 of the plan below are complete, on `AI-Tech-Research-Lab/FIDESlib`
`main`. F1, F2, F3, F5 and F6 are fixed; F4 is not fixed but is now reproducible on demand,
proven to be a race, localised to one code path, and mitigated by a default change.

| finding | state | commit |
|---|---|---|
| F1 two NaN fixes lost in the resync | recovered, with their regression tests | `57d3e6c`, `c00a45e` (+ build guards `056d3a2`, `048234d`) |
| F2 `freeSpecialLimbs` waits the wrong way | fixed, and the same shape in `freeLimbs` | `695d041` |
| F3 abort when #GPUs > K | fixed: guard + diagnostic at context creation | `359ca78` |
| F5 `invalid resource handle` at teardown | fixed; root cause was worse than teardown | `4af3de1` |
| F6 no multi-GPU test coverage | `--devices` for gtest, plus `tests/test_multigpu.py` here | `601a4c5`, PyFIDESlib `1927ad7` |
| F4 NaN from the extended hoisted rotation | **not fixed**; mitigated by defaulting to NCCL | `efd16ef` |

Three things worth reading past the table.

**F5 was not a teardown bug.** `GPUmalloc` indexes the caching memory pool by the CUDA
device ordinal; the eight `GPUfree` calls in `LimbPartition` passed `id`, the partition's
index into `GPUid`. The two agree only when `GPUid` is the identity -- one GPU, or
`[0,1,...]` -- which is why nothing caught it. When they differ, a chunk freed by partition
k lands in the pool of device k while living on device `GPUid[k]`, so a later allocation on
ordinal k can be served another device's memory: with `devices=[0,2,3]`, partition 2 (device
3) frees into device 2's pool. The crash at teardown was the same mix-up surfacing as an
event recorded on the wrong device's stream. Any pooled memory in a non-identity device set
was suspect, which makes this the most consequential fix of the batch -- and a candidate for
part of what F4 looked like.

**F4 now has an on-demand reproducer.** The plan called needing a foreign job "not a working
basis". `tests/tools/gpu_spin.cu` puts one resident block per SM on each GPU, starving it of
scheduling slots the way a saturated neighbour does. On two otherwise idle H100s that turns
AccumulateSum from 0 failures in 6 runs into 6 in 6 -- no borrowed hardware needed.

**F4 is localised, and the plan's suspect list was wrong about where.** With the reproducer:

* `CUDA_LAUNCH_BLOCKING=1` is clean (4/4) -- it is a race.
* `FIDESLIB_USE_MEMCPY_PEER=0`, which swaps the peer-copy special-limb exchange for
  `ncclBroadcast`, is clean (10/10 against 0/10 for the default). The bug is in the peer-copy
  path, not in the multi-GPU key switch at large.
* A plain `EvalRotate` and a mult chain are clean under the same load, confirming it is the
  extended path (`rotate_hoisted(..., ext=true)`) alone.
* An all-device `cudaDeviceSynchronize` at each of the five boundaries phase 5.2 proposed --
  after the result `extend`, after `modup`, after `fusedHoistRotate`, on the way out of
  `rotate_hoisted`, and after the extended `add` -- changes nothing. **The missing dependency
  is inside the exchange, not between it and its callers**, so the "insert a sync and see"
  method has already been run to exhaustion; what is left is `moddownMGPU`'s
  `cudaMemcpyPeerAsync` block (`src/CKKS/LimbPartitionMGPU.cu:2539`) and its counterpart in
  `modupMGPU`. Suspect (2) from F4 below -- the relative-comparison spin barriers -- is not
  ruled out, since those sit inside `modupMGPU`; suspect (1), a missing wait around
  `fusedHoistRotate`, is: a full drain there does not help.

Because contention exposes the race rather than creating it, the peer-copy path could not
stay the default. It is now off; `FIDESLIB_USE_MEMCPY_PEER=1` restores it. The cost is ~20%
on AccumulateSum (9.4 vs 7.8 ms at logN=16, L=24, two NVLinked H100s), still ahead of 10.7 ms
on one GPU.

### Which operations F4 actually reaches

The original finding below says bootstrap escapes the extended path because its linear
transforms pass `ext=false`. That is true of the linear transforms and wrong about bootstrap:
`Bootstrap.cu:100` and `:263` call `Accumulate`, which is the extended path. Measured, three
GPUs under the spin load, on the peer-copy transport:

| | result |
|---|---|
| add, sub (ct-ct and ct-plaintext), scalar multiply, negate | 420/420 checks clean |
| multiply ct*ct with relinearisation, square, multiply by plaintext | clean |
| rotations (1, 2, 3, 7), depth-4 chain with rescale, rotate straight off a multiply | clean |
| `AccumulateSum` | **12 failures / 30** |
| `CoeffsToSlots`, `SlotsToCoeffs` (linear transforms, `ext=false`) | 8/8 each |
| `OpenFHEBootstrapDense` | 8/8 |
| `OpenFHEBootstrap`, `IterativeBootstrap` (sparse slots) | **11 failures / 16** |

So ordinary arithmetic and rotation never touch the race -- not once in 420 checks on the
broken transport -- and bootstrap does, through its `Accumulate` step. Dense bootstrap is
clean for a structural reason rather than by luck: `Accumulate(ctxt, bStep, stride, size)` is
called with `stride = slots`, `size = N/2/slots`, so at full slots `size == 1` and the loop
never runs. Sparse bootstrap, which is what a level budget usually implies, does run it.

That is the answer to "is this needed for real use": on the peer-copy transport it was, for
anything that bootstraps or accumulates. On the NCCL default it is not -- same hardware, same
load, the bootstrap suite goes from 29/40 to **40/40**, and the mixed workload from 9 failing
runs in 10 to **0 in 10**. What remains is a latent race in a non-default code path.

### Suite results

`fideslib-test` on one GPU: **424/424**. The same suite on three (`--devices=0,1,2` over
physical 0, 2, 3, the first time it has ever run multi-GPU): **423/424** before the Conjugate
tolerance fix below, 424/424 after.

The bootstrap tests on three GPUs *under the spin load*: **40/40** on the NCCL default,
against 29/40 on the peer-copy transport.

`tests/test_multigpu.py --devices 0,2,3` here: all cases pass, and the contention group
passes under load.

The one multi-GPU failure was `OpenFHEInterfaceTest.Conjugate/1` (FIXEDAUTO), a precision
margin rather than a wrong result: values correct to ~46 bits, no NaN. Repeating it 5x each
way, the max error is systematically about 1.3x higher on three GPUs (6.0-9.6e-14) than on one
(3.0-8.6e-14) -- ordinary floating-point reassociation, since the limbs are summed in a
different order. It crossed the bound only because `ASSERT_ERROR_OK` derives its threshold
from the CPU reference's own precision estimate, which fluctuates by two bits run to run, and
this test takes 24 samples per run: every 3-GPU failure coincided with the tightest draw,
which never came up in the 1-GPU repeats. Given 2 bits of slack (`ASSERT_ERROR_OK_SLACK`,
worst observed ratio 1.74, so a 2.3x margin) it passes 8 repeats out of 8 on three GPUs.

### Known limitation, pre-existing and not multi-GPU

`EvalBootstrapSetup()` segfaults through the Python API on a single GPU as well, for every
parameter combination tried -- see the header of `tests/diag_bootstrap.py`, which predates
this work. `tests/mgpu_case.py` has a bootstrap case ready (`op=bootstrap`) but the suite
does not run it until that is fixed, so bootstrap has no multi-GPU coverage here.

### Reproducing

```
cmake -B build-mgpu -S . -DFIDESLIB_ARCH=90-real \
      -DFETCHCONTENT_SOURCE_DIR_FIDESLIB=/path/to/FIDESlib
cmake --build build-mgpu -j$(nproc)          # ~1 min incremental, was 4m27s

python tests/test_multigpu.py --devices 0,2,3          # 35 cases, all must pass
nvcc -O2 -arch=sm_90 -o tests/tools/gpu_spin tests/tools/gpu_spin.cu
python tests/test_multigpu.py --devices 2,3 --group contention --contend
/path/to/FIDESlib/build/fideslib-test --devices=0,2,3  # the gtest suite, multi-GPU
```

---

## 1. Status summary

Multi-GPU is compiled in and active: NCCL is linked into the Python module
(`libnccl.so.2`, `ncclCommInitRank` & co. are undefined symbols resolved at load).
Without NCCL, `Context.cu:49` calls `exit(-1)` as soon as more than one device is
requested -- a hard process kill, not an exception. It is therefore a build-time
dependency, not an optional runtime feature.

**Works, verified on 1/2/3/4 GPUs:** `EvalAdd`, `EvalSub`, `EvalMult` (ct*pt, ct*scalar and
ct*ct with relinearisation), `EvalSquare`, single `EvalRotate`, multiplicative chains with
rescale, and `EvalBootstrap`. A 30-iteration stress of depth-7 chains with interleaved
rotations on 4 GPUs produced 0 errors. The genuine multi-GPU key switch
(`RNSPoly::modup_ksk_moddown_mgpu`, `src/CKKS/RNSPoly.cpp:1151`) is solid.

**Broken:** the *extended* hoisted-rotation path -- `AccumulateSum` / `Broadcast`, i.e.
`Ciphertext::rotate_hoisted(..., ext=true)`. It either aborts the process (finding F3) or
returns NaN intermittently (finding F4). Bootstrap's linear transforms call
`rotate_hoisted(..., false)` (`src/CKKS/LinearTransform.cu:869,1054,1315`), while
`Accumulate` uses `ext=true` (`src/CKKS/AccumulateBroadcast.cu:33`). (Corrected 2026-09-05:
this was read at the time as bootstrap escaping the extended path altogether. It does not --
`Bootstrap.cu` calls `Accumulate` itself, at lines 100 and 263. See section 0.)

---

## 2. Findings

### F1 (highest value) -- two NaN fixes were lost in a branch resync

The fork's `pre-sync-main` branch holds a 2026-07-08/10 corruption hunt that produced two real
fixes:

| commit | title | fix |
|---|---|---|
| `3df8889` | Fix all-NaN from the first key-switch at ciphertext level >= 2 | one-time `cudaDeviceSynchronize()` in the first-creation branch of `getKeySwitchAux` / `getKeySwitchAux2` / `getModdownAux` |
| `a8f3112` | Fix NaN when chaining an op straight off a ciphertext from a preceding mult | `ct_gpu->c0.sync(); ct_gpu->c1.sync();` before the deep copy in `CryptoContextImpl::CopyDeviceCiphertext` |

`3df8889` root-caused a cross-stream race in the caching GPU pool: the per-context key-switch
scratch is allocated lazily, once, recycling chunks whose previous tenant still has GPU writes
in flight. `a8f3112` root-caused `CopyDeviceCiphertext` -- used by *every* `api::EvalXxx` call --
deep-copying a ciphertext without draining its per-limb streams.

The revert commit that closed the hunt (`57e4bb2`) states explicitly:
*"The two NaN fixes (3df8889, a8f3112) stay: both have regression tests that fail without them."*

**They are not in `origin/main`.** Verified by content, not by hash: in `main`,
`getKeySwitchAux()` has no sync before the pool allocation, and `CopyDeviceCiphertext` calls
`new_ct->copy(*ct_gpu)` with no drain. Their regression tests
(`FirstUseKeySwitchTests.cu`, `SecondRotateApiRepro.cu`) are absent from `test/`.

Commits present on `pre-sync-main` and missing from `origin/main`:

```
Add FIDESLIB_PARANOID_BOOT: drain between bootstrap stages            (intentionally dropped)
Add FIDESLIB_PARANOID_FREE: device drain before any chunk re-enters   (intentionally dropped)
Add FIDESLIB_PARANOID_OPS: drain the device at every operand fetch    (intentionally dropped)
Fence limb-buffer frees: fold per-limb streams before chunks re-enter (intentionally dropped)
Fence the aux-poly pool: drain streams before a buffer becomes reallocatable (intentionally dropped)
Revert race-hunt instrumentation and speculative pool fences          (intentionally dropped)
Fix all-NaN from the first key-switch at ciphertext level >= 2        <-- REGRESSION
Fix NaN when chaining an op straight off a ciphertext from a preceding mult <-- REGRESSION
Guard CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE for CUDA < 12.4 <-- lost build guard
Guard <nccl.h> include behind #ifdef NCCL in tests                    <-- lost build guard
```

The six "intentionally dropped" ones are the instrumentation plus its own revert: net zero,
correctly gone. The other four should not have been.

The last guard matters here too: without it `test/PeerUtilsTests.cu` and `test/Microbench.cu`
include `<nccl.h>` unconditionally, so the gtest suite will not build on a machine without NCCL.

**Every measurement in this document was taken without these two fixes**, since PyFIDESlib
pins `a84aabe`, a descendant of `main`. Both fixes concern timing-dependent races on pool reuse
and undrained copies, and both touch the very scratch buffers the multi-GPU key switch uses --
so an unknown share of finding F4 may simply be this regression.

### F2 -- `freeSpecialLimbs` waits in the wrong direction

Flagged in `57e4bb2` ("reported rather than carried here") and still unfixed.
`src/CKKS/LimbPartition.cu:816`:

```cpp
void LimbPartition::freeSpecialLimbs() {
    cudaSetDevice(device);
    for (size_t i = 0; i < SPECIALlimb.size(); ++i)
        STREAM(SPECIALlimb.at(i)).wait(s);   // wrong direction
    SPECIALlimb.clear();
    if (bufferSPECIAL != nullptr) { GPUfree(bufferSPECIAL, id, 0, s.ptr()); bufferSPECIAL = nullptr; }
}
```

It makes each special limb's stream wait on the partition stream and then frees immediately.
The dependency needed is the opposite: `s`, which owns the free, must wait for pending work on
the limb streams before `bufferSPECIAL` goes back to the pool. This is the special-limb
machinery -- exactly the code path the extended hoisted rotation exercises, i.e. where the NaNs
of F4 appear.

### F3 -- deterministic abort when #GPUs > K

`generateSplitSpecialMeta` (`src/CKKS/Context.cu:958`) splits the K special primes across the
devices; with more devices than special primes the trailing ones get an empty vector. The
extended branch of `LimbPartition::add` (`src/CKKS/LimbPartition.cu:478`) then evaluates
`cc.splitSpecialMeta.at(id).at(0).id` and throws `std::out_of_range`, which nothing catches:
the process aborts.

Empirical rule, **#GPUs > ceil(#towers / dnum)**, verified:

| depth (towers) | dnum | K | aborts from |
|---|---|---|---|
| 4 (5) | 5 | 1 | 2 GPUs |
| 4 (5) | 3 | 2 | 3 GPUs |
| 4 (5) | 2 | 3 | 4 GPUs |
| 8 (9) | 3 | 3 | 4 GPUs |
| 12 (13) | 5 | 3 | 4 GPUs |
| 12 (13) | 3 | 5 | no abort up to 4 |
| 24 (25) | 2 | 13 | no abort up to 4 |

Backtrace: `LimbPartition::add` <- `RNSPoly::add` (inside `GOMP_parallel`) <- `Ciphertext::add`
<- `FIDESlib::CKKS::Accumulate` <- `CryptoContextImpl::AccumulateSum`.

Roughly ten sites index `splitSpecialMeta.at(id).at(0)` with no guard:
`LimbPartition.cu:478,491,872,2029,2257,2441,2463,2514,2567` and
`LimbPartitionMGPU.cu:1314,2526`.

A design decision is needed before writing the fix: **is a GPU owning zero special primes a
legitimate configuration?** If yes, every site must degrade to a no-op for the special limbs
and the downstream kernels must be checked for the same assumption. If no, `ContextData` should
reject the configuration up front with a clear error instead of aborting mid-computation.
Recommended: both -- guard for correctness, plus an explicit diagnostic.

### F4 -- NaN from the extended hoisted rotation, under GPU contention

`AccumulateSum` returns NaN intermittently. Non-deterministic: different accumulate sizes fail
on different runs of the same binary and parameters, which rules out an indexing bug and points
at a race.

The trigger is **contention, not GPU count**. Every failing device set included GPU1, the GPU
held at 99% by a foreign job:

| device set | GPU state | result |
|---|---|---|
| `[2,3]` | idle | 4/4 runs OK |
| `[2,3]` + our own synthetic load | loaded | 4/4 runs OK |
| `[0,2,3]` | idle | 4/4 runs OK (values correct) |
| `[0,1]` | GPU1 at 99% | ~1 run in 3 has NaN |
| `[0,1]` + our own synthetic load | heavily loaded | 6/6 runs had NaN |
| `[0,1,2]`, `[0,1,2,3]` | GPU1 at 99% | frequent NaN |

So light co-scheduling is tolerated; a genuinely saturated peer is not. On an exclusive machine
the multi-GPU path computes correct results today.

Structural suspects, in order:

1. **`LimbPartition::fusedHoistRotate` (`src/CKKS/LimbPartitionMGPU.cu:457`) has no cross-device
   wait.** It waits on local streams only (`s.wait(src_c0.getS())`, `s.wait(c0[i]->s)`), yet the
   DIGIT limbs it reads were gathered from peers by `modupMGPU`. By contrast the non-hoisted
   path (`modup_ksk_moddown_mgpu`) carries the `signals`/`thread_stop` handshake through the
   whole fusion -- and that path never produced a wrong result in any test. This asymmetry is
   the strongest single clue.
2. **The CPU-side spin barriers in `modupMGPU` (`src/CKKS/LimbPartitionMGPU.cu:730-775`)** are
   relative comparisons on shared counters -- `while (*thread_stop[id] >= *thread_stop[id-1])`,
   `while (*thread_stop[id] > *thread_stop[cc.GPUid.size()-1])` -- rather than barriers on an
   expected value. With perturbed timing a thread can find the condition already satisfied and
   pass early. This is precisely the failure class that only shows up under contention.
3. **The cross-device polling kernels** (`src/PeerUtils.cu`: `p2p_polling_kernel`,
   `hostpin_polling_kernel`) spin on flags, which assumes exclusive GPUs. On a co-scheduled GPU
   the notifying kernel may not get SMs while the waiting kernel holds them.

If the cause is (2) or (3), the right fix is not to repair the condition but to replace the
ad-hoc handshake with **cross-device CUDA events** (`cudaEventRecord` on the producer,
`cudaStreamWaitEvent` on the consumer, which works across devices): driver-guaranteed ordering,
independent of SM scheduling. That is a redesign of the multi-GPU synchronisation layer, not a
patch, and it is why the estimate for this work is wide.

### F5 -- `invalid resource handle` during context teardown

With `devices=[0,2,3]` the computation is correct but destroying the context fails
deterministically (4/4 runs):

```
Cuda failure .../src/CudaUtils.cu:237: 'invalid resource handle'
  FIDESlib::GPUfree
  FIDESlib::CKKS::LimbPartition::~LimbPartition
  std::vector<LimbPartition>::~vector
  FIDESlib::CKKS::ContextData::~ContextData
  ...
  fideslib::CryptoContextImpl<unsigned int>::~CryptoContextImpl
```

`CudaUtils.cu:237` is inside `Stream::wait`, around
`cudaEventRecordWithFlags(ev, s, ...)` / `cudaStreamWaitEvent(ptr_, ev, ...)`. `invalid resource
handle` there means the event or the stream belongs to a different device than the current one,
or has already been destroyed. It does not reproduce with `[2,3]` or `[0,1,2]`, so it is
device-set dependent but deterministic for a given set -- which makes it the easiest of the
three to pin down.

It matters beyond process exit: a long-running service that creates and destroys contexts turns
this into a runtime crash.

### F6 -- multi-GPU has no test coverage anywhere

`test/ParametrizedTest.cuh:31` hardcodes `inline std::vector<int> devices{ 0 }` and
`test/ParameterizedTest.cu`'s `main` never overrides it, so the whole gtest suite only ever runs
on one GPU. The only tests that touch two devices are in `test/PeerUtilsTests.cu`, and they
exercise the raw P2P primitives, not CKKS operations. This is why the findings above survived.

---

## 3. Characterisation of multi-GPU resource sharing

Not bugs, but they set expectations and should inform whether this work is worth it.

### How VRAM is shared

The split is **per limb, not per object**: `ContextData::generateMeta`
(`src/CKKS/Context.cu:249`) distributes towers round-robin with `int dev = i % GPUid.size()`.
A single ciphertext is therefore spread over every GPU in the context; you cannot pin
ciphertext A to GPU0 and B to GPU1. Plaintexts are the same (`Plaintext` holds one `RNSPoly`).

Key-switching keys are **partially replicated**: `generateGPUdigits`
(`src/CKKS/Context.cu:333`) gives every digit to every GPU, and `generateDecompMeta`
(`src/CKKS/Context.cu:145`) then collects, for each such digit, limbs from the *global* meta --
so the DECOMP part of every key is rebuilt in full on each device. Multi-GPU keys additionally
allocate the regular limbs (`src/CKKS/KeySwitchingKey.cu:43`, `grow(cc->L, false, true)`).

Measured at logN=15, depth 12 (13 towers), dnum=3, 32 rotation keys + relin, 200 ciphertexts;
figures net of the ~538 MiB CUDA primary context each GPU carries:

| | 1 GPU | 2 GPUs | 4 GPUs |
|---|---|---|---|
| rotation keys, aggregate | 864 MiB | 1280 MiB | 1696 MiB |
| rotation keys, per GPU | 864 | 640 | **424** |
| 200 ciphertexts, aggregate | 3072 MiB | 3072 MiB | 4096 MiB |
| 200 ciphertexts, peak per GPU | 3072 | 2048 | **1024** |
| fixed working buffers, per GPU | ~316 MiB | ~800 MiB | ~950 MiB |

So: **ciphertexts and plaintexts scale nearly linearly** (~3x the capacity per GPU at 4 GPUs,
aggregate cost roughly flat); **keys only reach ~2x at 4 GPUs** (68% efficiency at 2, 51% at 4),
because of the DECOMP replication; and there is a **fixed tax** -- per-GPU working buffers grow
from ~316 MiB to ~800-950 MiB once multi-GPU mode is on, largely `bufferGATHER`, which is sized
over all L+1 towers (`generateGatherMeta`, `src/CKKS/Context.cu:315`) and allocated on every
device.

Also worth knowing: the process opens a ~538 MiB CUDA context on **every visible GPU**, even
ones absent from `SetDevices`. Use `CUDA_VISIBLE_DEVICES` if those GPUs are needed elsewhere.

### Performance

Single-ciphertext latency, `EvalRotateInPlace` (key switch), logN=16, L=24, mean over 100 ops,
measured only on idle GPUs:

| devices | ms/op |
|---|---|
| `[0]` | 2.102 |
| `[2]` | 1.958 |
| `[2,3]` (NVLink pair) | 1.774 |

About +10% from two NVLinked GPUs, not 2x. Any earlier figure involving GPU1 is invalid --
that GPU was saturated by a foreign job. The 4-GPU configuration could not be measured cleanly
for the same reason, and spans the cross-NUMA PCIe hop anyway.

Practical reading: FIDESlib's multi-GPU buys **capacity for one problem too large for a single
GPU**, not throughput. For batch-parallel workloads, one process per GPU with
`CUDA_VISIBLE_DEVICES` gives linear capacity, no synchronisation overhead and none of F3/F4/F5.

---

## 4. Reproducers

Minimal repro for F3 and F4. Vary `devices`; `dnum` and `depth` set K per the F3 table.

```python
import sys, json
sys.path.insert(0, "/home/falcettaa/PyFIDESlib")
import fideslib_py as fhe

devs, dnum, depth, logN = json.loads(sys.argv[1]), 3, 24, 14

p = fhe.CCParams()
p.SetSecurityLevel(fhe.HEStd_NotSet); p.SetRingDim(1 << logN)
p.SetMultiplicativeDepth(depth); p.SetScalingModSize(50); p.SetFirstModSize(60)
p.SetNumLargeDigits(dnum); p.SetBatchSize(1 << (logN - 1))
p.SetScalingTechnique(fhe.FLEXIBLEAUTO); p.SetKeySwitchTechnique(fhe.HYBRID)
p.SetDevices(devs)

cc = fhe.GenCryptoContext(p)
for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.FHE):
    cc.Enable(f)
k = cc.KeyGen(); cc.EvalMultKeyGen(k.secretKey)
idxs = set()
for n in (2, 4, 8, 64):
    idxs |= set(fhe.accumulate_rotation_indices(n, stride=1))
cc.EvalRotateKeyGen(k.secretKey, sorted(idxs))
cc.LoadContext(k.publicKey)

x = [float(i + 1) for i in range(64)]
ct = cc.Encrypt(k.publicKey, cc.MakeCKKSPackedPlaintext(x))
for n in (2, 4, 8, 64):
    r = cc.AccumulateSum(ct, n)
    pt = cc.Decrypt(k.secretKey, r); pt.SetLength(1)
    v, exp = pt.GetRealPackedValue()[0], sum(x[:n])
    print(f"n={n}: {'OK' if abs(v - exp) < 1e-4 else f'WRONG({v} vs {exp})'}", flush=True)
```

* **F3:** run with a `(depth, dnum, #GPUs)` combination above the K threshold -- e.g.
  `'[0,1,2,3]'` at `depth=8, dnum=3`. Deterministic abort.
* **F4:** run on a device set containing a GPU saturated by another workload, or start a
  competing job first. NaN on a varying subset of `n`.
* **F5:** run with `'[0,2,3]'`. Values correct, `invalid resource handle` at teardown.

Useful switches while debugging (`src/CKKS/LimbPartitionMGPU.cu:47-49`):
`FIDESLIB_USE_MEMCPY_PEER` (default 1), `FIDESLIB_USE_GRAPH_CAPTURE` (default 0),
`FIDESLIB_USE_PEER_ACCESS` (default 0).

---

## 5. Plan

### Phase 0 -- make iteration possible (~half a day)

* The fork is at `~/FIDESlib` (`origin` = `AI-Tech-Research-Lab/FIDESlib`,
  `upstream` = `CAPS-UMU/FIDESlib`), already synced to `a84aabe`. Work on a `mgpu-fixes` branch
  and point the build at it with `-DFIDESLIB_REPOSITORY=/home/falcettaa/FIDESlib
  -DFIDESLIB_GIT_TAG=mgpu-fixes` (or `FETCHCONTENT_SOURCE_DIR_FIDESLIB`). Editing
  `build/_deps/fideslib-src` in place is not an option: it is a FetchContent clone at a pinned
  tag, and the changes are neither committable nor durable.
* Cut the build time. `FIDESLIB_ARCH` currently defaults to
  `80-real;86-real;89-real;90-real;90-virtual;100-real;120-real`; a measured incremental rebuild
  of a single `.cu` took **4m27s**. On H100 only `-DFIDESLIB_ARCH=90-real` is needed. A ten-minute
  edit/test loop makes Phase 4 impractical.
* Minimum viable regression harness: the reproducer above, parameterised over
  (devices x dnum x depth x logN), one subprocess per case, asserting values rather than absence
  of crashes. Full version in Phase 6.
* Machine access. F4 needs both exclusive runs (to validate fixes) and controlled contention
  (to reproduce). GPU1 currently belongs to somebody else.

### Phase 1 -- recover what the resync lost (~1 day, high confidence)

Cherry-pick `3df8889`, `a8f3112` and the two build guards from `pre-sync-main`, restore their
regression tests, push, re-pin `FIDESLIB_GIT_TAG` in PyFIDESlib's `CMakeLists.txt`. Then
**re-run the F4 reproducer under contention**.

This goes first because it redefines what is left to hunt. Both fixes address timing-dependent
races -- pool chunk reuse with writes in flight, and an undrained deep copy -- on the same
scratch buffers the multi-GPU key switch uses. Some unknown fraction of F4 may be this
regression rather than a multi-GPU bug at all.

### Phase 2 -- `freeSpecialLimbs` (~half a day)

Invert the wait direction and re-measure F4. Near-zero cost, and it is the most direct candidate
for NaNs in the extended path.

### Phase 3 -- F3, the `#GPUs > K` abort (~2 days, high confidence)

Take the design decision above, then guard the ~10 sites. Acceptance criterion is the K table in
F3: every row must either compute correctly or fail with a clear diagnostic at context creation.

### Phase 4 -- F5, the teardown failure (~1 day)

Deterministic on `[0,2,3]`: break on `CudaUtils.cu:237` under `cuda-gdb`, inspect the current
device and the owning devices of `ev` and `ptr_`. Likely a missing `cudaSetDevice` in the
destructor path, or an event awaited on another device's stream after its primary context is
gone. Worth doing before Phase 5 as a warm-up on the same device/stream lifetime code.

### Phase 5 -- whatever remains of F4 (1-2 weeks, low confidence)

Only after Phases 1-2, since they may remove it.

1. **Get an on-demand reproducer.** Needing a foreign job is not a working basis. Options, in
   order of practicality: MPS with a low `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` to starve SMs in a
   controlled way; a spin kernel of tunable duration launched by us; injected delays at chosen
   points.
2. **Localise the missing dependency by elimination.** Insert an all-device
   `cudaDeviceSynchronize()`, one at a time, at four points in `Ciphertext::rotate_hoisted`
   (`src/CKKS/Ciphertext.cpp:1125` onward): after `in.modup()`, after `fusedHoistRotate`, after
   `extend()`, after the extended `add`. The first one that removes the NaNs locates it.
3. **Fix**, per the three suspects in F4 -- most likely by replacing the ad-hoc handshake with
   cross-device CUDA events.

On tooling, without illusions: `compute-sanitizer` is available
(`/usr/local/cuda-13/bin/compute-sanitizer`), but `racecheck` only finds intra-kernel shared
memory races, not missing cross-stream/cross-device ordering -- it will not help here.
`initcheck` is useful to separate stale data from uninitialised. `CUDA_LAUNCH_BLOCKING=1` is
useful to confirm a race. The decisive tool is **Nsight Systems**: the timeline shows directly
whether a consumer kernel starts before its peer transfer completed. This is the same
methodology the 2026-07 hunt used successfully.

**A caution from that precedent.** That campaign produced five commits of "paranoid" drains
that turned out to have no effect on the corruption rate: the bug being chased was not a race at
all but a CKKS bootstrap K-overflow, fixed client-side. Before adding synchronisation, prove it
is a race -- clean under `CUDA_LAUNCH_BLOCKING=1`, clean under `initcheck`, and the failure
disappearing with a sync at one specific point and not at another. For F4 the dependence on
contention is a strong indication but not yet proof.

### Phase 6 -- test coverage

Consolidate the harness into `tests/test_multigpu.py`, and add a `--devices` command-line option
to the upstream gtest suite so the whole of it can run multi-GPU. Both should be pulled forward
into Phase 1 in whatever minimal form validates the cherry-picks.

---

## 6. Estimates

| phase | effort | confidence |
|---|---|---|
| 0 - tooling | 0.5 d | high |
| 1 - recover lost fixes | 1 d | high |
| 2 - `freeSpecialLimbs` | 0.5 d | medium |
| 3 - `#GPUs > K` abort | 2 d | high |
| 4 - teardown | 1 d | high |
| 5 - residual race | 1-2 weeks | **low** |
| 6 - test coverage | 2-3 d | high |

Phases 0-4 are ordinary work and yield a multi-GPU that does not crash. Phase 5 is the one that
cannot be estimated honestly: if the cause is suspect (1), it is a handful of events; if it is
(2) or (3), it is a rewrite of the synchronisation layer.

## 7. Decisions needed

1. **Upstream first?** F2-F6 are upstream bugs, not fork bugs. Filing them on
   `CAPS-UMU/FIDESlib` with the reproducers -- before investing weeks -- may reveal work already
   in progress. F1 is ours alone.
2. **Is a GPU with zero special primes legal?** Blocks the Phase 3 fix (see F3).
3. **Is multi-GPU actually required?** If the workload is batch-parallel, one process per GPU
   with `CUDA_VISIBLE_DEVICES` gives linear capacity, no synchronisation overhead and none of
   these bugs, today. FIDESlib's multi-GPU is worth the spend only if a single problem does not
   fit in one GPU -- and, per section 3, it buys ~3x ciphertext capacity but only ~2x key
   capacity at 4 GPUs.
