# fideslib_py

Python bindings (pybind11) for [FIDESlib](https://github.com/CAPS-UMU/FIDESlib), a GPU
implementation of CKKS that interoperates with OpenFHE. The API follows openfhe-python and
covers context setup, key generation, encoding, encryption, leveled arithmetic, rotations,
`EvalChebyshevSeries`, `AccumulateSum`, bootstrapping and Fourier-extension functional
bootstrapping, on one or more GPUs.

## Installation

Requirements:
- CUDA toolkit ≥ 12.4
- gcc ≥ 11
- CMake ≥ 3.25
- NCCL
- Python development headers
- an SSH key with access to `AI-Tech-Research-Lab/FIDESlib`

```bash
./build.sh                    # or: ./build.sh /path/to/python
export PYTHONPATH=/path/to/PyFIDESlib
```

The build clones and compiles a pinned FIDESlib fork, and the OpenFHE that FIDESlib vendors,
inside `build/`. This version targets **FIDESlib 2.2.0 and patched OpenFHE 1.6.1**,
including the fork's VRAM caches. CMake rebuilds OpenFHE when the installed version is
incompatible with the pin. The fork and commit are `FIDESLIB_REPOSITORY` / `FIDESLIB_GIT_TAG` in
`CMakeLists.txt`. The first build takes 10-30 minutes because it compiles OpenFHE. The module
only works with the Python minor version it was built with.

## Usage

```python
import fideslib_py as fhe

params = fhe.CCParams()
params.SetSecurityLevel(fhe.HEStd_128_classic)
params.SetRingDim(1 << 16)
params.SetMultiplicativeDepth(20)
params.SetScalingModSize(50)
params.SetScalingTechnique(fhe.FLEXIBLEAUTO)
params.SetKeySwitchTechnique(fhe.HYBRID)
params.SetDevices([0])                    # several GPUs: [0, 1, ...]

cc = fhe.GenCryptoContext(params)
for f in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE):
    cc.Enable(f)

keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalRotateKeyGen(keys.secretKey, [1, -1])
cc.LoadContext(keys.publicKey)            # moves keys to the GPU; generate every key before it

x = [1.0, 2.0, 3.0, 4.0]
ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(x))
ct = cc.EvalRotate(cc.EvalMult(ct, ct), 1)

pt = cc.Decrypt(keys.secretKey, ct)
pt.SetLength(len(x))
print(pt.GetRealPackedValue())            # [4, 9, 16, ...]
```

Everything that creates keys has to run before `LoadContext()`, which uploads them to the GPU.
That includes `EvalMultKeyGen`, `EvalRotateKeyGen`, `EvalBootstrapSetup`, `EvalFEFuncBootstrapSetup` and
`EvalBootstrapKeyGen`, and they all raise if called after it.

Things that differ from openfhe-python:
- **Do not `import openfhe` in the same process.** The two embed different OpenFHE versions.
- **`EvalSum`, `EvalSumRows` and `EvalSumCols` take no key map.** Generate their keys with
  `EvalSumKeyGen`, `EvalSumRowsKeyGen(sk, rowSize)` and `EvalSumColsKeyGen` before
  `LoadContext()`; the keygens return nothing and the keys stay in the context. The GPU results
  are bit-identical to OpenFHE 1.6.1, quirks included: `EvalSumRows`' second argument is the row
  size (OpenFHE calls it `numRows`) and the sum covers the whole ring, so with sparse packing each
  column sum comes out multiplied by `(N/2)/slots`. The sizes passed to `EvalSum` and `EvalSumCols`
  must be powers of two: as in OpenFHE, the fold otherwise sums the next power of two terms.
  `EvalSumCols` consumes a level (mask multiplication). `subringDim` must be 0 or the cyclotomic order. For a sum over an arbitrary
  stride, `cc.AccumulateSum(ct, n, stride)` with keys from `fhe.accumulate_rotation_indices(n, stride)`
  remains available.
- **Running out of GPU memory kills the process.** No exception is raised.
- **Bootstrapping requires sufficient depth, compatible scaling parameters and matching
  precomputation.** Start from `examples/04_bootstrap.py`; StC-first needs more levels
  remaining on entry than the standard pipeline.

For a convolution with scalar weights shared across slots, combine rotated ciphertexts
with the GPU weighted-sum primitive:

```python
ctxs = cc.EvalFastRotation(ct, [-1, 0, 1])
weights = [0.25, 0.5, 0.25]
out = ct.Clone()
out.evalLinearWSumMutable(len(ctxs), ctxs, weights)
cc.RescaleInPlace(out)
```

`evalLinearWSumMutable(n, ctxs, weights)` overwrites its receiver with the weighted sum
of the first `n` inputs, independently in each slot. It returns `None`, releases the GIL,
and requires a GPU context. Inputs must belong to the receiver's context, use the same
key, and have noise scale degree 1. The receiver must have no more remaining levels
than any input (its Python `GetLevel()` must be at least theirs). Other input ciphertexts
are preserved; the receiver can itself be an input. The result has noise scale degree 2
and is **not rescaled automatically**: call `cc.RescaleInPlace(out)` before another
weighted sum. Offloaded inputs are reloaded automatically. Rotations use cyclic slot
boundaries; padding/masking for a convolution must be handled separately.

`cc.EvalFastRotation(ct, indices)` returns a list of GPU hoisted rotations in the same
order as `indices`, sharing the decomposition work across rotations. It preserves the
input and returns independent copies for zero rotations. Generate keys for every
nonzero rotation before `LoadContext()`. No precomputation handle is required on GPU.

For weights or masks that differ between slots, use plaintext dot products instead:

```python
pts = [cc.MakeCKKSPackedPlaintext(w, level=ct.GetLevel()) for w in slot_weights]
out = ct.Clone()
out.dotProductPt(ctxs, pts)               # sum_i ctxs[i] * pts[i], slot by slot
cc.RescaleInPlace(out)
```

`dotProductPt(ctxs, pts)` overwrites its receiver and returns `None`. Both lists must
have the same positive length. All operands must have the same context and level,
and inputs must have noise scale degree 1; ciphertext inputs must use the same key.
The receiver can also be an input. The result has degree 2, without automatic rescale.
Both ciphertext and plaintext cache budgets are supported.

`cc.ConvolutionTransformInPlace(ct, gStep, bStep, pts, indexes, stride=1, rowSize=0)`
combines baby-step rotations and plaintext products, then rotates and sums giant-step
groups. Let `R_k` be a left cyclic rotation by `k` slots. Its packing formula is:

```text
out = sum_j R_(stride*(gStep-j))(sum_i pts[j*bStep+i] * R_indexes[i](ct))
```

The plaintext weights are rotated along with each inner sum. `indexes` has `bStep`
entries, `pts` has `gStep*bStep` entries, and `rowSize` must be 0 or that same product.
The binding requires a positive power-of-two `gStep`: the current upstream tree
reduction can lose contributions for other even sizes. Use
`fhe.convolution_rotation_indices(gStep, bStep, indexes, stride)` to generate **all**
required rotation keys before `LoadContext()`, including the giant-step rotations.
Plaintexts need degree 1 and the input level. If the input has degree 2, the transform
rescales it first, so encode the plaintexts at `ct.GetLevel()+1`. The transform returns
`None`, preserves Python aliases of `ct`, and leaves a degree-2 result requiring an
explicit final `cc.RescaleInPlace(ct)`. These three operations require a GPU context
and release the GIL. See `examples/05_convolution.py` for a complete example.

Run `python tests/test_convolution_ops.py` for numerical GPU checks, including the
weighted-sum tests. Set `WSUM_SCALING=FIXEDMANUAL` to test manual scaling, or
`CUDA_VISIBLE_DEVICES=0,1 WSUM_DEVICES=0,1` to run on two GPUs.

Additional arithmetic helpers work on both CPU and GPU and release the GIL:

| Method | Behavior |
|---|---|
| `cc.EvalAddMany(ciphertexts)` | Sum a nonempty list, preserving inputs; a singleton returns a clone. |
| `cc.EvalAddManyInPlace(ciphertexts)` | Store the sum in `ciphertexts[0]`; other entries may also change. Repeated ciphertext objects are rejected. |
| `cc.EvalSquareInPlace(ct)` | Square `ct`, preserving its Python object identity. |
| `cc.EvalNegateInPlace(ct)` | Negate `ct`, preserving its Python object identity. |
| `cc.SetLevel(ct, level)` | Drop `ct` to a larger Python level index; cannot restore consumed levels. |

The in-place helpers return `None`. Inputs must belong to the calling context;
ciphertexts being added must use the same key. Squaring and negation follow the scaling
behavior of the existing `EvalSquare` and `EvalNegate`. Negation preserves the level,
scaling factor and noise scale degree. `SetLevel` only drops modulus towers: it preserves
the scaling factor and noise scale degree, accepts levels from `ct.GetLevel()` through
the context's multiplicative depth, and does not perform a rescale. Use
`python tests/test_arithmetic_helpers.py` to check these bindings, including CPU behavior;
the same `WSUM_SCALING` and `WSUM_DEVICES` settings apply.

## Functional bootstrapping

Fourier-extension functional bootstrapping (FE-FBT) refreshes a ciphertext and evaluates
a real function in the same pass, on CPU or GPU. Names, keyword arguments and defaults
follow the [official openfhe-python bindings](https://github.com/openfheorg/openfhe-python/blob/main/src/lib/pke/fbt.cpp).

```python
budget = [3, 2]
# Exact series for cos(pi*m) + 0.25*cos(5*pi*m), m in [-0.5, 0.5).
coefficients = [0, 0.5, 0, 0, 0, 0.125]
depth = fhe.CryptoContext.GetFEFBTDepth(budget, coefficients, fhe.SPARSE_ENCAPSULATED) + 6
# Create cc with this depth, ScalingModSize=59, FirstModSize=60, HYBRID,
# SPARSE_ENCAPSULATED, and enable PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE.
cc.EvalFEFuncBootstrapSetup(levelBudget=budget, dim1=[0, 0], slots=slots)
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalBootstrapKeyGen(keys.secretKey, slots)
cc.LoadContext(keys.publicKey)  # omit for CPU, with SetDevices([])

pt = cc.MakeCKKSPackedPlaintext(x, level=depth - (budget[1] + 1), slots=slots)
ct = cc.Encrypt(keys.publicKey, pt)
result = cc.EvalFEFuncBootstrap(ciphertext=ct, coefficients=coefficients)

# Share the refresh across functions; the longest series must have degree >= 5.
powers = cc.EvalFEFuncBootstrapPrecompute(ciphertext=ct, coefficients=coefficients)
same = cc.EvalFEFuncBootstrapWithPrecomp(powers=powers, coefficients=coefficients)
sine = cc.EvalFEFuncBootstrapWithPrecomp(powers=powers, coefficients=[0, -0.5j])
```

`coefficients` accepts real or complex sequences, with `c_0` first. These are Fourier
extension coefficients: the output is `2*Re(sum(c_j*exp(pi*i*j*m)))`. For a function
on `[-B, B]`, encode `m = x/(2*B)` in `[-0.5, 0.5)` and generate its coefficients with
[fefbt-python](https://github.com/openfheorg/fefbt-python). The input ciphertext is
preserved; output slots are real, including with `COMPLEX` input (imaginary parts are
discarded). Inputs outside this interval do not raise an error and give incorrect results.

Setup requires `FirstModSize == ScalingModSize + 1`, two entries in each budget/dimension
vector, and `slots=0` (full packing) or a power of two up to `N/2`. Run it before
`LoadContext()`. FE-FBT and ordinary bootstrapping reserve different precomputations
and cannot share a slot count. The input needs enough levels for SlotsToCoeffs; the
example above encodes at its depletion level. `GetFEFBTDepth` defaults to
`skd=SPARSE_TERNARY, firstModSize=60` as upstream does; pass the context's actual
distribution, preferably `SPARSE_ENCAPSULATED`, and add a margin for later operations.

`SeriesPowers` is an opaque reusable handle tied to its context and CPU/GPU backend.
Recompute it after switching to GPU. It keeps its native context alive; release the
handle to free the powers. GPU powers remain in VRAM outside `SetCiphertextCache`;
bootstrap matrices and keys respect `SetBootstrapCache` and `SetRotationKeyCache`.
All five calls release the GIL. This fork implements FE-FBT; LUT-based `EvalFBT`,
composite scaling and iterative FE-FBT are not supported.

Run `python examples/05_functional_bootstrap.py` (or add `--cpu`) for a complete example.
`OMP_NUM_THREADS=8 python tests/test_functional_bootstrap.py` checks exact CPU/GPU
ciphertext parity, decryption, shared powers, cache reloads and invalid arguments.
The example and tests use small, non-secure parameters. In production select a security
level and let OpenFHE size the ring; sharing decrypted CKKS outputs also requires an
appropriate noise flooding policy, which this wrapper does not apply automatically.

## FIDESlib 2.2.0 operations

`cc.EvalBootstrap(ct, numIterations=1, precision=0, prescaled=False)` supports one or two
iterations. The second bootstraps the first pass's error to refine precision, at the cost
of another pass. `precision` is the estimated precision in bits of one pass, used by the
correction; it is ignored for a single iteration. `cc.EvalBootstrapInPlace(...)` keeps
Python aliases of its ciphertext valid and uses the native in-place GPU path.

The new StC-first pipeline runs SlotsToCoeffs before ModRaise, followed by CoeffsToSlots
and approximate modular reduction. It requires its own setup:

```python
cc.EvalBootstrapSetup([3, 3], [0, 0], slots, btsfirstboot=True)
cc.EvalBootstrapKeyGen(keys.secretKey, slots)
cc.LoadContext(keys.publicKey)
# ct must still have enough levels for the initial SlotsToCoeffs transform.
fresh = cc.EvalBootstrapStCFirst(ct, numIterations=2, precision=12)
# EvalBootstrap / EvalBootstrapInPlace also dispatch according to this setup.
```

The budget order remains `[CoeffsToSlots, SlotsToCoeffs]`. Standard and StC-first cannot
share precomputation for the same slot count in one context. StC-first must run earlier
in the multiplication chain; the example computes its input level explicitly. Try
`python examples/04_bootstrap.py --stc-first --iterations 2 --precision 12`.

For complex data, call `params.SetCKKSDataType(fhe.COMPLEX)` **before** `GenCryptoContext`.
`GetCKKSDataType()` returns `fhe.REAL` (the default) or `fhe.COMPLEX`.
`MakeCKKSPackedPlaintext` accepts real or complex sequences; read complex results with
`pt.GetCKKSPackedValue()`. `MakeCKKSPackedPlaintextF64` remains a real-array fast path.
Ciphertexts expose `GetSlots()` and `GetScalingFactor()` as well as level and noise degree.

`MultByMonomial(ct, power)` and `MultByMonomialInPlace(ct, power)` mirror OpenFHE's
`SchemeBase::MultByMonomial`: they multiply by `X^power` modulo `X^N + 1`, with `power`
taken mod `2N`. With `N = cc.GetRingDimension()`, `N // 2` multiplies every slot by `i`
and `3 * N // 2` by `-i`. This is exact: no level, scale or noise cost and no keys,
unlike `EvalMult` with a plaintext of `i`, which costs a level. On GPU, a monomial is
cached per power and rebuilt when the ciphertext level changes, so prefer calling it at a
consistent level. `python tests/test_mult_by_monomial.py` checks the slot factors and
CPU/GPU bit-exactness.

`EvalConjugate(ct)` maps every slot from `a + ib` to `a - ib` on the GPU
(`CKKS::Ciphertext::conjugate`: the automorphism `2N - 1` followed by a key switch). It
returns an independent ciphertext with the input's level, noise scale degree and scaling
factor, and leaves the input untouched. It needs a loaded GPU context and the conjugation
key, which `EvalBootstrapSetup` and `EvalBootstrapKeyGen` before `LoadContext` provide;
otherwise it raises (`RuntimeError` for a CPU or unloaded context, `ValueError` for a
missing key). With two real vectors packed as `z = a + ib`, `(z + conj(z)) / 2` gives `a`
and `(z - conj(z)) / (2i)` gives `b`, where the division by `i` is
`MultByMonomial(·, 3 * N // 2)`. `python tests/test_eval_conjugate.py` checks the values,
metadata, aliasing and error cases.

`EvalAddMutable`, `EvalSubMutable` and `EvalMultMutable` accept ciphertext/ciphertext,
ciphertext/plaintext and plaintext/ciphertext pairs. They return a result and may modify
operand scales/levels, visible through existing Python aliases. Their `*MutableInPlace`
forms accept two ciphertexts, overwrite the first, may adjust the second, and return
`None`. `EvalSquareMutable(ct)` also returns a result and may adjust its input. Clone
inputs first if their state must be preserved. These operations work on CPU and GPU.
With `FIXEDMANUAL`, align scales/noise degrees explicitly before addition or subtraction,
using `RescaleInPlace` as needed, just as with the corresponding OpenFHE operations.

To export a computed ciphertext without a secret key:

```python
cc.UnloadCiphertext(result)                 # materialize CPU data and release the GPU handle
assert fhe.SerializeToFile("result.bin", result, fhe.BINARY)
restored = cc.DeserializeCiphertextFromFile("result.bin", fhe.BINARY)
```

`SerializeToFile` also works directly on loaded or offloaded ciphertexts and returns a
boolean. Deserialization returns a host ciphertext attached to the calling context and
raises on failure; the file must use compatible context parameters. `fhe.JSON` is also
supported. `UnloadCiphertext` differs from `ct.Offload()`: it produces the ordinary CPU
ciphertext used for export, while offload retains a snapshot for GPU reload.

Run `python tests/test_upstream_220.py` for CPU/GPU mutable arithmetic, complex data,
serialization, and 19 isolated bootstrap configurations. Select GPUs through
`CUDA_VISIBLE_DEVICES`; for two GPUs also set `PYFIDES_TEST_DEVICES=0,1`.

The `examples/` directory has, in order:
- `00_onboarding.py`, a first walkthrough;
- `01_chebyshev.py`, `EvalChebyshevSeries`;
- `02_step_herminirocket.py`, an inference step;
- `03_offload.py`, manual offload;
- `04_bootstrap.py`, bootstrapping and the parameters it requires;
- `05_convolution.py`, three ways to evaluate a cyclic convolution.

## GPU plaintext encoding

Contexts made by `GenCryptoContext` finish CKKS plaintext encoding on the GPU. Turn it off with
`cc.SetGPUPlaintextEncoding(False)`; `cc.GetGPUPlaintextEncoding()` reports the setting.
`MakeCKKSPackedPlaintext` and `MakeCKKSPackedPlaintextF64` then only run the part of OpenFHE's
encoding that every RNS limb shares on the CPU: the IFFT over the slots, then scaling and
rounding to `2*slots` integers. That part takes about 1 ms. The plaintext keeps those
integers. Its first use, or its reload after a plaintext-cache eviction, uploads them once.
The GPU then reduces them modulo each prime and takes the NTT. The device plaintext is bit-identical to the
CPU encoding's, so results do not change. Host memory is 8 bytes per coefficient instead of a
full-ring element per limb.

At logN=17, 59-bit primes, FLEXIBLEAUTOEXT and 32768 slots, measured on an H100 with
`OMP_NUM_THREADS=1` (times per plaintext unless noted):

| Limbs | Encoding | `MakeCKKSPackedPlaintextF64`, 1 thread | 513 plaintexts, 16 threads | First load | Reload |
|---|---|---:|---:|---:|---:|
| 28 | CPU | 76.1 ms | 2.83 s | 10.2 ms | 10.2 ms |
| 28 | GPU | 1.4 ms | 0.24 s | 0.48 ms | 0.45 ms |
| 9 | CPU | 26.7 ms | 0.89 s | 3.2 ms | 3.1 ms |
| 9 | GPU | 0.9 ms | 0.05 s | 0.21 ms | 0.19 ms |

Things to know:
- **Encoding can still run in worker threads.** `MakeCKKSPackedPlaintext*` does no GPU work
  and releases the GIL while it encodes. The GPU work happens in the thread that uses the plaintext,
  as uploads did before.
- **`Encrypt` of such a plaintext encodes it on the CPU** at that point, once. Encryption costs
  what it did before; only the moment the encoding is paid changes.
- `pt.IsGPUEncoded()` tells whether a plaintext took this path. A context without devices, and
  any input OpenFHE rejects, uses the CPU encoding with the same errors.

`python tests/test_gpu_plaintext_encoding.py` compares results with both encodings bit for bit.
FIDESlib's `GpuPlaintextEncodingTests` compare the device limbs directly.

## VRAM cache

Four independent caches each cap the VRAM used by one kind of object. Every cache keeps the
objects it uses most in VRAM, moves the others to host RAM and reloads them on demand. Results
do not change.

| Cache | What it covers | When to set it |
|---|---|---|
| `SetRotationKeyCache(bytes)` | rotation keys, bootstrapping ones included | before `LoadContext()` |
| `SetBootstrapCache(bytes)` | bootstrap matrices (CoeffsToSlots / SlotsToCoeffs) | before `LoadContext()` |
| `SetPlaintextCache(bytes)` | plaintexts you create | any time |
| `SetCiphertextCache(bytes)` | ciphertexts | any time |

Example: a bootstrappable context that must fit a small GPU.

```python
GiB = 1 << 30
slots = 1 << 15

keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)
cc.EvalRotateKeyGen(keys.secretKey, [1, 2, 4])
cc.EvalBootstrapSetup([4, 4], [0, 0], slots)
cc.EvalBootstrapKeyGen(keys.secretKey, slots)

cc.SetRotationKeyCache(6 * GiB)           # these two only affect objects built after them,
cc.SetBootstrapCache(6 * GiB)             # so they come before LoadContext()
cc.LoadContext(keys.publicKey)
cc.SetPlaintextCache(4 * GiB)
cc.SetCiphertextCache(8 * GiB)

ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext([0.5] * slots))
cc.PinCiphertext(ct)                      # never evict a hot object
fresh = cc.EvalBootstrap(ct)              # keys and matrices stream in under their budgets

print(cc.GetRotationKeyCacheResidentBytes(), cc.GetBootstrapCacheResidentBytes(),
      cc.GetPlaintextCacheResidentBytes(), cc.GetCiphertextCacheResidentBytes())
```

Each cache also has `Get*Cache()` (`None` means unlimited, the default), `Offload*()` to evict
now, and `Pin*()` (the bootstrap cache has no `Pin*()`). Things to know:

- **Budgets are soft.** An operation keeps everything it is currently using in VRAM, such as
  the keys of a hoisted rotation, a bootstrap stage, or the operands of an operation. Peak VRAM
  is also higher than the sum of the budgets, because of the operations' own working memory.
  Size the budgets from a measured peak. Compression can make bootstrap stages very
  different in size; under a small bootstrap budget, MRU may still reload every stage.
  Use `GetBootstrapCacheLoadedBytes()` to measure the actual transfer traffic.
- **Rotation keys and bootstrap matrices need the budget before `LoadContext()`.** Only objects
  built under a finite budget keep the host copy they reload from. Calling these setters after
  `LoadContext()` only changes the limit for objects that already have one.
- **Host memory.** Evicted keys, matrices and ciphertexts keep snapshots in host RAM.
  Keys and matrices use page-locked buffers for fast reloads. Matrix row padding can make
  these snapshots larger than the compressed GPU payload.
- **Measure with `Get*ResidentBytes()`, not `nvidia-smi`.** Freed VRAM goes back to FIDESlib's
  allocator, not to the driver. `cc.TrimGPUMemoryPool()` returns idle slabs to the driver.
  Counters report owned payload bytes; the allocator rounds small compressed limbs up to
  at least 1 KiB, so its live/reserved bytes can be higher.
- **Shared limits.** Contexts built from identical parameters share one GPU context, and with
  it the rotation-key and bootstrap budgets.
- **Freeing destroyed ciphertexts.** FIDESlib also keeps destroyed ciphertext polynomials for
  reuse. Set `FIDESLIB_AUX_POLY_CACHE_LIMIT=0` before starting Python to turn that off, or call
  `cc.ClearAuxiliaryPolyPool()`.

To move one ciphertext by hand: `ct.Offload()` / `ct.Reload()` / `ct.IsOffloaded()`.

Each cache has its own test, `tests/test_{rotation_key,plaintext,ciphertext,bootstrap}_cache.py --all`.
