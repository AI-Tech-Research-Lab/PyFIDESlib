# fideslib_py

Python bindings (pybind11) for [FIDESlib](https://github.com/CAPS-UMU/FIDESlib), a GPU
implementation of CKKS that interoperates with OpenFHE. The API follows openfhe-python and
covers context setup, key generation, encoding, encryption, leveled arithmetic, rotations,
`EvalChebyshevSeries`, `AccumulateSum` and bootstrapping, on one or more GPUs.

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
inside `build/`. The fork and commit are `FIDESLIB_REPOSITORY` / `FIDESLIB_GIT_TAG` in
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
That includes `EvalMultKeyGen`, `EvalRotateKeyGen`, `EvalBootstrapSetup` and
`EvalBootstrapKeyGen`, and they all raise if called after it.

Things that differ from openfhe-python:
- **Do not `import openfhe` in the same process.** The two embed different OpenFHE versions.
- **`EvalSum` is not available.** Use `cc.AccumulateSum(ct, n, stride=1)` instead, with keys
  from `fhe.accumulate_rotation_indices(n, stride)`. Every index you rotate by needs its own
  rotation key.
- **Running out of GPU memory kills the process.** No exception is raised.
- **Bootstrapping is strict about its parameters**, and it can segfault or decrypt to noise
  instead of raising. Start from `examples/04_bootstrap.py`.

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
behavior of the existing `EvalSquare` and `EvalNegate`: on GPU, negation is implemented
as multiplication by -1, so its noise scale degree can change. `SetLevel` uses FIDESlib's
level adjustment for the selected scaling technique, and accepts levels from
`ct.GetLevel()` through the context's multiplicative depth. Use
`python tests/test_arithmetic_helpers.py` to check these bindings, including CPU behavior;
the same `WSUM_SCALING` and `WSUM_DEVICES` settings apply.

The `examples/` directory has, in order:
- `00_onboarding.py`, a first walkthrough;
- `01_chebyshev.py`, `EvalChebyshevSeries`;
- `02_step_herminirocket.py`, an inference step;
- `03_offload.py`, manual offload;
- `04_bootstrap.py`, bootstrapping and the parameters it requires;
- `05_convolution.py`, three ways to evaluate a cyclic convolution.

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
  Size the budgets from a measured peak.
- **Rotation keys and bootstrap matrices need the budget before `LoadContext()`.** Only objects
  built under a finite budget keep the host copy they reload from. Calling these setters after
  `LoadContext()` only changes the limit for objects that already have one.
- **Host memory.** Evicted keys, matrices and ciphertexts are kept in host RAM, as much as the
  VRAM they would use. Keys and matrices use page-locked memory, so they reload at PCIe speed.
- **Measure with `Get*ResidentBytes()`, not `nvidia-smi`.** Freed VRAM goes back to FIDESlib's
  allocator, not to the driver. `cc.TrimGPUMemoryPool()` returns idle slabs to the driver.
- **Shared limits.** Contexts built from identical parameters share one GPU context, and with
  it the rotation-key and bootstrap budgets.
- **Freeing destroyed ciphertexts.** FIDESlib also keeps destroyed ciphertext polynomials for
  reuse. Set `FIDESLIB_AUX_POLY_CACHE_LIMIT=0` before starting Python to turn that off, or call
  `cc.ClearAuxiliaryPolyPool()`.

To move one ciphertext by hand: `ct.Offload()` / `ct.Reload()` / `ct.IsOffloaded()`.

Each cache has its own test, `tests/test_{rotation_key,plaintext,ciphertext,bootstrap}_cache.py --all`.
