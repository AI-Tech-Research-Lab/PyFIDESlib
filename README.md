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

The `examples/` directory has, in order:
- `00_onboarding.py`, a first walkthrough;
- `01_chebyshev.py`, `EvalChebyshevSeries`;
- `02_step_herminirocket.py`, an inference step;
- `03_offload.py`, manual offload;
- `04_bootstrap.py`, bootstrapping and the parameters it requires.

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
