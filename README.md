# fideslib_py — Python bindings for FIDESlib (CKKS on GPU)

Minimal pybind11 wrapper around [FIDESlib](https://github.com/CAPS-UMU/FIDESlib)
v2.1.2, the CKKS GPU library interoperable with OpenFHE. Exposes the subset of
the API needed for encrypted inference pipelines (HerMiniRocket/PolyMiniRocket):
context setup, key generation, encoding, encrypt/decrypt, leveled arithmetic,
rotations, `EvalChebyshevSeries`, `AccumulateSum` and bootstrapping.

The compiled module **statically embeds FIDESlib and a patched OpenFHE 1.5.1**.
FIDESlib is pinned as a git submodule (`external/FIDESlib`) and built from source by
`build.sh`, so the wrapper is tied to an exact FIDESlib commit rather than a
machine-specific install. Its only runtime dependencies are the CUDA runtime
(RPATH'd to `/usr/local/cuda/lib64`) and an NVIDIA GPU.

## Build

```bash
git submodule update --init external/FIDESlib   # fetch the pinned FIDESlib commit

# One-time: build the patched OpenFHE 1.5.1 that FIDESlib depends on (~30 min).
( cd external/FIDESlib/deps && ./build.sh "$PWD/openfhe-install" )

./build.sh                          # builds FIDESlib (from the submodule) + the wrapper
./build.sh /usr/bin/python3         # against a specific interpreter
```

`build.sh` compiles the pinned `external/FIDESlib` submodule into a repo-relative
install (`external/FIDESlib/install`) and links the wrapper against it — no
machine-specific path. Point `OPENFHE_LOCAL=<prefix>` at an existing patched
OpenFHE 1.5.1 install to skip the one-time OpenFHE build.

Prereqs: CUDA toolkit ≥ 12.4, gcc ≥ 11, CMake ≥ 3.25, network access (submodule +
pybind11/OpenFHE fetches), and the Python dev headers for the chosen interpreter.

The module is bound to the Python **minor version** it was built against
(e.g. `_core.cpython-312-x86_64-linux-gnu.so` ⇒ Python 3.12). Rebuild to switch.

## Use

```bash
export PYTHONPATH=$HOME/Fideslib/pyfideslib    # or sys.path.insert / pip install -e
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
