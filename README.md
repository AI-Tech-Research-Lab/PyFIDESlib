# fideslib_py — Python bindings for FIDESlib (CKKS on GPU)

Minimal pybind11 wrapper around [FIDESlib](https://github.com/CAPS-UMU/FIDESlib)
v2.1.2, the CKKS GPU library interoperable with OpenFHE. Exposes the subset of
the API needed for encrypted inference pipelines (HerMiniRocket/PolyMiniRocket):
context setup, key generation, encoding, encrypt/decrypt, leveled arithmetic,
rotations, `EvalChebyshevSeries`, `AccumulateSum` and bootstrapping.

The compiled module **statically embeds FIDESlib and a patched OpenFHE 1.5.1**
(from `~/Fideslib/local`). Its only runtime dependencies are the CUDA runtime
(RPATH'd to `/usr/local/cuda/lib64`) and an NVIDIA GPU.

## Build

```bash
./build.sh                          # builds against ~/HerMiniRocket/.venv python
./build.sh /usr/bin/python3         # or any other interpreter
```

Prereqs (already satisfied on this machine): the FIDESlib local install at
`~/Fideslib/local` (see `~/Fideslib/CLAUDE.md`), CUDA toolkit ≥ 12.4, gcc ≥ 11,
CMake ≥ 3.25, network access for the pybind11 fetch, and the Python dev headers
for the chosen interpreter.

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

## Examples (in suggested order)

| Script | What it shows | Needs |
|---|---|---|
| `examples/00_onboarding.py` | context → keys → encrypt → add/mult/rotate/sum → decrypt | <1 GB VRAM, seconds |
| `examples/01_chebyshev.py` | deg-31 polynomial + X4 cleaning vs CPU Clenshaw reference | ~1 GB VRAM, seconds |
| `examples/02_step_herminirocket.py` | full Step() (Lee α=8 + 2×X4) at logN=17, **secure** params | ~2–4 GB VRAM, ~1–2 min |

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
